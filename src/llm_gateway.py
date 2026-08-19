import os
import json
import re
from dataclasses import dataclass


MODEL_CLASS_ORDER = ("CHEAP", "STANDARD", "STRONG")


@dataclass
class LLMRequest:
    prompt: str
    stage: str
    requested_model_class: str = "CHEAP"
    reason_for_escalation: str = ""
    task_class: str = ""
    semantic_task: str = ""


class BudgetExceeded(Exception):
    pass


class MissingLLMProvider(Exception):
    pass


class MalformedStructuredOutput(Exception):
    pass


class LLMBudgetManager:
    def __init__(self, max_run_usd=None, max_daily_usd=None, max_strong_calls=None, max_agent_iterations=None):
        self.max_run_usd = float(max_run_usd if max_run_usd is not None else os.environ.get("MAX_RUN_LLM_USD", "0"))
        self.max_daily_usd = float(max_daily_usd if max_daily_usd is not None else os.environ.get("MAX_DAILY_LLM_USD", "0"))
        self.max_strong_calls = int(max_strong_calls if max_strong_calls is not None else os.environ.get("MAX_STRONG_CALLS_PER_RUN", "0"))
        self.max_agent_iterations = int(max_agent_iterations if max_agent_iterations is not None else os.environ.get("MAX_AGENT_ITERATIONS", "20"))

    def can_spend(self, state, estimated_cost, model_class, semantic_task=""):
        from src.budget_control import agent_iteration_usage, authorized_limit
        budget = state.setdefault("budget", {"llm_usd": 0.0, "strong_calls": 0, "calls": []})
        if self.max_run_usd and budget.get("llm_usd", 0.0) + estimated_cost > self.max_run_usd:
            return False, "MAX_RUN_LLM_USD"
        if model_class == "STRONG" and self.max_strong_calls and budget.get("strong_calls", 0) + 1 > self.max_strong_calls:
            return False, "MAX_STRONG_CALLS_PER_RUN"
        semantic_calls = agent_iteration_usage(state)
        effective_limit = authorized_limit(state, "agent_iterations", self.max_agent_iterations)
        if effective_limit and semantic_calls + 1 > effective_limit:
            continuation = state.get("active_external_continuation") or {}
            allowance = continuation.get("downstream_semantic_allowance") or {}
            allowed_tasks = set(allowance.get("semantic_tasks", []))
            used_tasks = set(allowance.get("used_semantic_tasks", []))
            if (continuation.get("status") == "CONTINUATION_STARTED"
                    and semantic_task in allowed_tasks and semantic_task not in used_tasks):
                allowance.setdefault("used_semantic_tasks", []).append(semantic_task)
                allowance.setdefault("authorizations", []).append({
                    "semantic_task": semantic_task,
                    "reason": "EXTERNAL_CONTINUATION_DOWNSTREAM_VALIDATION",
                })
                return True, None
            return False, "MAX_AGENT_ITERATIONS"
        return True, None

    def record(self, state, call):
        budget = state.setdefault("budget", {"llm_usd": 0.0, "strong_calls": 0, "calls": []})
        actual_cost = float(call.get("actual_cost", 0.0) or 0.0)
        budget["llm_usd"] = round(float(budget.get("llm_usd", 0.0)) + actual_cost, 8)
        if "estimated_cost" in call:
            budget["estimated_llm_usd"] = round(float(budget.get("estimated_llm_usd", 0.0)) + float(call.get("estimated_cost", 0.0) or 0.0), 8)
        if call.get("model_class") == "STRONG":
            budget["strong_calls"] = int(budget.get("strong_calls", 0)) + 1
        budget.setdefault("calls", []).append(call)
        if call.get("failure_type") in {"TRANSIENT_LOCAL_RUNTIME_FAILURE", "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"}:
            budget["infrastructure_attempts"] = int(budget.get("infrastructure_attempts", 0)) + 1
            if int(call.get("infrastructure_retry_number", 0) or 0) > 0:
                budget["infrastructure_retries"] = int(budget.get("infrastructure_retries", 0)) + 1
        elif call.get("retry_budget_type") == "semantic":
            budget["semantic_retries"] = int(budget.get("semantic_retries", 0)) + 1
        elif call.get("retry_budget_type") == "structured_output":
            budget["structured_output_repairs"] = int(budget.get("structured_output_repairs", 0)) + 1
        elif call.get("retry_budget_type") == "model_quality_escalation":
            budget["model_quality_escalations"] = int(budget.get("model_quality_escalations", 0)) + 1


class LLMProvider:
    def generate(self, request):
        raise NotImplementedError

    def generate_structured(self, request, schema=None, semantic_validator=None):
        return self.generate(request)


class NullLLMProvider(LLMProvider):
    available = False

    def generate(self, request):
        return {
            "text": "",
            "model": "none",
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
        }


class GeminiLLMProvider(LLMProvider):
    available = True

    def __init__(self, cheap_model=None, standard_model=None, strong_model=None):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.models = {
            "CHEAP": cheap_model or os.environ.get("RESEARCH_LLM_MODEL_CHEAP") or os.environ.get("RESEARCH_LLM_MODEL", "models/gemini-3.1-flash-lite"),
            "STANDARD": standard_model or os.environ.get("RESEARCH_LLM_MODEL_STANDARD") or os.environ.get("RESEARCH_LLM_STANDARD_MODEL", "models/gemini-3.5-flash"),
            "STRONG": strong_model or os.environ.get("RESEARCH_LLM_MODEL_STRONG") or os.environ.get("RESEARCH_LLM_STRONG_MODEL", "models/gemini-2.0-flash"),
        }

    def generate(self, request):
        model = self.models[request.requested_model_class]
        response = self.client.models.generate_content(model=model, contents=request.prompt)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        return {
            "text": response.text,
            "model": model,
            "input_tokens": input_tokens,
            "cached_input_tokens": int(getattr(usage, "cached_content_token_count", 0) or 0),
            "output_tokens": output_tokens,
            "estimated_cost": 0.0,
        }


def provider_from_env():
    provider = os.environ.get("RESEARCH_LLM_PROVIDER", "none").lower()
    if provider in ("none", "null", "disabled"):
        return NullLLMProvider()
    if provider == "gemini":
        return GeminiLLMProvider()
    if provider == "local":
        from src.local_inference import LocalLLMProvider
        fallback = None
        if os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0") == "1":
            remote = os.environ.get("RESEARCH_REMOTE_LLM_PROVIDER", "").lower()
            if remote == "gemini":
                fallback = GeminiLLMProvider()
        return LocalLLMProvider(fallback_provider=fallback)
    raise ValueError(f"unknown RESEARCH_LLM_PROVIDER: {provider}")


class ModelGateway:
    def __init__(self, provider=None, budget_manager=None):
        self.provider = provider or provider_from_env()
        self.budget_manager = budget_manager or LLMBudgetManager()

    def generate(self, state, request, estimated_cost=0.0):
        model_class = request.requested_model_class
        if model_class not in MODEL_CLASS_ORDER:
            raise ValueError(f"invalid model class: {model_class}")
        if model_class == "STRONG" and not request.reason_for_escalation:
            raise ValueError("STRONG model requests require reason_for_escalation")
        ok, reason = self.budget_manager.can_spend(state, estimated_cost, model_class, request.semantic_task)
        if not ok:
            raise BudgetExceeded(reason)
        if not getattr(self.provider, "available", True):
            raise MissingLLMProvider("RESEARCH_LLM_PROVIDER is not configured")
        response = self.provider.generate(request)
        call = {
            "stage": request.stage,
            "model_class": model_class,
            "requested_model_class": request.requested_model_class,
            "actual_model": response.get("model"),
            "reason_for_escalation": request.reason_for_escalation,
            "input_tokens": response.get("input_tokens", 0),
            "cached_input_tokens": response.get("cached_input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "estimated_cost": response.get("estimated_cost", estimated_cost),
            "actual_cost": response.get("actual_cost", 0.0),
        }
        self.budget_manager.record(state, call)
        return response

    def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
        required_keys = required_keys or []
        prompt = request.prompt
        last_error = None
        for attempt in range(max_repairs + 1):
            structured_request = LLMRequest(
                prompt=prompt,
                stage=request.stage,
                requested_model_class=request.requested_model_class,
                reason_for_escalation=request.reason_for_escalation,
                task_class=request.task_class,
                semantic_task=request.semantic_task,
            )
            if type(self.provider).generate_structured is not LLMProvider.generate_structured:
                ok, reason = self.budget_manager.can_spend(
                    state, estimated_cost, structured_request.requested_model_class, structured_request.semantic_task)
                if not ok:
                    raise BudgetExceeded(reason)
                if not getattr(self.provider, "available", True):
                    raise MissingLLMProvider("RESEARCH_LLM_PROVIDER is not configured")
                schema = schema or {
                    "type": "object",
                    "required": required_keys,
                    "properties": {key: {} for key in required_keys},
                }
                try:
                    if semantic_validator is None:
                        response = self.provider.generate_structured(structured_request, schema=schema)
                    else:
                        response = self.provider.generate_structured(structured_request, schema=schema, semantic_validator=semantic_validator)
                except Exception as exc:
                    attempts = getattr(exc, "attempts", None)
                    if attempts:
                        self._record_structured_attempts(state, structured_request, attempts, error=str(exc))
                    else:
                        self.budget_manager.record(state, {
                            "stage": structured_request.stage,
                            "task_class": structured_request.task_class,
                            "semantic_task": structured_request.semantic_task,
                            "model_class": structured_request.requested_model_class,
                            "requested_model_class": structured_request.requested_model_class,
                            "actual_model": getattr(exc, "local_model", None),
                            "reason_for_escalation": structured_request.reason_for_escalation,
                            "input_tokens": len(structured_request.prompt.split()),
                            "cached_input_tokens": 0,
                            "output_tokens": 0,
                            "estimated_cost": estimated_cost,
                            "actual_cost": 0.0,
                            "status": "FAILED",
                            "error": str(exc),
                            "failure_type": classify_llm_failure(exc),
                        })
                    raise
                attempts = response.get("attempts")
                if attempts:
                    self._record_structured_attempts(state, structured_request, attempts)
                else:
                    self.budget_manager.record(state, {
                        "stage": structured_request.stage,
                        "task_class": structured_request.task_class,
                        "semantic_task": structured_request.semantic_task,
                        "model_class": structured_request.requested_model_class,
                        "requested_model_class": structured_request.requested_model_class,
                        "actual_model": response.get("model"),
                        "reason_for_escalation": structured_request.reason_for_escalation,
                        "input_tokens": response.get("input_tokens", 0),
                        "cached_input_tokens": response.get("cached_input_tokens", 0),
                        "output_tokens": response.get("output_tokens", 0),
                        "estimated_cost": response.get("estimated_cost", estimated_cost),
                        "actual_cost": response.get("actual_cost", 0.0),
                        "status": response.get("status", "SUCCESS"),
                        "selected_configuration_id": response.get("selected_configuration_id"),
                        "weight_quantization": response.get("weight_quantization"),
                        "kv_quantization": response.get("kv_quantization"),
                        "local_profile": response.get("local_profile"),
                        "prompt_tokens_per_second": response.get("prompt_tokens_per_second"),
                        "generation_tokens_per_second": response.get("generation_tokens_per_second"),
                        "wall_clock_duration_seconds": response.get("wall_clock_duration_seconds"),
                    })
            else:
                response = self.generate(state, structured_request, estimated_cost=estimated_cost)
            try:
                data = response.get("structured") or parse_json_object(response.get("text", ""))
                missing = [key for key in required_keys if key not in data]
                if missing:
                    raise MalformedStructuredOutput(f"missing required keys: {missing}")
                if semantic_validator:
                    semantic_errors = list(semantic_validator(data) or [])
                    if semantic_errors:
                        raise MalformedStructuredOutput(f"semantic validation failed: {semantic_errors}")
                response["structured"] = data
                return response
            except Exception as exc:
                last_error = exc
                prompt = (
                    "Repair the previous response into one valid JSON object only. "
                    f"Required keys: {required_keys}. Error: {exc}\n\nPrevious response:\n{response.get('text', '')}"
                )
        raise MalformedStructuredOutput(str(last_error))

    def _record_structured_attempts(self, state, request, attempts, error=None):
        for attempt in attempts:
            status = attempt.get("status", "FAILED")
            call = {
                "stage": request.stage,
                "task_class": request.task_class,
                "semantic_task": request.semantic_task,
                "model_class": request.requested_model_class,
                "requested_model_class": request.requested_model_class,
                "actual_model": attempt.get("actual_model") or attempt.get("selected_configuration_id"),
                "reason_for_escalation": request.reason_for_escalation,
                "input_tokens": attempt.get("input_tokens", 0),
                "cached_input_tokens": attempt.get("cached_input_tokens", 0),
                "output_tokens": attempt.get("output_tokens", 0),
                "estimated_cost": attempt.get("estimated_cost", 0.0),
                "actual_cost": attempt.get("actual_cost", 0.0),
                "status": "SUCCESS" if status == "SUCCESS" else "FAILED",
                "failure_type": attempt.get("failure_type"),
                "error": error if status != "SUCCESS" else None,
                "attempt_number": attempt.get("attempt_number"),
                "repair_attempt": attempt.get("repair_attempt", False),
                "selected_configuration_id": attempt.get("selected_configuration_id"),
                "weight_quantization": attempt.get("weight_quantization"),
                "kv_quantization": attempt.get("kv_quantization"),
                "local_profile": attempt.get("local_profile"),
                "prompt_tokens_per_second": attempt.get("prompt_tokens_per_second"),
                "generation_tokens_per_second": attempt.get("generation_tokens_per_second"),
                "wall_clock_duration_seconds": attempt.get("wall_clock_duration_seconds"),
                "raw_response": attempt.get("raw_response"),
                "command": attempt.get("command"),
                "stdout": attempt.get("stdout"),
                "stderr": attempt.get("stderr"),
                "exit_code": attempt.get("exit_code"),
                "assistant_generation_text": attempt.get("assistant_generation_text"),
                "isolated_assistant_text": attempt.get("isolated_assistant_text"),
                "extraction_strategy": attempt.get("extraction_strategy"),
                "extraction_diagnostics": attempt.get("extraction_diagnostics"),
                "schema": attempt.get("schema"),
                "parsed_response": attempt.get("parsed_response"),
                "schema_errors": attempt.get("schema_errors", []),
                "semantic_errors": attempt.get("semantic_errors", []),
                "infrastructure_errors": attempt.get("infrastructure_errors", []),
                "infrastructure_retry_number": attempt.get("infrastructure_retry_number", 0),
                "infrastructure_retry_budget": attempt.get("infrastructure_retry_budget", 0),
                "retry_budget_type": attempt.get("retry_budget_type"),
                "retry_reason": attempt.get("retry_reason"),
                "subprocess_started": attempt.get("subprocess_started", False),
                "generation_began": attempt.get("generation_began", False),
                "generation_produced": attempt.get("generation_produced", False),
                "resource_diagnostics": attempt.get("resource_diagnostics", {}),
            }
            self.budget_manager.record(state, call)


def parse_json_object(text):
    cleaned = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(cleaned):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise


def classify_llm_failure(exc):
    name = exc.__class__.__name__
    message = str(exc)
    if name == "LocalRuntimeInfrastructureFailure" or "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE" in message:
        return "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE"
    if name == "UnknownLLMTaskClass" or "UNKNOWN_LLM_TASK_CLASS" in message:
        return "UNKNOWN_TASK_CLASS"
    if name == "NoEligibleLocalModel" or "NO_ELIGIBLE_LOCAL_MODEL" in message:
        return "NO_ELIGIBLE_CONFIGURATION"
    if "profile" in message.lower():
        return "NO_PROFILE_AVAILABLE"
    if name == "StructuredDecodingConfigurationFailure" or "STRUCTURED_DECODING_CONFIGURATION_FAILURE" in message:
        return "STRUCTURED_DECODING_CONFIGURATION_FAILURE"
    if "semantic validation" in message or "SEMANTIC_VALIDATION_FAILURE" in message:
        return "SEMANTIC_VALIDATION_FAILURE"
    if "structured generation" in message or "schema validation" in message:
        return "SCHEMA_VALIDATION_FAILURE"
    if name == "StructuredGenerationExhausted":
        return "MODEL_OUTPUT_INVALID"
    return "MODEL_EXECUTION_FAILED"
