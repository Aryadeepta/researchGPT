import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.llm_gateway import LLMRequest, ModelGateway
from src.local_inference import (
    CapabilityBenchmark,
    ContextReducer,
    HardwareProbe,
    KVCachePolicy,
    LocalInferenceManager,
    LocalRuntimeInfrastructureFailure,
    LocalInferenceBackendIncompatible,
    LocalLLMProvider,
    LocalModelRouter,
    LlamaCppRuntime,
    ModelRegistry,
    QuantizationPolicy,
    RuntimeRegistry,
    STRUCTURED_SMOKE_SCHEMA,
    StructuredGenerationExhausted,
    estimate_fit,
    extract_llama_assistant_response,
    render_programmatic_llama_prompt,
    infer_quantization_from_name,
    is_structured_decoding_error,
    json_parse_mode,
    parse_json_object,
    classify_local_infrastructure_failure,
    probe_loopback_bind,
    structured_generation_config,
)


def loopback_available():
    return {"status": "LOOPBACK_BIND_AVAILABLE", "host": "127.0.0.1", "port": 12345,
            "errno": None, "message": None}


def loopback_unavailable():
    return {"status": "LOOPBACK_BIND_UNAVAILABLE", "host": "127.0.0.1", "port": 0,
            "errno": 13, "message": "permission denied"}


class FakeRuntime:
    name = "fake-runtime"

    def discover(self):
        return {"runtime": self.name, "available": True, "capabilities": {"kv_quantization": True, "turboquant": False}}

    def supports(self, model):
        return model.get("format") == "gguf"

    def generate(self, prompt, config):
        return {"text": '{"ok": true}', "duration_seconds": 0.01, "model": config["id"]}


class DiagnosticRuntime(FakeRuntime):
    def __init__(self):
        self.calls = []

    def run_cli(self, prompt, config):
        self.calls.append({"prompt": prompt, "config": config})
        return {
            "command": ["/tmp/llama", "cli", "-m", config["model"]["path"]],
            "exit_code": 0,
            "stdout": '{"ok":true}\n',
            "stderr": "llama_perf_context_print: prompt eval time = 10.00 ms / 5 tokens (500.00 tokens per second)\nllama_perf_context_print: eval time = 20.00 ms / 4 runs (200.00 tokens per second)\n",
            "wall_clock_duration_seconds": 0.03,
            "parsed_assistant_output": '{"ok":true}',
            "parse_status": "SUCCESS",
            "status": "SUCCESS",
            "timing_data": {
                "prompt_tokens_per_second": 500.0,
                "generation_tokens_per_second": 200.0,
                "raw_timing_output": "",
            },
        }


class EmptyRuntime(FakeRuntime):
    def run_cli(self, prompt, config):
        return {
            "command": ["/tmp/llama"],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "wall_clock_duration_seconds": 0.01,
            "parsed_assistant_output": "",
            "parse_status": "EMPTY_OUTPUT",
            "status": "EMPTY_OUTPUT",
            "failure_class": "MODEL_OUTPUT_FAILURE",
            "timing_data": {},
        }


class ThinkingProseRuntime(DiagnosticRuntime):
    def run_cli(self, prompt, config):
        self.calls.append({"prompt": prompt, "config": config})
        return {"command": ["/tmp/llama-cli"], "exit_code": 0, "stdout": "</think>\nOkay, let's reason.",
                "stderr": "", "wall_clock_duration_seconds": 0.01,
                "parsed_assistant_output": "</think>\nOkay, let's reason.",
                "parse_status": "ASSISTANT_OUTPUT_EXTRACTION_FAILURE",
                "status": "ASSISTANT_OUTPUT_EXTRACTION_FAILURE", "timing_data": {}}


class EnvironmentFailureRuntime(FakeRuntime):
    def run_cli(self, prompt, config):
        return {
            "command": ["/tmp/llama"],
            "exit_code": 134,
            "stdout": "Loading model...",
            "stderr": "failed to get a free port",
            "wall_clock_duration_seconds": 0.01,
            "parsed_assistant_output": "",
            "parse_status": "EXECUTION_FAILED",
            "status": "EXECUTION_FAILED",
            "failure_class": "RUNTIME_ENVIRONMENT_FAILURE",
            "timing_data": {},
        }


class SequenceRuntime(FakeRuntime):
    name = "llama.cpp"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompt, config):
        self.calls.append({"prompt": prompt, "config": config})
        text = self.outputs.pop(0)
        if isinstance(text, Exception):
            raise text
        return {
            "text": text,
            "duration_seconds": 0.01,
            "model": config["id"],
            "diagnostic": {"timing_data": {"prompt_tokens_per_second": 10, "generation_tokens_per_second": 20}},
        }


class FeasibilityBenchmarkRuntime(FakeRuntime):
    def run_cli(self, prompt, config):
        if "verdict" in prompt or "required evidence" in prompt:
            text = '{"verdict":"FEASIBLE","evidence_requirements":["raw records"],"resource_constraints":["local python"],"validation_plan":["recompute output"]}'
        else:
            text = '{"ok":true}' if '{"ok": true}' in prompt or '{"ok":true}' in prompt else '{"steps":["one","two","three"],"code":"open(\\"out.json\\",\\"w\\").write(\\"{}\\")","claim_supported":true,"evidence_id":"A","flaws":["test with held-out evidence"]}'
        return {
            "command": ["/tmp/llama"],
            "exit_code": 0,
            "stdout": text,
            "stderr": "llama_perf_context_print: prompt eval time = 10.00 ms / 5 tokens (500.00 tokens per second)\nllama_perf_context_print: eval time = 20.00 ms / 4 runs (200.00 tokens per second)\n",
            "wall_clock_duration_seconds": 0.03,
            "parsed_assistant_output": text,
            "parse_status": "SUCCESS",
            "status": "SUCCESS",
            "timing_data": {"prompt_tokens_per_second": 500.0, "generation_tokens_per_second": 200.0, "raw_timing_output": ""},
        }


def write_measured_profile(root, q3=True, q4=True):
    profile_dir = Path(root) / "local_inference"
    profile_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    if q3:
        configs.append({
            "id": "llama.cpp:local-default:Q3_K_M:q4:1024",
            "model": {"id": "local-default", "path": "/tmp/q3.gguf", "format": "gguf", "quantization": "Q3_K_M", "profile": "DEFAULT"},
            "context": 1024,
            "kv_quantization": {"id": "q4"},
            "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"],
            "measured_capability": {"structured_output": 0.9, "planning": 0.8},
            "ram_estimate": 1,
        })
    if q4:
        configs.append({
            "id": "llama.cpp:local-quality:Q4_K_M:q4:1024",
            "model": {"id": "local-quality", "path": "/tmp/q4.gguf", "format": "gguf", "quantization": "Q4_K_M", "profile": "QUALITY"},
            "context": 1024,
            "kv_quantization": {"id": "q4"},
            "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"],
            "measured_capability": {"structured_output": 1.0, "planning": 0.9},
            "ram_estimate": 2,
        })
    (profile_dir / "profile.json").write_text(json.dumps({
        "configurations": configs,
        "routing_profile": {"task_assignments": {"candidate_question_generation": configs[0]["id"] if configs else "HUMAN_REQUIRED"}},
    }), encoding="utf-8")


def local_structured_provider(root, outputs):
    runtime = SequenceRuntime(outputs)

    class Registry:
        def discover(self):
            return [runtime.discover()]
        def first_supporting(self, model):
            return runtime

    manager = LocalInferenceManager(root=root, runtime_registry=Registry())
    return LocalLLMProvider(manager=manager), runtime


class FakeDownloadRuntime:
    executable = "/tmp/llama"

    def download_command(self, args):
        return [self.executable, "download", *args]


class FakeHardwareProbe:
    def persist(self, root):
        return {
            "available_ram_bytes": 16 * 1024**3,
            "system_ram_bytes": 16 * 1024**3,
            "gpus": [],
            "supported_compute_backends": ["cpu"],
        }


class FallbackProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return {"text": "remote", "model": "remote", "input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1, "estimated_cost": 0.01}


class LocalInferenceTests(unittest.TestCase):
    @staticmethod
    def _port_failure():
        exc = RuntimeError("failed to get a free port")
        exc.diagnostic = {"command": ["/tmp/llama", "cli"], "exit_code": 1, "stdout": "", "stderr": "failed to get a free port",
                          "status": "EXECUTION_FAILED", "assistant_generation_text": "", "parsed_assistant_output": "",
                          "wall_clock_duration_seconds": 0.004, "resource_diagnostics": {"internal_router": True}}
        return exc

    def test_port_failure_is_infrastructure_and_retries_original_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, runtime = local_structured_provider(d, [self._port_failure(), '{"ok":true}'])
            with unittest.mock.patch.dict(os.environ, {"RESEARCH_LOCAL_INFRASTRUCTURE_RETRIES": "1",
                                                       "RESEARCH_LOCAL_INFRASTRUCTURE_RETRY_DELAY_SECONDS": "0"}):
                result = provider.generate_structured(
                    LLMRequest("ORIGINAL SEMANTIC TASK", "question_refinement", task_class="candidate_question_generation"),
                    schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
            self.assertEqual(len(runtime.calls), 2)
            self.assertEqual(runtime.calls[0]["prompt"], runtime.calls[1]["prompt"])
            self.assertNotIn("Repair the previous response", runtime.calls[1]["prompt"])
            self.assertEqual(result["attempts"][0]["failure_type"], "TRANSIENT_LOCAL_RUNTIME_FAILURE")
            self.assertFalse(result["attempts"][0]["repair_attempt"])
            self.assertEqual(result["attempts"][0]["infrastructure_retry_number"], 0)
            self.assertEqual(result["attempts"][1]["infrastructure_retry_number"], 1)
            self.assertFalse(result["attempts"][1]["repair_attempt"])

    def test_persistent_port_failure_does_not_escalate_model_or_semantic_budget(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, runtime = local_structured_provider(d, [self._port_failure(), self._port_failure()])
            with unittest.mock.patch.dict(os.environ, {"RESEARCH_LOCAL_INFRASTRUCTURE_RETRIES": "1",
                                                       "RESEARCH_LOCAL_INFRASTRUCTURE_RETRY_DELAY_SECONDS": "0"}):
                with self.assertRaises(LocalRuntimeInfrastructureFailure) as ctx:
                    provider.generate_structured(
                        LLMRequest("ORIGINAL SEMANTIC TASK", "question_refinement", task_class="candidate_question_generation"),
                        schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
            self.assertEqual(len(runtime.calls), 2)
            self.assertTrue(all(call["config"]["id"].endswith("Q3_K_M:q4:1024") for call in runtime.calls))
            self.assertTrue(all(call["prompt"] == "ORIGINAL SEMANTIC TASK" for call in runtime.calls))
            self.assertTrue(all(not attempt["repair_attempt"] for attempt in ctx.exception.attempts))
            self.assertTrue(all(attempt["retry_budget_type"] in {"initial", "infrastructure"} for attempt in ctx.exception.attempts))

    def test_gateway_records_infrastructure_attempts_without_schema_repair(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, _ = local_structured_provider(d, [self._port_failure(), self._port_failure()])
            state = {"budget": {"llm_usd": 0.0, "strong_calls": 0, "calls": []}}
            gateway = ModelGateway(provider=provider)
            with unittest.mock.patch.dict(os.environ, {"RESEARCH_LOCAL_INFRASTRUCTURE_RETRIES": "1",
                                                       "RESEARCH_LOCAL_INFRASTRUCTURE_RETRY_DELAY_SECONDS": "0"}):
                with self.assertRaises(LocalRuntimeInfrastructureFailure):
                    gateway.generate_structured(state,
                        LLMRequest("ORIGINAL", "question_refinement", task_class="candidate_question_generation"),
                        schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
            self.assertEqual([c["failure_type"] for c in state["budget"]["calls"]],
                             ["TRANSIENT_LOCAL_RUNTIME_FAILURE", "TRANSIENT_LOCAL_RUNTIME_FAILURE"])
            self.assertTrue(all(c["schema_errors"] == [] and c["semantic_errors"] == [] for c in state["budget"]["calls"]))
            self.assertTrue(all(not c["generation_began"] and not c["generation_produced"] for c in state["budget"]["calls"]))
            self.assertEqual(state["budget"]["infrastructure_attempts"], 2)
            self.assertEqual(state["budget"]["infrastructure_retries"], 1)
            self.assertEqual(state["budget"].get("semantic_retries", 0), 0)
            self.assertEqual(state["budget"].get("structured_output_repairs", 0), 0)
            self.assertEqual(state["budget"].get("model_quality_escalations", 0), 0)

    def test_generated_text_prevents_infrastructure_misclassification(self):
        diagnostic = {"stderr": "failed to get a free port", "exit_code": 1,
                      "assistant_generation_text": '{"ok":', "parsed_assistant_output": ""}
        self.assertIsNone(classify_local_infrastructure_failure(diagnostic))
    def _fake_executable(self, directory, name):
        path = Path(directory) / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def test_hardware_detection(self):
        profile = HardwareProbe().probe()
        self.assertIn("logical_cores", profile)
        self.assertIn("available_ram_bytes", profile)
        self.assertIn("supported_compute_backends", profile)

    def test_low_memory_model_rejection(self):
        hardware = {"available_ram_bytes": 2 * 1024**3, "system_ram_bytes": 2 * 1024**3, "gpus": []}
        model = {"estimated_size_bytes": 10 * 1024**3, "hidden_size": 4096, "layers": 24}
        fit = estimate_fit(model, hardware, context=4096, kv_cache={"bits": 16}, ram_fraction=0.5)
        self.assertFalse(fit["fits_ram"])

    def test_quantization_metadata(self):
        q = infer_quantization_from_name("model-IQ3_XS.gguf")
        self.assertTrue(q.importance_aware)
        self.assertLessEqual(q.bits_per_weight, 3)

    def test_model_fit_calculation(self):
        hardware = {"available_ram_bytes": 16 * 1024**3, "system_ram_bytes": 16 * 1024**3, "gpus": []}
        model = {"estimated_size_bytes": 2 * 1024**3, "hidden_size": 1024, "layers": 12}
        self.assertTrue(estimate_fit(model, hardware, context=1024, kv_cache={"bits": 4}, ram_fraction=0.6)["fits_ram"])

    def test_task_quality_thresholds(self):
        caps = {"structured_output": 0.9, "planning": 0.7, "coding": 0.4}
        eligible = CapabilityBenchmark().eligible_tasks(caps)
        self.assertIn("metadata_extraction", eligible)
        self.assertNotIn("skill_code_generation", eligible)

    def test_weak_configuration_cannot_handle_higher_tier_task(self):
        router = LocalModelRouter({"configurations": [{
            "id": "weak",
            "ram_estimate": 1,
            "tokens_per_second": 10,
            "eligible_task_classes": ["metadata_extraction"],
        }]})
        self.assertIsNone(router.select("STANDARD", "skill_code_generation"))

    def test_automatic_local_escalation(self):
        profile = {"configurations": [
            {"id": "tiny", "ram_estimate": 1, "tokens_per_second": 20, "eligible_task_classes": ["metadata_extraction"]},
            {"id": "small", "ram_estimate": 2, "tokens_per_second": 10, "eligible_task_classes": ["candidate_question_generation", "research_feasibility_analysis", "skill_code_generation", "adversarial_criticism"]},
        ]}
        self.assertEqual(LocalModelRouter(profile).select("STANDARD", "skill_code_generation")["id"], "small")

    def test_kv_compression_separate_from_weights(self):
        kvs = KVCachePolicy("aggressive").candidates({"kv_quantization": True})
        self.assertTrue(any(kv.id == "q4" for kv in kvs))
        q = QuantizationPolicy("aggressive").candidates()[0]
        self.assertNotEqual(kvs[0].method, q.method)

    def test_provider_fallback_after_local_failure(self):
        with tempfile.TemporaryDirectory() as d:
            fallback = FallbackProvider()
            provider = LocalLLMProvider(manager=LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([])), fallback_provider=fallback)
            old = os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK")
            os.environ["RESEARCH_ALLOW_PAID_FALLBACK"] = "1"
            try:
                result = provider.generate(LLMRequest("x", "stage"))
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_ALLOW_PAID_FALLBACK", None)
                else:
                    os.environ["RESEARCH_ALLOW_PAID_FALLBACK"] = old
            self.assertEqual(result["model"], "remote")
            self.assertEqual(fallback.calls, 1)

    def test_aggressive_mode_never_triggers_paid_api_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            fallback = FallbackProvider()
            provider = LocalLLMProvider(manager=LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([])), fallback_provider=fallback)
            os.environ["RESEARCH_LOCAL_OPTIMIZATION"] = "aggressive"
            os.environ["RESEARCH_ALLOW_PAID_FALLBACK"] = "0"
            with self.assertRaises(RuntimeError):
                provider.generate(LLMRequest("x", "stage"))
            self.assertEqual(fallback.calls, 0)

    def test_model_cache_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.gguf"
            path.write_bytes(b"x" * 1024)
            registry = ModelRegistry(Path(d) / "cache", size_limit_bytes=1)
            registry.register({"id": "m", "path": str(path), "format": "gguf", "managed_cache_file": str(path), "pinned": False})
            removed = registry.enforce_limit()
            self.assertEqual(removed, ["m"])

    def test_local_capability_profile_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "tiny-q4.gguf"
            model_path.write_bytes(b"x" * 1024)
            registry = ModelRegistry(Path(d) / "cache")
            registry.register({"id": "tiny-q4", "path": str(model_path), "format": "gguf", "quantization": "Q4", "bits_per_weight": 4, "estimated_size_bytes": 1024})
            manager = LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([FakeRuntime()]), model_registry=registry, hardware_probe=FakeHardwareProbe())
            profile = manager.build_profile()
            self.assertTrue((Path(d) / "local_inference" / "profile.json").exists())
            self.assertTrue(profile["configurations"])
            self.assertIn("capability_prior", profile["configurations"][0])
            self.assertEqual(profile["configurations"][0]["measured_capability"], {})
            self.assertEqual(profile["configurations"][0]["eligible_task_classes"], [])

    def test_profile_discovery_preserves_existing_measured_routing(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "tiny-q4.gguf"
            model_path.write_bytes(b"x" * 1024)
            registry = ModelRegistry(Path(d) / "cache")
            registry.register({"id": "tiny-q4", "path": str(model_path), "format": "gguf",
                               "quantization": "Q4", "bits_per_weight": 4,
                               "estimated_size_bytes": 1024})
            manager = LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([FakeRuntime()]),
                                            model_registry=registry, hardware_probe=FakeHardwareProbe())
            original = manager.build_profile()
            original["configurations"][0]["measured_capability"] = {"structured_output": 1.0}
            original["configurations"][0]["eligible_task_classes"] = ["metadata_extraction"]
            original["routing_profile"] = {"task_assignments": {"metadata_extraction": original["configurations"][0]["id"]}}
            profile_path = Path(d) / "local_inference" / "profile.json"
            profile_path.write_text(json.dumps(original), encoding="utf-8")
            refreshed = manager.build_profile()
            self.assertEqual(refreshed["configurations"][0]["measured_capability"], {"structured_output": 1.0})
            self.assertEqual(refreshed["configurations"][0]["eligible_task_classes"], ["metadata_extraction"])
            self.assertEqual(refreshed["routing_profile"], original["routing_profile"])

    def test_load_profile_recovers_measured_benchmark_after_old_smoke_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "local_inference"
            benchmarks = root / "benchmarks"
            benchmarks.mkdir(parents=True)
            profile = {"configurations": [{"id": "cfg", "measured_capability": {},
                                             "eligible_task_classes": []}]}
            (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            measured = {"id": "cfg", "measured_capability": {"structured_output": 1.0},
                        "eligible_task_classes": ["metadata_extraction"], "tokens_per_second": 2.0}
            (benchmarks / "cfg:t2.json").write_text(json.dumps(measured), encoding="utf-8")
            routing = {"task_assignments": {"metadata_extraction": "cfg"}}
            (root / "routing_profile.json").write_text(json.dumps(routing), encoding="utf-8")
            loaded = LocalInferenceManager(root=d).load_or_build_profile()
            self.assertEqual(loaded["configurations"][0]["measured_capability"], {"structured_output": 1.0})
            self.assertEqual(loaded["configurations"][0]["eligible_task_classes"], ["metadata_extraction"])
            self.assertEqual(loaded["routing_profile"], routing)

    def test_context_reduction(self):
        reducer = ContextReducer(max_context=5)
        reduced = reducer.reduce({"task": "one two three four five six", "irrelevant": "drop me"})
        self.assertLessEqual(len(reduced.split()), 5)

    def test_no_research_evidence_derives_solely_from_llm_assertions(self):
        from src.ledger import EvidenceLedger

        with tempfile.TemporaryDirectory() as d:
            ledger = EvidenceLedger(d)
            with self.assertRaises(ValueError):
                ledger.add_claim({"claim": "model says result", "status": "VERIFIED_MEASUREMENT", "producer": "local_llm", "artifacts": []})

    def test_structured_q3_valid_success(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, runtime = local_structured_provider(d, ['{"selected_question":"q","candidate_evaluations":[],"rationale":"r"}'])
            result = provider.generate_structured(
                LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"]},
            )
            self.assertEqual(result["structured"]["selected_question"], "q")
            self.assertEqual(len(result["attempts"]), 1)
            self.assertEqual(runtime.calls[0]["config"]["id"], "llama.cpp:local-default:Q3_K_M:q4:1024")

    def test_structured_q3_invalid_repair_success(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, runtime = local_structured_provider(d, ['{"wrong":1}', '{"selected_question":"q","candidate_evaluations":[],"rationale":"r"}'])
            result = provider.generate_structured(
                LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"]},
            )
            self.assertEqual(len(result["attempts"]), 2)
            self.assertTrue(result["attempts"][1]["repair_attempt"])
            self.assertIn("missing required key: selected_question", result["attempts"][0]["schema_errors"])
            self.assertEqual(runtime.calls[1]["config"]["id"], "llama.cpp:local-default:Q3_K_M:q4:1024")

    def test_structured_q3_repair_invalid_q4_success(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, runtime = local_structured_provider(d, ['{"wrong":1}', '{"also_wrong":2}', '{"selected_question":"q","candidate_evaluations":[],"rationale":"r"}'])
            result = provider.generate_structured(
                LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"]},
            )
            self.assertEqual(len(result["attempts"]), 3)
            self.assertEqual(runtime.calls[2]["config"]["id"], "llama.cpp:local-quality:Q4_K_M:q4:1024")
            self.assertEqual(result["attempts"][2]["status"], "SUCCESS")

    def test_gateway_records_recovered_structured_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, _ = local_structured_provider(d, ['{"wrong":1}', '{"also_wrong":2}', '{"selected_question":"q","candidate_evaluations":[],"rationale":"r"}'])
            state = {"budget": {"llm_usd": 0.0, "strong_calls": 0, "calls": []}}
            gateway = ModelGateway(provider=provider)
            response = gateway.generate_structured(
                state,
                LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                required_keys=["selected_question", "candidate_evaluations", "rationale"],
            )
            self.assertEqual(response["structured"]["selected_question"], "q")
            self.assertEqual(len(state["budget"]["calls"]), 3)
            self.assertEqual(state["budget"]["calls"][0]["failure_type"], "SCHEMA_VALIDATION_FAILURE")
            self.assertTrue(state["budget"]["calls"][1]["repair_attempt"])
            self.assertEqual(state["budget"]["calls"][2]["status"], "SUCCESS")
            self.assertEqual(state["budget"]["llm_usd"], 0.0)
            self.assertEqual(state["budget"]["calls"][0]["raw_response"], '{"wrong":1}')

    def test_structured_q3_q4_invalid_human_required_signal(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, _ = local_structured_provider(d, ['{"wrong":1}', '{"also_wrong":2}', '{"still_wrong":3}'])
            with self.assertRaises(StructuredGenerationExhausted) as ctx:
                provider.generate_structured(
                    LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                    schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"]},
                )
            self.assertEqual(len(ctx.exception.attempts), 3)
            self.assertTrue(all(attempt["actual_cost"] == 0.0 for attempt in ctx.exception.attempts))
            self.assertEqual(ctx.exception.attempts[0]["raw_response"], '{"wrong":1}')

    def test_semantic_failure_repair_success(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, _ = local_structured_provider(d, [
                '{"selected_question":"What question should be selected?","candidate_evaluations":[],"rationale":"too short"}',
                '{"selected_question":"What question should be selected?","candidate_evaluations":[{"question":"What question should be selected?","feasibility":0.8,"novelty_potential":0.5,"falsifiability":0.7,"evidence_accessibility":0.9,"rationale":"This is specific and testable."}],"rationale":"This selection has accessible evidence and can be validated."}',
            ])
            def validator(data):
                return [] if data.get("candidate_evaluations") else ["candidate_evaluations must not be empty"]
            result = provider.generate_structured(
                LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"], "properties": {"selected_question": {"type": "string"}, "candidate_evaluations": {"type": "array"}, "rationale": {"type": "string"}}},
                semantic_validator=validator,
            )
            self.assertEqual(result["attempts"][0]["failure_type"], "SEMANTIC_VALIDATION_FAILURE")
            self.assertTrue(result["attempts"][1]["repair_attempt"])
            self.assertEqual(result["attempts"][1]["status"], "SUCCESS")

    def test_semantic_failure_q4_success(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, runtime = local_structured_provider(d, [
                '{"selected_question":"What question should be selected?","candidate_evaluations":[],"rationale":"too short"}',
                '{"selected_question":"What question should be selected?","candidate_evaluations":[],"rationale":"too short"}',
                '{"selected_question":"What question should be selected?","candidate_evaluations":[{"question":"What question should be selected?","feasibility":0.8,"novelty_potential":0.5,"falsifiability":0.7,"evidence_accessibility":0.9,"rationale":"This is specific and testable."}],"rationale":"This selection has accessible evidence and can be validated."}',
            ])
            def validator(data):
                return [] if data.get("candidate_evaluations") else ["candidate_evaluations must not be empty"]
            result = provider.generate_structured(
                LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"], "properties": {"selected_question": {"type": "string"}, "candidate_evaluations": {"type": "array"}, "rationale": {"type": "string"}}},
                semantic_validator=validator,
            )
            self.assertEqual(runtime.calls[2]["config"]["id"], "llama.cpp:local-quality:Q4_K_M:q4:1024")
            self.assertEqual(result["attempts"][2]["status"], "SUCCESS")

    def test_semantic_failure_exhaustion_preserves_errors(self):
        with tempfile.TemporaryDirectory() as d:
            write_measured_profile(d)
            provider, _ = local_structured_provider(d, [
                '{"selected_question":"What question should be selected?","candidate_evaluations":[],"rationale":"too short"}',
                '{"selected_question":"What question should be selected?","candidate_evaluations":[],"rationale":"too short"}',
                '{"selected_question":"What question should be selected?","candidate_evaluations":[],"rationale":"too short"}',
            ])
            with self.assertRaises(StructuredGenerationExhausted) as ctx:
                provider.generate_structured(
                    LLMRequest("task", "question_refinement", task_class="candidate_question_generation"),
                    schema={"type": "object", "required": ["selected_question", "candidate_evaluations", "rationale"], "properties": {"selected_question": {"type": "string"}, "candidate_evaluations": {"type": "array"}, "rationale": {"type": "string"}}},
                    semantic_validator=lambda data: ["candidate_evaluations must not be empty"] if not data.get("candidate_evaluations") else [],
                )
            self.assertEqual(ctx.exception.attempts[-1]["failure_type"], "SEMANTIC_VALIDATION_FAILURE")
            self.assertIn("candidate_evaluations must not be empty", ctx.exception.attempts[-1]["semantic_errors"])

    def test_unified_llama_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="unified", loopback_probe=loopback_available)
            discovered = runtime.discover()
            self.assertTrue(discovered["available"])
            self.assertEqual(discovered["executable"], binary)
            self.assertEqual(discovered["invocation_mode"], "UNIFIED_CLI_ROUTER")
            self.assertTrue(discovered["requires_local_socket"])

    def test_legacy_llama_cli_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama-cli")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="legacy")
            discovered = runtime.discover()
            self.assertTrue(discovered["available"])
            self.assertEqual(discovered["invocation_mode"], "DIRECT_CLI")

    def test_explicit_llama_cpp_bin(self):
        old = os.environ.get("LLAMA_CPP_BIN")
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama")
            os.environ["LLAMA_CPP_BIN"] = binary
            try:
                runtime = LlamaCppRuntime()
            finally:
                if old is None:
                    os.environ.pop("LLAMA_CPP_BIN", None)
                else:
                    os.environ["LLAMA_CPP_BIN"] = old
            self.assertEqual(runtime.executable, binary)
            self.assertEqual(runtime.invocation_mode, "UNIFIED_CLI_ROUTER")

    def test_unified_cli_command_construction(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="unified", loopback_probe=loopback_available)
            self.assertEqual(runtime.cli_command(["-m", "model.gguf"]), [binary, "cli", "-m", "model.gguf"])

    def test_direct_cli_descriptor_and_command_omit_subcommand(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama-cli")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="DIRECT_CLI",
                                      loopback_probe=loopback_unavailable)
            discovered = runtime.discover()
            self.assertEqual(discovered["invocation_mode"], "DIRECT_CLI")
            self.assertFalse(discovered["requires_local_socket"])
            self.assertEqual(discovered["backend_compatibility"], "COMPATIBLE")
            self.assertEqual(runtime.cli_command(["-m", "model.gguf"]), [binary, "-m", "model.gguf"])

    def test_router_backend_rejected_without_loopback_and_direct_still_supported(self):
        with tempfile.TemporaryDirectory() as d:
            unified = self._fake_executable(d, "llama")
            direct = self._fake_executable(d, "llama-cli")
            router = LlamaCppRuntime(executable=unified, invocation_mode="UNIFIED_CLI_ROUTER",
                                     loopback_probe=loopback_unavailable)
            direct_runtime = LlamaCppRuntime(executable=direct, invocation_mode="DIRECT_CLI",
                                             loopback_probe=loopback_unavailable)
            model = {"format": "gguf"}
            self.assertFalse(router.supports(model))
            self.assertEqual(router.discover()["backend_compatibility"], "BACKEND_INCOMPATIBLE_WITH_ENVIRONMENT")
            self.assertTrue(direct_runtime.supports(model))
            with unittest.mock.patch("src.local_inference.subprocess.run") as run:
                with self.assertRaises(LocalInferenceBackendIncompatible):
                    router.run_cli("prompt", {"id": "cfg", "model": {"path": "model.gguf"}})
                run.assert_not_called()

    def test_explicit_direct_cli_environment_precedes_unified_binary(self):
        with tempfile.TemporaryDirectory() as d:
            direct = self._fake_executable(d, "llama-cli")
            unified = self._fake_executable(d, "llama")
            with unittest.mock.patch.dict(os.environ, {"LLAMA_CPP_CLI_BIN": direct, "LLAMA_CPP_BIN": unified}):
                runtime = LlamaCppRuntime(loopback_probe=loopback_unavailable)
            self.assertEqual(runtime.executable, direct)
            self.assertEqual(runtime.invocation_mode, "DIRECT_CLI")
            self.assertEqual(runtime.discovery_source, "LLAMA_CPP_CLI_BIN")

    def test_loopback_probe_reports_bind_failure_deterministically(self):
        fake_socket = unittest.mock.Mock()
        fake_socket.bind.side_effect = OSError(13, "permission denied")
        with unittest.mock.patch("src.local_inference.socket.socket", return_value=fake_socket):
            result = probe_loopback_bind()
        self.assertEqual(result["status"], "LOOPBACK_BIND_UNAVAILABLE")
        self.assertEqual(result["errno"], 13)
        fake_socket.close.assert_called_once()

    def test_structured_flags_are_preserved_for_direct_cli(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama-cli")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="DIRECT_CLI",
                                      loopback_probe=loopback_unavailable)
            with unittest.mock.patch("src.local_inference.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = '{"ok":true}'
                run.return_value.stderr = ""
                diagnostic = runtime.run_cli("Output JSON", {"id": "cfg", "model": {"path": "model.gguf"},
                    "context": 128, "threads": 2, "json_schema": {"type": "object"},
                    "kv_cache": {"id": "q4"}, "reasoning": "off"})
            command = diagnostic["command"]
            self.assertNotEqual(command[1], "cli")
            for flag in ("--reasoning", "--no-jinja", "--json-schema", "--cache-type-k", "--cache-type-v"):
                self.assertIn(flag, command)
            self.assertEqual(diagnostic["resource_diagnostics"]["invocation_mode"], "DIRECT_CLI")

    def test_programmatic_cli_uses_simple_io_and_json_schema(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama")
            model = Path(d) / "model-Q2.gguf"
            model.write_bytes(b"x")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="unified", loopback_probe=loopback_available)
            with unittest.mock.patch("src.local_inference.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = '{"ok":true}'
                run.return_value.stderr = ""
                diagnostic = runtime.run_cli("Output JSON", {
                    "id": "cfg",
                    "model": {"path": str(model)},
                    "context": 128,
                    "threads": 2,
                    "json_schema": {"type": "object", "required": ["ok"]},
                })
            cmd = diagnostic["command"]
            self.assertIn("--simple-io", cmd)
            self.assertIn("-st", cmd)
            self.assertIn("--no-jinja", cmd)
            self.assertIn("--json-schema", cmd)
            self.assertEqual(diagnostic["exit_code"], 0)
            self.assertEqual(diagnostic["parsed_assistant_output"], '{"ok":true}')

    def test_llama_extractor_ignores_prompt_json_and_returns_assistant_json(self):
        prompt = 'OBJECTIVE\nWrite query.\nCONTEXT\n{"topic":"example"}\nEXPECTED\nReturn query.'
        stdout = (
            "llama.cpp build banner\n"
            f"> {render_programmatic_llama_prompt(prompt)[:20]} ...\n"
            '{"query":"example scholarly query"}\n'
            "[ Prompt: 10.0 t/s | Generation: 5.0 t/s ]\nExiting...\n"
        )
        extraction = extract_llama_assistant_response(stdout, prompt, structured=True)
        self.assertEqual(extraction["status"], "SUCCESS")
        self.assertEqual(extraction["parsed_response"], '{"query":"example scholarly query"}')
        self.assertEqual(extraction["assistant_text"], '{"query":"example scholarly query"}')
        self.assertNotIn("llama.cpp", extraction["assistant_text"])
        self.assertNotIn("> ", extraction["assistant_text"])
        self.assertNotIn("[ Prompt:", extraction["assistant_text"])
        self.assertNotIn("Exiting", extraction["assistant_text"])

    def test_llama_extractor_ignores_nested_context_json_and_footer(self):
        prompt = (
            "CONTEXT\n"
            '{"candidate":{"question":"What is in the prompt?"},"records":[{"title":"Prompt title"}]}\n'
            "EXPECTED JSON."
        )
        assistant = '{"selected_question":"What generated question should be used?","candidate_evaluations":[],"rationale":"Generated rationale."}'
        stdout = f"startup\n> {render_programmatic_llama_prompt(prompt)[:30]} ...\n{assistant}\n[ Prompt: 1 t/s | Generation: 1 t/s ]\nExiting...\n"
        extraction = extract_llama_assistant_response(stdout, prompt, structured=True)
        self.assertEqual(extraction["parsed_response"], assistant)
        self.assertNotEqual(parse_json_object(extraction["parsed_response"]).get("candidate"), {"question": "What is in the prompt?"})

    def test_llama_extractor_fails_without_generation_boundary(self):
        prompt = 'CONTEXT {"topic":"example"}'
        stdout = 'banner\nCONTEXT {"topic":"example"}\n{"query":"assistant query"}\n'
        extraction = extract_llama_assistant_response(stdout, prompt="different exact prompt", structured=True)
        self.assertEqual(extraction["status"], "ASSISTANT_OUTPUT_EXTRACTION_FAILURE")
        self.assertEqual(extraction["parsed_response"], "")

    def test_llama_extractor_ambiguous_prompt_only_json_fails(self):
        prompt = 'CONTEXT {"topic":"example"}'
        stdout = 'banner\nCONTEXT {"topic":"example"}\nllama_perf_context_print: eval time = 1 ms\n'
        extraction = extract_llama_assistant_response(stdout, prompt=prompt, structured=True)
        self.assertEqual(extraction["status"], "ASSISTANT_OUTPUT_EXTRACTION_FAILURE")
        self.assertEqual(extraction["parsed_response"], "")

    def test_llama_extractor_malformed_assistant_output_is_explicit(self):
        prompt = 'CONTEXT {"topic":"example"}'
        stdout = f"> {prompt}\n{{bad json\n"
        extraction = extract_llama_assistant_response(stdout, prompt, structured=True)
        self.assertEqual(extraction["status"], "STRUCTURED_OUTPUT_INCOMPLETE")
        self.assertEqual(extraction["parsed_response"], "")

    def test_llama_extractor_incomplete_outer_json_does_not_return_nested_object(self):
        prompt = 'CONTEXT {"candidate":{"question":"prompt object"}}'
        stdout = (
            f"> {prompt[:18]} ...\n"
            '{"candidate_evaluations":[{"question":"Nested complete?","score":0.5},{"question":\n'
            "[ Prompt: 1 t/s | Generation: 1 t/s ]\nExiting...\n"
        )
        extraction = extract_llama_assistant_response(stdout, prompt, structured=True)
        self.assertEqual(extraction["status"], "STRUCTURED_OUTPUT_INCOMPLETE")
        self.assertEqual(extraction["parsed_response"], "")

    def test_llama_extractor_incomplete_generation_does_not_fall_back_to_prompt_json(self):
        prompt = '{"candidate_questions":[{"question":"Prompt candidate?"}]}'
        stdout = f"> {prompt[:25]} ...\n{{\"candidate_evaluations\": [{{\"evidence_accessibility\":\n"
        extraction = extract_llama_assistant_response(stdout, prompt, structured=True)
        self.assertEqual(extraction["status"], "STRUCTURED_OUTPUT_INCOMPLETE")
        self.assertEqual(extraction["parsed_response"], "")

    def test_llama_extractor_two_top_level_objects_fails(self):
        prompt = "Return JSON"
        stdout = f"> {prompt}\n{{\"a\":1}}\n{{\"b\":2}}\n"
        extraction = extract_llama_assistant_response(stdout, prompt, structured=True)
        self.assertEqual(extraction["status"], "STRUCTURED_OUTPUT_INVALID_JSON")
        self.assertEqual(extraction["parsed_response"], "")

    def test_structured_decoding_failure_not_parsed_from_stderr_grammar(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama")
            model = Path(d) / "model-Q3.gguf"
            model.write_bytes(b"x")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="unified", loopback_probe=loopback_available)
            stderr = 'error initializing grammar sampler for grammar:\nroot ::= "{" space ok-kv space "}"\n'
            with unittest.mock.patch("src.local_inference.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "Error: Failed to initialize samplers: std::exception\n"
                run.return_value.stderr = stderr
                diagnostic = runtime.run_cli("Output JSON", {
                    "id": "cfg",
                    "model": {"path": str(model)},
                    "context": 128,
                    "threads": 2,
                    "json_schema": {"type": "object", "required": ["ok"]},
                })
            self.assertEqual(diagnostic["status"], "STRUCTURED_DECODING_CONFIGURATION_FAILURE")
            self.assertEqual(diagnostic["parsed_assistant_output"], "")
            self.assertTrue(is_structured_decoding_error(diagnostic["stdout"], diagnostic["stderr"]))

    def test_semantic_escalation_dedupes_kv_variants(self):
        profile = {"configurations": [
            {"id": "q3-q4", "model": {"id": "q3", "profile": "DEFAULT", "quantization": "Q3_K_M"}, "kv_quantization": {"id": "q4"}, "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"], "ram_estimate": 1},
            {"id": "q3-q8", "model": {"id": "q3", "profile": "DEFAULT", "quantization": "Q3_K_M"}, "kv_quantization": {"id": "q8"}, "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"], "ram_estimate": 2},
            {"id": "q4-q4", "model": {"id": "q4", "profile": "QUALITY", "quantization": "Q4_K_M"}, "kv_quantization": {"id": "q4"}, "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"], "ram_estimate": 3},
            {"id": "q4-fp16", "model": {"id": "q4", "profile": "QUALITY", "quantization": "Q4_K_M"}, "kv_quantization": {"id": "fp16"}, "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"], "ram_estimate": 4},
        ]}
        candidates = LocalModelRouter(profile).candidates("CHEAP", "candidate_question_generation")
        self.assertEqual([c["id"] for c in candidates], ["q3-q4", "q4-q4"])

    def test_json_parse_mode_distinguishes_strict_and_found(self):
        parsed = parse_json_object('thanks\n{"ok":true}\nextra')
        self.assertEqual(parsed, {"ok": True})
        self.assertEqual(json_parse_mode('{"ok":true}', {"ok": True}), "STRICT_JSON_ONLY")
        self.assertEqual(json_parse_mode('thanks\n{"ok":true}', {"ok": True}), "JSON_FOUND")

    def test_legacy_cli_command_construction(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama-cli")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="legacy")
            self.assertEqual(runtime.cli_command(["-m", "model.gguf"]), [binary, "-m", "model.gguf"])

    def test_unified_serve_command_construction(self):
        with tempfile.TemporaryDirectory() as d:
            binary = self._fake_executable(d, "llama")
            runtime = LlamaCppRuntime(executable=binary, invocation_mode="unified", loopback_probe=loopback_available)
            self.assertEqual(runtime.serve_command(["-m", "model.gguf"]), [binary, "serve", "-m", "model.gguf"])

    def test_no_llama_binary_available(self):
        runtime = LlamaCppRuntime(executable=None, invocation_mode=None)
        runtime.executable = None
        runtime.invocation_mode = None
        discovered = runtime.discover()
        self.assertFalse(discovered["available"])
        self.assertIsNone(discovered["executable"])

    def test_missing_research_local_models_reported(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_MODELS")
            old_specs = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            os.environ["RESEARCH_LOCAL_MODELS"] = "/path/to/Qwen3-1.7B-Q4_K_M.gguf:/path/to/Qwen3-4B-Q4_K_M.gguf"
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(Path(d) / "none.json")
            try:
                registry = ModelRegistry(Path(d) / "cache")
                self.assertEqual(registry.discover_from_env(), [])
                missing = registry.missing_from_env()
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_MODELS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODELS"] = old
                if old_specs is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old_specs
            self.assertEqual(len(missing), 2)
            self.assertTrue(all(item["reason"] == "missing" for item in missing))

    def test_hf_cache_symlink_registers_configured_model(self):
        with tempfile.TemporaryDirectory() as d:
            hf_root = Path(d) / "hf" / "hub"
            repo_dir = hf_root / "models--example--repo"
            blob = repo_dir / "blobs" / "abc123"
            snapshot = repo_dir / "snapshots" / "rev1"
            blob.parent.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            blob.write_bytes(b"model-bytes")
            symlink = snapshot / "model-Q2_K.gguf"
            symlink.symlink_to(Path("../../blobs/abc123"))
            spec = Path(d) / "models.json"
            spec.write_text(json.dumps({"models": [{
                "id": "local-fast",
                "profile": "fast",
                "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q2_K", "filename": "model-Q2_K.gguf"},
                "quantization": "Q2_K",
                "task_tier": "FAST",
            }]}), encoding="utf-8")
            old_specs = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            old_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(spec)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_root)
            try:
                registry = ModelRegistry(Path(d) / "cache")
                models = registry.discover_from_env()
            finally:
                if old_specs is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old_specs
                if old_cache is None:
                    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
                else:
                    os.environ["HUGGINGFACE_HUB_CACHE"] = old_cache
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["path"], str(symlink))
            self.assertEqual(models[0]["resolved_path"], str(blob.resolve()))
            self.assertTrue(models[0]["symlink_backed"])
            self.assertEqual(models[0]["size_bytes"], len(b"model-bytes"))
            self.assertEqual(models[0]["quantization"], "Q2_K")

    def test_install_profile_reuses_cached_hf_model(self):
        with tempfile.TemporaryDirectory() as d:
            hf_root = Path(d) / "hf" / "hub"
            repo_dir = hf_root / "models--example--repo"
            blob = repo_dir / "blobs" / "abc123"
            snapshot = repo_dir / "snapshots" / "rev1"
            blob.parent.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            blob.write_bytes(b"model-bytes")
            (snapshot / "model-Q3_K_M.gguf").symlink_to(Path("../../blobs/abc123"))
            spec = Path(d) / "models.json"
            spec.write_text(json.dumps({"models": [{
                "id": "local-default",
                "profile": "default",
                "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q3_K_M", "filename": "model-Q3_K_M.gguf"},
                "quantization": "Q3_K_M",
                "task_tier": "DEFAULT",
            }]}), encoding="utf-8")
            old_specs = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            old_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(spec)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_root)
            try:
                registry = ModelRegistry(Path(d) / "cache")
                with unittest.mock.patch("src.local_inference.subprocess.run") as run:
                    result = registry.install_profile("default", runtime=FakeDownloadRuntime())
            finally:
                if old_specs is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old_specs
                if old_cache is None:
                    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
                else:
                    os.environ["HUGGINGFACE_HUB_CACHE"] = old_cache
            self.assertEqual(result["status"], "already_present")
            run.assert_not_called()

    def test_missing_environment_paths_are_separate_from_configured_models(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "model-Q4_K_M.gguf"
            model.write_bytes(b"x" * 1024)
            spec = Path(d) / "models.json"
            spec.write_text(json.dumps({"models": [{
                "id": "local-quality",
                "profile": "quality",
                "path": str(model),
                "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q4_K_M"},
                "quantization": "Q4_K_M",
                "task_tier": "QUALITY",
            }]}), encoding="utf-8")
            old_models = os.environ.get("RESEARCH_LOCAL_MODELS")
            old_specs = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            os.environ["RESEARCH_LOCAL_MODELS"] = "/exact/path/to/Qwen3-1.7B-Q4_K_M.gguf"
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(spec)
            try:
                registry = ModelRegistry(Path(d) / "cache")
                models = registry.discover_from_env()
                missing = registry.missing_by_source()
            finally:
                if old_models is None:
                    os.environ.pop("RESEARCH_LOCAL_MODELS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODELS"] = old_models
                if old_specs is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old_specs
            self.assertEqual(len(models), 1)
            self.assertEqual(len(missing["missing_environment_models"]), 1)
            self.assertEqual(missing["missing_configured_models"], [])

    def test_benchmark_preserves_execution_diagnostics(self):
        config = {"id": "cfg", "model": {"path": "/tmp/model.gguf"}, "context": 128}
        result = CapabilityBenchmark().measured_benchmark(DiagnosticRuntime(), config, threads=2)
        diagnostic = result["raw_outputs"]["structured_output"]
        self.assertEqual(diagnostic["status"], "SUCCESS")
        self.assertEqual(diagnostic["exit_code"], 0)
        self.assertEqual(diagnostic["parsed_assistant_output"], '{"ok":true}')
        self.assertIn("stdout", diagnostic)
        self.assertIn("stderr", diagnostic)
        self.assertEqual(result["capabilities"]["structured_output"], 1.0)
        self.assertEqual(result["prompt_tokens_per_second"], 500.0)

    def test_feasibility_benchmark_measures_eligibility(self):
        config = {"id": "cfg", "model": {"path": "/tmp/model.gguf"}, "context": 128}
        result = CapabilityBenchmark().measured_benchmark(FeasibilityBenchmarkRuntime(), config, threads=2)
        self.assertEqual(result["capabilities"]["feasibility_reasoning"], 1.0)
        self.assertIn("research_feasibility_analysis", result["eligible_task_classes"])

    def test_benchmark_empty_output_is_not_scored_zero(self):
        config = {"id": "cfg", "model": {"path": "/tmp/model.gguf"}, "context": 128}
        result = CapabilityBenchmark().measured_benchmark(EmptyRuntime(), config, threads=2)
        self.assertEqual(result["raw_outputs"]["structured_output"]["status"], "EMPTY_OUTPUT")
        self.assertEqual(result["raw_outputs"]["structured_output"]["failure_class"], "MODEL_OUTPUT_FAILURE")
        self.assertNotIn("structured_output", result["capabilities"])

    def test_smoke_reports_runtime_environment_failure(self):
        config = {"id": "cfg", "model": {"path": "/tmp/model.gguf"}, "context": 128}
        result = CapabilityBenchmark().diagnostic_smoke(EnvironmentFailureRuntime(), config, threads=2)
        self.assertEqual(result["failure_class"], "RUNTIME_ENVIRONMENT_FAILURE")
        self.assertNotEqual(result["status"], "SUCCESS")

    def test_smoke_uses_production_structured_configuration_and_const_schema(self):
        runtime = DiagnosticRuntime()
        config = {"id": "cfg", "model": {"path": "/tmp/model.gguf"}, "context": 128}
        result = CapabilityBenchmark().diagnostic_smoke(runtime, config, threads=2)
        used = runtime.calls[0]["config"]
        self.assertEqual(used["json_schema"], STRUCTURED_SMOKE_SCHEMA)
        self.assertEqual(used["reasoning"], "off")
        self.assertEqual(used["generation_mode"], "STRUCTURED_GENERATION")
        self.assertTrue(result["structured_decoding"])
        self.assertTrue(result["json_schema_enabled"])
        self.assertEqual(result["parsed_json"], {"ok": True})
        self.assertEqual(result["json_parse_mode"], "STRICT_JSON_ONLY")

    def test_smoke_rejects_thinking_prose_without_json(self):
        result = CapabilityBenchmark().diagnostic_smoke(
            ThinkingProseRuntime(), {"id": "cfg", "model": {"path": "/tmp/model.gguf"}, "context": 128})
        self.assertNotEqual(result["status"], "SUCCESS")
        self.assertIsNone(result["parsed_json"])
        self.assertNotEqual(result.get("failure_class"), None)

    def test_structured_config_requires_schema_but_freeform_does_not_gain_one(self):
        structured = structured_generation_config({"context": 128}, STRUCTURED_SMOKE_SCHEMA)
        self.assertIn("json_schema", structured)
        self.assertNotIn("json_schema", {"context": 128, "reasoning": "off", "generation_mode": "FREEFORM_GENERATION"})
        with self.assertRaises(ValueError):
            structured_generation_config({}, None)

    def test_manager_smoke_summary_passes_with_fake_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "tiny-Q2.gguf"
            model_path.write_bytes(b"x" * 1024)
            registry = ModelRegistry(Path(d) / "cache")
            registry.register({"id": "tiny-q2", "path": str(model_path), "format": "gguf", "quantization": "Q2", "bits_per_weight": 2, "estimated_size_bytes": 1024, "task_tier": "FAST"})
            manager = LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([DiagnosticRuntime()]), model_registry=registry, hardware_probe=FakeHardwareProbe())
            result = manager.smoke_test()
            self.assertTrue(result["pass"])
            self.assertEqual(result["summary"]["status"], "PASS")
            self.assertEqual(result["summary"]["parsed_response"], '{"ok":true}')

    def test_benchmark_refuses_when_smoke_fails(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "tiny-Q2.gguf"
            model_path.write_bytes(b"x" * 1024)
            registry = ModelRegistry(Path(d) / "cache")
            registry.register({"id": "tiny-q2", "path": str(model_path), "format": "gguf", "quantization": "Q2", "bits_per_weight": 2, "estimated_size_bytes": 1024, "task_tier": "FAST"})
            manager = LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([EnvironmentFailureRuntime()]), model_registry=registry, hardware_probe=FakeHardwareProbe())
            result = manager.benchmark_configurations(thread_counts=[2], max_configs=1)
            self.assertEqual(result["benchmark_status"], "REFUSED_SMOKE_FAILED")
            self.assertFalse(result["smoke"]["pass"])
            self.assertEqual(result["smoke"]["summary"]["failure_class"], "RUNTIME_ENVIRONMENT_FAILURE")

    def test_configured_model_spec_registers_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "model-Q3_K_M.gguf"
            model.write_bytes(b"x" * 1024)
            spec = Path(d) / "models.json"
            spec.write_text(json.dumps({"models": [{
                "id": "local-default",
                "profile": "default",
                "path": str(model),
                "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q3_K_M"},
                "quantization": "Q3_K_M",
                "task_tier": "DEFAULT",
            }]}), encoding="utf-8")
            old = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(spec)
            try:
                registry = ModelRegistry(Path(d) / "cache")
                models = registry.discover_from_env()
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old
            self.assertEqual(models[0]["task_tier"], "DEFAULT")
            self.assertEqual(models[0]["quantization"], "Q3_K_M")

    def test_install_profile_requires_configured_spec(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(Path(d) / "none.json")
            registry = ModelRegistry(Path(d) / "cache")
            try:
                with self.assertRaises(ValueError):
                    registry.install_profile("fast", runtime=FakeDownloadRuntime())
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old

    def test_download_command_for_configured_profile(self):
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / "models.json"
            spec.write_text(json.dumps({"models": [{
                "id": "local-fast",
                "profile": "fast",
                "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q2_K"},
                "quantization": "Q2_K",
                "task_tier": "FAST",
            }]}), encoding="utf-8")
            old = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(spec)
            try:
                registry = ModelRegistry(Path(d) / "cache")
                with unittest.mock.patch("src.local_inference.subprocess.run") as run:
                    run.return_value.returncode = 0
                    run.return_value.stdout = ""
                    run.return_value.stderr = ""
                    result = registry.install_profile("fast", runtime=FakeDownloadRuntime())
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old
            self.assertEqual(result["model"]["install_command"], ["/tmp/llama", "download", "-hf", "example/repo:Q2_K"])

    def test_benchmark_routing_profile_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "tiny-Q3.gguf"
            model_path.write_bytes(b"x" * 1024)
            registry = ModelRegistry(Path(d) / "cache")
            registry.register({"id": "tiny-q3", "path": str(model_path), "format": "gguf", "quantization": "Q3", "bits_per_weight": 3, "estimated_size_bytes": 1024, "task_tier": "DEFAULT"})
            manager = LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([FakeRuntime()]), model_registry=registry, hardware_probe=FakeHardwareProbe())
            profile = manager.benchmark_configurations(thread_counts=[2], max_configs=1)
            self.assertTrue((Path(d) / "local_inference" / "routing_profile.json").exists())
            self.assertIn("routing_profile", profile)

    def test_profile_reports_rejected_and_uninstalled_models(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / "too-big-Q4.gguf"
            model_path.write_bytes(b"x" * 1024)
            spec = Path(d) / "models.json"
            spec.write_text(json.dumps({"models": [
                {"id": "big", "profile": "quality", "path": str(model_path), "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q4_K_M"}, "quantization": "Q4_K_M"},
                {"id": "missing-fast", "profile": "fast", "source": {"type": "huggingface", "repo": "example/repo", "quant": "Q2_K"}, "quantization": "Q2_K"}
            ]}), encoding="utf-8")
            old = os.environ.get("RESEARCH_LOCAL_MODEL_SPECS")
            os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = str(spec)
            class TinyHardware:
                def persist(self, root):
                    return {"available_ram_bytes": 1, "system_ram_bytes": 1, "gpus": [], "supported_compute_backends": ["cpu"]}
            try:
                manager = LocalInferenceManager(root=d, runtime_registry=RuntimeRegistry([FakeRuntime()]), model_registry=ModelRegistry(Path(d) / "cache"), hardware_probe=TinyHardware())
                profile = manager.build_profile()
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_MODEL_SPECS", None)
                else:
                    os.environ["RESEARCH_LOCAL_MODEL_SPECS"] = old
            self.assertTrue(profile["rejected_configurations"])
            self.assertTrue(profile["uninstalled_models"])


if __name__ == "__main__":
    unittest.main()
