import json
import os
import tempfile
import unittest
from pathlib import Path

from src.llm_gateway import LLMBudgetManager, LLMRequest
from src.local_inference import (LocalRuntimeInfrastructureFailure, StructuredDecodingConfigurationFailure,
                                 StructuredGenerationExhausted)
from src.research_runtime import (
    GuidedToolController,
    GenericResearchRuntime,
    QUESTION_REFINEMENT_SCHEMA,
    candidate_question_semantic_validation,
    computational_measurement_decision_contract,
    external_reasoning_options,
    literature_relevance_report,
    placeholder_like,
    repair_invalid_evidence_relevance,
    repair_recoverable_structured_generation_failures,
    regenerate_external_reasoning_bundle,
    reconcile_external_decision_continuation,
    response_schema_for_node,
    submit_research_decision,
    validate_search_query,
    validate_atomic_planning_score,
    validate_dimension_score,
    validate_feasibility_requirement,
    validate_novelty_assessment,
)
from src.research_state import acquire_node_lease, create_run_state
from src.research_cli import research_report
from src.storage import LocalArtifactStore
from src.verification import verify_research_run


class FakeGateway:
    def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
        state.setdefault("budget", {"calls": [], "llm_usd": 0.0, "strong_calls": 0}).setdefault("calls", []).append({
            "stage": request.stage,
            "task_class": request.task_class,
            "model_class": request.requested_model_class,
            "actual_model": "fake",
            "input_tokens": len(request.prompt.split()),
            "output_tokens": 10,
            "estimated_cost": 0.0,
        })
        if request.stage == "question_discovery":
            data = {
                "candidate_questions": [{"question": "What patterns appear in the retrieved literature metadata?", "why_interesting": "feasible", "falsifiability": "metadata can refute", "local_executability": "yes"}],
                "search_queries": ["runtime generic evidence"],
            }
        elif request.stage == "question_refinement":
            data = {
                "selected_question": "What patterns appear in the retrieved literature metadata?",
                "candidate_evaluations": [{
                    "question": "What patterns appear in the retrieved literature metadata?",
                    "feasibility": 0.9,
                    "novelty_potential": 0.4,
                    "falsifiability": 0.8,
                    "evidence_accessibility": 0.9,
                    "rationale": "The retrieved metadata can be analyzed and independently checked.",
                }],
                "rationale": "This question is locally executable with retrieved metadata and deterministic validation.",
            }
        else:
            data = {
                "research_question": "What patterns appear in the retrieved literature metadata?",
                "methodology": "systematic literature metadata analysis",
                "feasibility_verdict": "FEASIBLE",
                "evidence_requirements": ["retrieved metadata records", "deduplicated record inventory"],
                "resource_constraints": ["local Python execution", "no paid external services"],
                "validation_plan": ["recompute record counts from raw retrievals"],
                "hypotheses": ["retrieval yields reproducible metadata"],
                "falsification_criteria": ["deduped count cannot be reproduced"],
                "required_claims": ["metadata inventory count"],
                "completion_contracts": ["validated metrics and replication report"],
                "replication_tolerance": {"exact_match": True},
            }
        return {"structured": data, "text": json.dumps(data), "model": "fake"}


class FakeLiteratureProvider:
    provider_name = "fake"

    def __init__(self, count=4):
        self.count = count

    def search(self, query, limit=10):
        records = []
        for idx in range(self.count):
            records.append({
                "identifier": f"W{idx}",
                "title": f"{query} record {idx}",
                "authors": ["A"],
                "year": 2020 + idx,
                "venue": "Venue",
                "doi": f"10.test/{idx}",
                "stable_url": f"https://example.test/{idx}",
                "abstract": f"This record discusses {query} and reproducible metadata evidence.",
                "source_provider": "fake",
                "retrieval_timestamp": "now",
                "search_query": query,
                "verification_status": "VERIFIED_METADATA",
                "limitations": [],
            })
        return {"provider": "fake", "query": query, "retrieval_timestamp": "now", "records": records, "raw_response": {"ok": True}}


class TopicAwareLiteratureProvider:
    provider_name = "fake"

    def __init__(self):
        self.queries = []

    def search(self, query, limit=10):
        self.queries.append(query)
        if "broad runtime topic" in query.lower():
            records = [{
                "identifier": "R1",
                "title": "Broad runtime topic metadata analysis",
                "authors": [],
                "year": 2024,
                "venue": "Venue",
                "doi": None,
                "stable_url": "https://example.test/relevant",
                "abstract": "This study analyzes broad runtime topic metadata with local reproducible methods.",
                "source_provider": "fake",
                "retrieval_timestamp": "now",
                "search_query": query,
                "verification_status": "VERIFIED_METADATA",
                "limitations": [],
            }, {
                "identifier": "R2",
                "title": "Runtime topic reproducible evidence",
                "authors": [],
                "year": 2024,
                "venue": "Venue",
                "doi": None,
                "stable_url": "https://example.test/relevant2",
                "abstract": "Broad runtime topic evidence can be checked using metadata records and deterministic scripts.",
                "source_provider": "fake",
                "retrieval_timestamp": "now",
                "search_query": query,
                "verification_status": "VERIFIED_METADATA",
                "limitations": [],
            }]
        else:
            records = [{
                "identifier": "U1",
                "title": "Unrelated molecular database",
                "authors": [],
                "year": 2020,
                "venue": "Venue",
                "doi": None,
                "stable_url": "https://example.test/unrelated",
                "abstract": "This verified metadata record describes molecular assays and compound databases.",
                "source_provider": "fake",
                "retrieval_timestamp": "now",
                "search_query": query,
                "verification_status": "VERIFIED_METADATA",
                "limitations": [],
            }]
        return {"provider": "fake", "query": query, "retrieval_timestamp": "now", "records": records, "raw_response": {"ok": True}}


class UnrelatedLiteratureProvider(TopicAwareLiteratureProvider):
    def search(self, query, limit=10):
        self.queries.append(query)
        records = [{
            "identifier": f"U{len(self.queries)}",
            "title": "Unrelated molecular database",
            "authors": [],
            "year": 2020,
            "venue": "Venue",
            "doi": None,
            "stable_url": "https://example.test/unrelated",
            "abstract": "This verified metadata record describes molecular assays and compound databases.",
            "source_provider": "fake",
            "retrieval_timestamp": "now",
            "search_query": query,
            "verification_status": "VERIFIED_METADATA",
            "limitations": [],
        }]
        return {"provider": "fake", "query": query, "retrieval_timestamp": "now", "records": records, "raw_response": {"ok": True}}


def run_runtime_to_completion(root, count=4):
    store = LocalArtifactStore(Path(root) / "runs")
    state = create_run_state("run", "broad runtime topic")
    store.atomic_update_state("run", state)
    runtime = GenericResearchRuntime(store, work_root=Path(root) / "work", literature_provider=FakeLiteratureProvider(count), gateway=FakeGateway())
    while True:
        node = acquire_node_lease(state, "test")
        if not node:
            break
        runtime.execute(state, node)
        store.atomic_update_state("run", state)
        if state.get("status", "").startswith("BLOCKED") or state.get("status") == "FAILED":
            break
    return store, state


def seed_relevant_literature(state, query="broad runtime topic"):
    state["literature_cache"] = TopicAwareLiteratureProvider().search(query)["records"]
    state["literature_relevance"] = literature_relevance_report(state["literature_cache"], state["topic"], state.get("candidate_questions", []))


class ResearchRuntimeTests(unittest.TestCase):
    def test_broad_topic_to_candidate_questions(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertTrue(state["candidate_questions"])
            self.assertTrue(state["search_strategy"]["queries"])

    def test_high_guidance_query_step_invokes_literature_search_then_question_microstep(self):
        class GuidedGateway:
            def __init__(self):
                self.calls = 0
                self.prompts = []

            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                self.calls += 1
                self.prompts.append(request.prompt)
                if self.calls == 1:
                    data = {"query": "broad runtime topic metadata"}
                else:
                    data = {"question": "How does broad runtime topic metadata vary across observable conditions?"}
                if semantic_validator:
                    errors = semantic_validator(data)
                    if errors:
                        raise AssertionError(errors)
                state.setdefault("budget", {"calls": [], "llm_usd": 0.0, "strong_calls": 0}).setdefault("calls", []).append({
                    "stage": request.stage,
                    "task_class": request.task_class,
                    "actual_model": "fake-local",
                    "status": "SUCCESS",
                    "actual_cost": 0.0,
                    "input_tokens": len(request.prompt.split()),
                })
                return {"structured": data, "text": json.dumps(data), "model": "fake-local"}

        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                gateway = GuidedGateway()
                provider = TopicAwareLiteratureProvider()
                store = LocalArtifactStore(Path(d) / "runs")
                state = create_run_state("run", "broad runtime topic")
                runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=provider, gateway=gateway)
                node = state["dag"]["nodes"]["question_discovery"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else:
                    os.environ["RESEARCH_LOCAL_GUIDANCE"] = old
            self.assertEqual(node["status"], "COMPLETED")
            self.assertEqual(state["search_strategy"]["queries"], ["broad runtime topic metadata"])
            self.assertEqual(
                state["candidate_questions"][0]["question"],
                "How does broad runtime topic metadata vary across observable conditions?",
            )
            provenance = state["candidate_question_field_provenance"][0]
            self.assertEqual(provenance["question"]["origin"], "local_model")
            self.assertEqual(provenance["why_interesting"]["origin"], "deterministic_supervisor")
            self.assertTrue(state["guided_agent_steps"])
            self.assertTrue(state["guided_literature_results"])
            self.assertGreaterEqual(len(provider.queries), 1)
            self.assertIn("OBJECTIVE", gateway.prompts[0])
            self.assertIn("Write one scholarly search query", gateway.prompts[0])
            self.assertIn("Write one empirically testable research question", gateway.prompts[1])

    def test_invalid_model_query_uses_deterministic_fallback_before_human(self):
        class BadQueryGateway:
            def __init__(self):
                self.calls = 0

            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                self.calls += 1
                data = {"query": "literature_search"} if self.calls == 1 else {"question": "How does broad runtime topic metadata vary across observable conditions?"}
                return {"structured": data, "text": json.dumps(data), "model": "fake-local"}

        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                gateway = BadQueryGateway()
                provider = TopicAwareLiteratureProvider()
                store = LocalArtifactStore(Path(d) / "runs")
                state = create_run_state("run", "broad runtime topic")
                runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=provider, gateway=gateway)
                node = state["dag"]["nodes"]["question_discovery"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else:
                    os.environ["RESEARCH_LOCAL_GUIDANCE"] = old
            self.assertEqual(node["status"], "COMPLETED")
            self.assertEqual(state["search_strategy"]["query_origin"], "deterministic_fallback")
            self.assertGreaterEqual(len(provider.queries), 1)
            self.assertNotEqual(provider.queries[0], "literature_search")

    def test_candidate_generation_failure_does_not_install_meta_question(self):
        class BadQuestionGateway:
            def __init__(self):
                self.calls = 0

            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                self.calls += 1
                data = {"query": "broad runtime topic metadata"} if self.calls == 1 else {"question": "question"}
                return {"structured": data, "text": json.dumps(data), "model": "fake-local"}

        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                provider = TopicAwareLiteratureProvider()
                store = LocalArtifactStore(Path(d) / "runs")
                state = create_run_state("run", "broad runtime topic")
                runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=provider, gateway=BadQuestionGateway())
                node = state["dag"]["nodes"]["question_discovery"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else:
                    os.environ["RESEARCH_LOCAL_GUIDANCE"] = old
            self.assertEqual(node["status"], "WAITING_FOR_HUMAN")
            self.assertEqual(state.get("candidate_questions", []), [])
            self.assertNotIn("deterministic_fallback", json.dumps(state.get("candidate_question_field_provenance", [])))

    def test_atomic_question_refinement_supervisor_assembles_one_candidate(self):
        class AtomicRefinementGateway:
            def __init__(self):
                self.prompts = []
                self.schemas = []

            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                self.prompts.append(request.prompt)
                self.schemas.append(schema)
                if "search phrase" in request.prompt:
                    data = {"query": "broad runtime topic observable conditions"}
                elif "already well covered" in request.prompt:
                    data = {"assessment": "plausible_gap", "reason": "Retrieved prior work leaves this exact comparison and setting insufficiently covered.", "confidence": 0.6}
                elif "contradicted or constrained" in request.prompt:
                    data = {"score": 0.7, "reason": "Observable measurements showing no association would count against the proposed relationship."}
                else:
                    data = {"score": 0.7, "reason": "Available local data and free computational tools make the next investigation practical."}
                if semantic_validator:
                    errors = semantic_validator(data)
                    if errors:
                        raise AssertionError(errors)
                state.setdefault("budget", {"calls": [], "llm_usd": 0.0, "strong_calls": 0}).setdefault("calls", []).append({
                    "stage": request.stage,
                    "task_class": request.task_class,
                    "actual_model": "fake-local",
                    "status": "SUCCESS",
                    "actual_cost": 0.0,
                })
                return {"structured": data, "text": json.dumps(data), "model": "fake-local"}

        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                gateway = AtomicRefinementGateway()
                store = LocalArtifactStore(Path(d) / "runs")
                state = create_run_state("run", "broad runtime topic")
                records = TopicAwareLiteratureProvider().search("broad runtime topic")["records"]
                state["literature_cache"] = records
                state["literature_relevance"] = literature_relevance_report(records, state["topic"], [])
                state["candidate_questions"] = [{
                    "question": "How does broad runtime topic metadata vary across observable conditions?",
                    "why_interesting": "The question can guide planning.",
                    "falsifiability": "Records can contradict assumptions.",
                    "local_executability": "Metadata checks can run locally.",
                }]
                state["candidate_question_field_provenance"] = [{"question": {"origin": "local_model"}}]
                runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=gateway)
                node = state["dag"]["nodes"]["question_refinement"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else:
                    os.environ["RESEARCH_LOCAL_GUIDANCE"] = old
            self.assertEqual(node["status"], "COMPLETED")
            self.assertEqual(state["selected_question"], "How does broad runtime topic metadata vary across observable conditions?")
            self.assertEqual(len(state["candidate_evaluations"]), 1)
            self.assertEqual(state["candidate_evaluations"][0]["question"], state["selected_question"])
            self.assertEqual(state["candidate_evaluations"][0]["automation_closure"], "UNKNOWN")
            self.assertIn("required_evidence_modalities", state["candidate_evaluations"][0])
            self.assertEqual(state["question_refinement_field_provenance"]["selected_question"]["origin"], "local_model")
            self.assertEqual(state["question_refinement_field_provenance"]["rationale"]["origin"], "deterministic_assembly")
            self.assertEqual(len(gateway.prompts), 4)
            self.assertNotIn("PRIOR-WORK", gateway.prompts[0])
            self.assertIn("Ignore whether we currently possess enough literature", gateway.prompts[0])
            self.assertEqual(set(gateway.schemas[0]["required"]), {"score", "reason"})
            self.assertIn("novelty_challenge", state)
            self.assertEqual(state["literature_cache"], records)
            self.assertNotEqual(id(state["novelty_challenge"]["results"]), id(state["literature_cache"]))
            self.assertTrue(state["novelty_challenge"]["raw_query"])
            self.assertTrue(state["novelty_challenge"]["normalized_query"])

    def test_atomic_refinement_rejects_empirical_answer_like_reason(self):
        from src.research_runtime import validate_atomic_planning_score
        errors = validate_atomic_planning_score({
            "score": 0.8,
            "reason": "The evidence proves the intervention has a positive result.",
        })
        self.assertTrue(any("answer the research question" in error for error in errors))

    def test_guided_prompt_omits_known_placeholder_values(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = GenericResearchRuntime(LocalArtifactStore(Path(d) / "runs"), work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            prompt = runtime._guided_prompt("Find literature.", {"topic": "broad runtime topic"}, "Choose action.")
            forbidden = [
                "specific answerable research question text",
                "actual scholarly search text tied to the topic",
                "non-empty string",
                "string non-empty",
                "broad discovery query",
                "targeted query",
            ]
            for phrase in forbidden:
                self.assertNotIn(phrase, prompt)

    def test_guided_tools_web_fetch_python_and_bash_provenance(self):
        class FakeWeb:
            def search(self, query, limit=5):
                return {"results": [{"url": "https://example.test", "title": "Example"}], "query": query}
            def fetch(self, url):
                return {"requested_url": url, "final_url": url, "retrieval_timestamp": "now", "status": "SUCCESS", "content_type": "text/plain", "sha256": "abc", "text": "source text"}

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            store.atomic_update_state("run", state)
            controller = GuidedToolController(store, "run", Path(d) / "work", FakeLiteratureProvider(), web_provider=FakeWeb())
            web = controller.execute({"tool": "web_search", "arguments": {"query": "broad runtime topic"}})
            fetched = controller.execute({"tool": "fetch_web_source", "arguments": {"url": "https://example.test"}})
            py = controller.execute({"tool": "run_python", "arguments": {"code": "print(2+2)"}})
            bash = controller.execute({"tool": "run_bash", "arguments": {"command": "printf ok"}})
            unsafe = controller.execute({"tool": "run_bash", "arguments": {"command": "sudo apt-get update"}})
            self.assertIn("web_search returned", web["observation"])
            self.assertIn("fetched", fetched["observation"])
            self.assertIn("stdout=4", py["observation"])
            self.assertIn("stdout=ok", bash["observation"])
            self.assertEqual(unsafe["result"]["status"], "HUMAN_REQUIRED")
            manifest_paths = {a["path"] for a in store.load_manifest("run")["artifacts"]}
            self.assertTrue(any(path.startswith("guided_tools/") for path in manifest_paths))

    def test_generic_tool_protocol_normalizes_nested_tool_when_unambiguous(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            store.atomic_update_state("run", state)
            provider = TopicAwareLiteratureProvider()
            controller = GuidedToolController(store, "run", Path(d) / "work", provider)
            result = controller.execute({"action": "tool", "arguments": {"tool": "literature_search", "query": "broad runtime topic"}})
            self.assertIn("literature_search returned", result["observation"])
            self.assertEqual(provider.queries, ["broad runtime topic"])
            artifact = json.loads((store.raw_root("run") / result["artifact"]).read_text(encoding="utf-8"))
            self.assertIn("arguments.tool_or_method_to_top_level_tool", artifact["normalization_rules_applied"])

    def test_query_literal_literature_search_is_semantically_invalid(self):
        self.assertTrue(validate_search_query("literature_search", "broad runtime topic"))

    def test_unconfigured_web_search_absent_from_model_visible_tools(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.environ.pop("RESEARCH_WEB_SEARCH_PROVIDER", None)
            try:
                runtime = GenericResearchRuntime(LocalArtifactStore(Path(d) / "runs"), work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
                self.assertNotIn("web_search", runtime._tool_cards())
            finally:
                if old is not None:
                    os.environ["RESEARCH_WEB_SEARCH_PROVIDER"] = old

    def test_guided_loop_exhaustion_creates_clean_escalation(self):
        class LoopGateway:
            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                data = {"action": "tool", "tool": "literature_search", "arguments": {"query": "broad runtime topic"}}
                if semantic_validator:
                    errors = semantic_validator(data)
                    if errors:
                        raise StructuredGenerationExhausted("SEMANTIC_VALIDATION_FAILURE loop exhausted", [])
                return {"structured": data, "text": json.dumps(data), "model": "fake"}

        with tempfile.TemporaryDirectory() as d:
            old_steps = os.environ.get("MAX_LOCAL_AGENT_STEPS")
            old_tools = os.environ.get("MAX_TOOL_CALLS_PER_NODE")
            old_guidance = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["MAX_LOCAL_AGENT_STEPS"] = "1"
            os.environ["MAX_TOOL_CALLS_PER_NODE"] = "1"
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                store = LocalArtifactStore(Path(d) / "runs")
                state = create_run_state("run", "broad runtime topic")
                runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=LoopGateway())
                node = state["dag"]["nodes"]["question_discovery"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                for key, old in (("MAX_LOCAL_AGENT_STEPS", old_steps), ("MAX_TOOL_CALLS_PER_NODE", old_tools), ("RESEARCH_LOCAL_GUIDANCE", old_guidance)):
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
            self.assertIn(node["status"], {"COMPLETED", "WAITING_FOR_HUMAN"})

    def test_placeholder_search_queries_are_rejected_before_retrieval(self):
        self.assertTrue(placeholder_like("broad discovery query"))
        with tempfile.TemporaryDirectory() as d:
            provider = TopicAwareLiteratureProvider()
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What evidence exists for broad runtime topic?", "why_interesting": "testable", "falsifiability": "records can refute", "local_executability": "yes"}]
            state["search_strategy"] = {"queries": ["broad discovery query"]}
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=provider, gateway=FakeGateway())
            node = state["dag"]["nodes"]["evidence_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertNotIn("broad discovery query", provider.queries)
            self.assertIn("broad runtime topic", provider.queries)
            self.assertEqual(node["status"], "COMPLETED")

    def test_verified_metadata_does_not_imply_topical_relevance(self):
        records = UnrelatedLiteratureProvider().search("broad runtime topic")["records"]
        report = literature_relevance_report(records, "broad runtime topic", [])
        self.assertEqual(records[0]["verification_status"], "VERIFIED_METADATA")
        self.assertFalse(report["usable"])

    def test_unrelated_retrieval_fails_relevance_gate_and_creates_human_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            provider = UnrelatedLiteratureProvider()
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What evidence exists for broad runtime topic?", "why_interesting": "testable", "falsifiability": "records can refute", "local_executability": "yes"}]
            state["search_strategy"] = {"queries": ["broad runtime topic"]}
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=provider, gateway=FakeGateway())
            node = state["dag"]["nodes"]["evidence_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertEqual(state["status"], "WAITING_FOR_HUMAN")
            self.assertEqual(node["status"], "WAITING_FOR_HUMAN")
            self.assertFalse(state["literature_relevance"]["usable"])
            decision = state["decisions"][0]
            self.assertIn("LITERATURE_RETRIEVAL_UNSUCCESSFUL", decision["why_human_is_needed"])
            bundle = decision["external_reasoning_bundle"]
            context = json.loads((store.run_root("run") / bundle / "context.json").read_text(encoding="utf-8"))
            self.assertTrue(context["prior_stage_outputs"]["evidence_discovery_attempts"])
            self.assertIn("literature_relevance", context["prior_stage_outputs"])

    def test_irrelevant_planner_query_triggers_bounded_automatic_rewrite_success(self):
        with tempfile.TemporaryDirectory() as d:
            provider = TopicAwareLiteratureProvider()
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What evidence exists for broad runtime topic?", "why_interesting": "testable", "falsifiability": "records can refute", "local_executability": "yes"}]
            state["search_strategy"] = {"queries": ["unrelated molecular database"]}
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=provider, gateway=FakeGateway())
            node = state["dag"]["nodes"]["evidence_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertEqual(node["status"], "COMPLETED")
            self.assertTrue(state["literature_relevance"]["usable"])
            self.assertIn("broad runtime topic", provider.queries)

    def test_question_refinement_rejects_placeholder_candidates_and_empty_relevance(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = ["question", "why_interesting"]
            state["literature_cache"] = TopicAwareLiteratureProvider().search("broad runtime topic")["records"]
            state["literature_relevance"] = literature_relevance_report(state["literature_cache"], "broad runtime topic", state["candidate_questions"])
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertEqual(node["status"], "BLOCKED_MISSING_EVIDENCE")
            self.assertIsNone(state.get("selected_question"))

    def test_repair_invalid_evidence_preserves_history_and_invalidates_dependents(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = ["question", "why_interesting", "falsifiability", "local_executability"]
            state["search_strategy"] = {"queries": ["broad discovery query"]}
            state["literature_cache"] = UnrelatedLiteratureProvider().search("broad discovery query")["records"]
            state["literature_relevance"] = literature_relevance_report(state["literature_cache"], state["topic"], state["candidate_questions"])
            state["dag"]["nodes"]["question_discovery"]["status"] = "COMPLETED"
            state["dag"]["nodes"]["evidence_discovery"]["status"] = "COMPLETED"
            state["dag"]["nodes"]["question_refinement"]["status"] = "WAITING_FOR_HUMAN"
            state["decisions"].append({"decision_id": "D1", "status": "WAITING_FOR_HUMAN", "blocked_nodes": ["question_refinement"]})
            result = repair_invalid_evidence_relevance(state)
            self.assertTrue(result["repaired"])
            self.assertIn("question_discovery", result["invalidated_nodes"])
            self.assertEqual(len(state["invalidated_evidence_history"][0]["literature_cache"]), 1)
            self.assertEqual(state["literature_cache"], [])
            self.assertEqual(state["candidate_questions"], [])
            self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_UPSTREAM_EVIDENCE_REPAIR")

    def test_structured_generation_repair_preserves_relevant_literature(self):
        state = create_run_state("run", "broad runtime topic")
        records = TopicAwareLiteratureProvider().search("broad runtime topic")["records"]
        state["literature_cache"] = records
        state["literature_relevance"] = literature_relevance_report(records, state["topic"], [])
        state["status"] = "WAITING_FOR_HUMAN"
        node = state["dag"]["nodes"]["question_refinement"]
        node["status"] = "WAITING_FOR_HUMAN"
        node["failure_reason"] = "SCHEMA_VALIDATION_FAILURE local structured generation exhausted"
        state["decisions"].append({
            "decision_id": "D1",
            "status": "WAITING_FOR_HUMAN",
            "why_human_is_needed": "SCHEMA_VALIDATION_FAILURE local structured generation exhausted",
            "blocked_nodes": ["question_refinement"],
        })
        repaired = repair_recoverable_structured_generation_failures(state)
        self.assertEqual(repaired, ["question_refinement"])
        self.assertEqual(state["literature_cache"], records)
        self.assertEqual(node["status"], "PENDING")
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_STRUCTURED_DECODING_FIX")
        self.assertEqual(state["status"], "PLANNED_RESEARCH")

    def test_structured_generation_repair_restores_preserved_atomic_question(self):
        state = create_run_state("run", "broad runtime topic")
        records = TopicAwareLiteratureProvider().search("broad runtime topic")["records"]
        state["literature_cache"] = records
        state["candidate_questions"] = [{
            "question": "What evidence in retrieved literature can bound or test broad runtime topic?",
            "why_interesting": "Grounded in retrieved literature and suitable for later feasibility checks.",
            "falsifiability": "The retrieved evidence may fail to support a measurable or testable relationship.",
            "local_executability": "Initial retrieval and metadata checks can run with local/free tools.",
        }]
        state["budget"]["calls"].append({
            "stage": "question_discovery",
            "status": "SUCCESS",
            "parsed_response": {"question": "How does broad runtime topic metadata vary across observable conditions?"},
            "raw_response": '{"question":"How does broad runtime topic metadata vary across observable conditions?"}',
            "actual_model": "llama.cpp:local-default",
            "selected_configuration_id": "local-default",
            "actual_cost": 0.0,
        })
        state["status"] = "WAITING_FOR_HUMAN"
        state["dag"]["nodes"]["question_refinement"]["status"] = "WAITING_FOR_HUMAN"
        state["decisions"].append({
            "decision_id": "D1",
            "status": "WAITING_FOR_HUMAN",
            "why_human_is_needed": "SCHEMA_VALIDATION_FAILURE local structured generation exhausted",
            "blocked_nodes": ["question_refinement"],
        })
        repair_recoverable_structured_generation_failures(state)
        self.assertEqual(state["candidate_questions"][0]["question"], "How does broad runtime topic metadata vary across observable conditions?")
        self.assertEqual(state["candidate_question_field_provenance"][0]["question"]["origin"], "local_model")
        self.assertEqual(state["literature_cache"], records)

    def test_structured_generation_repair_handles_retryable_local_runtime_environment_failure(self):
        state = create_run_state("run", "broad runtime topic")
        state["status"] = "WAITING_FOR_HUMAN"
        node = state["dag"]["nodes"]["question_refinement"]
        node["status"] = "WAITING_FOR_HUMAN"
        node["failure_reason"] = "EXTERNAL_REASONING_REQUIRED: D1"
        state["budget"]["calls"].append({
            "stage": "question_refinement",
            "status": "FAILED",
            "failure_type": "MODEL_EXECUTION_FAILED",
            "schema_errors": ["failed to get a free port"],
            "actual_cost": 0.0,
        })
        state["decisions"].append({
            "decision_id": "D1",
            "status": "WAITING_FOR_HUMAN",
            "why_human_is_needed": "MODEL_EXECUTION_FAILED local structured generation exhausted",
            "blocked_nodes": ["question_refinement"],
        })
        repaired = repair_recoverable_structured_generation_failures(state)
        self.assertEqual(repaired, ["question_refinement"])
        self.assertEqual(node["status"], "PENDING")
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_STRUCTURED_DECODING_FIX")

    def test_core_relevance_gate_has_no_domain_specific_keywords(self):
        source = Path("src/research_runtime.py").read_text(encoding="utf-8").lower()
        for term in ("rooftop", "heat retention", "cryptography", "biology", "cancer genomics"):
            self.assertNotIn(term, source)

    def test_candidate_question_validation_is_topic_sensitive(self):
        empirical = candidate_question_semantic_validation(
            "What evidence in retrieved literature can bound or test soil moisture?",
            "soil moisture effects on plant growth",
        )
        self.assertFalse(empirical["substantive_question"])
        self.assertTrue(empirical["meta_research_question"])
        synthesis = candidate_question_semantic_validation(
            "What does the literature show about publication bias in clinical meta-analysis?",
            "evidence synthesis and publication bias in clinical meta-analysis",
        )
        self.assertTrue(synthesis["substantive_question"])

    def test_repair_artifact_reason_rejected_but_valid_low_score_is_not_execution_failure(self):
        errors = validate_atomic_planning_score({
            "score": 0.0,
            "reason": "The previous response was repaired into a JSON object satisfying the schema.",
        }, "ordinary empirical planning")
        self.assertTrue(any("procedural repair artifact" in error for error in errors))
        policy = external_reasoning_options([], "CANDIDATE_QUESTION_UNSUITABLE atomic scores below threshold")
        self.assertNotIn("could not complete", policy["question"].lower())

    def test_falsifiability_rejects_literature_availability_as_wrong_dimension(self):
        errors = validate_dimension_score({
            "score": 0.0,
            "reason": "There are not enough papers currently available in the retrieved literature.",
        }, "falsifiability", "ordinary empirical topic")
        self.assertTrue(any("observable result" in error for error in errors))
        valid = validate_dimension_score({
            "score": 0.8,
            "reason": "Observable measurements showing no association would count against the relationship.",
        }, "falsifiability", "ordinary empirical topic")
        self.assertEqual(valid, [])
        contaminated = validate_dimension_score({
            "score": 0.0,
            "reason": "There is no literature, so the relationship is not possible to measure or observe.",
        }, "falsifiability", "ordinary empirical topic")
        self.assertTrue(any("observable result" in error for error in contaminated))

    def test_novelty_uncertainty_and_empirical_leakage_semantics(self):
        uncertain = validate_novelty_assessment({
            "assessment": "uncertain",
            "reason": "Retrieved prior work is too limited to establish coverage of the exact comparison.",
            "confidence": 0.2,
        }, "ordinary empirical topic")
        self.assertEqual(uncertain, [])
        leakage = validate_novelty_assessment({
            "assessment": "plausible_gap",
            "reason": "The intervention reduces the measured outcome and therefore this research gap is novel.",
            "confidence": 0.8,
        }, "ordinary empirical topic")
        self.assertTrue(any("answers the empirical question" in error for error in leakage))
        false_gap = validate_novelty_assessment({
            "assessment": "plausible_gap",
            "reason": "No provided information about the plausible gap in prior work.",
            "confidence": 0.5,
        }, "ordinary empirical topic")
        self.assertTrue(any("must be assessed as uncertain" in error for error in false_gap))

    def test_query_normalization_preserves_raw_and_normalized_provenance(self):
        class Gateway:
            def __init__(self):
                self.calls = 0
            def generate_structured(self, state, request, **kwargs):
                self.calls += 1
                data = {"query": "broad_runtime_topic"} if self.calls == 1 else {
                    "question": "How does broad runtime topic performance vary across observable conditions?"
                }
                validator = kwargs.get("semantic_validator")
                if validator:
                    assert not validator(data)
                return {"structured": data, "model": "fake-local"}

        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                state = create_run_state("run", "broad runtime topic")
                provider = TopicAwareLiteratureProvider()
                runtime = GenericResearchRuntime(
                    LocalArtifactStore(Path(d) / "runs"), Path(d) / "work",
                    provider, Gateway(),
                )
                node = state["dag"]["nodes"]["question_discovery"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else:
                    os.environ["RESEARCH_LOCAL_GUIDANCE"] = old
            self.assertEqual(state["search_strategy"]["raw_query"], "broad_runtime_topic")
            self.assertEqual(state["search_strategy"]["normalized_query"], "broad runtime topic")
            self.assertEqual(provider.queries[0], "broad runtime topic")

    def test_question_discovery_uses_candidate_question_generation_route(self):
        class RoutingGateway(FakeGateway):
            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                self.called = True
                self.task_class = request.task_class
                response = super().generate_structured(state, request, required_keys, estimated_cost, max_repairs, schema, semantic_validator)
                state["budget"]["calls"][-1]["actual_model"] = "llama.cpp:local-default:Q3_K_M:fp16:1024"
                return response

        with tempfile.TemporaryDirectory() as d:
            gateway = RoutingGateway()
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=gateway)
            node = state["dag"]["nodes"]["question_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertTrue(gateway.called)
            self.assertEqual(gateway.task_class, "candidate_question_generation")
            self.assertFalse(state["decisions"])
            self.assertEqual(state["budget"]["calls"][0]["actual_model"], "llama.cpp:local-default:Q3_K_M:fp16:1024")

    def test_feasibility_node_uses_research_feasibility_task_class(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["feasibility_analysis"]
            node["status"] = "LEASED"
            state["selected_question"] = "What patterns appear in the retrieved literature metadata?"
            state["literature_cache"] = FakeLiteratureProvider().search("q")["records"]
            runtime.execute(state, node)
            self.assertEqual(state["budget"]["calls"][0]["task_class"], "research_feasibility_analysis")

    def test_high_guidance_feasibility_uses_atomic_planning_and_supervisor_matching(self):
        class AtomicFeasibilityGateway:
            def __init__(self):
                self.requests = []
                self.requirements = 0

            def generate_structured(self, state, request, schema=None, semantic_validator=None, **kwargs):
                self.requests.append(request)
                if request.semantic_task == "feasibility_operationalization":
                    data = {"observable_test": "Compare measured outcomes across exposed and comparison units."}
                elif request.semantic_task == "feasibility_route_generation":
                    prior = sum(req.semantic_task == "feasibility_route_generation" for req in self.requests[:-1])
                    data = ({"approach": "secondary_data_analysis", "reason": "A dataset analysis route can test the stated observable comparison without asserting results."}
                            if prior == 0 else
                            {"approach": "simulation", "reason": "A computational model route can test the observable comparison without asserting results."})
                elif request.semantic_task == "feasibility_requirement_generation":
                    self.requirements += 1
                    data = {"requirement_type": "method", "requirement": "A reproducible procedure for applying the planned comparison"}
                else:
                    data = {"fit": "good", "reason": "The route directly addresses the observable comparison in the selected question."}
                if semantic_validator:
                    assert not semantic_validator(data)
                state.setdefault("budget", {"calls": [], "llm_usd": 0.0, "strong_calls": 0})["calls"].append({
                    "stage": request.stage, "task_class": request.task_class,
                    "semantic_task": request.semantic_task, "status": "SUCCESS", "actual_cost": 0.0,
                })
                return {"structured": data, "model": "fake-atomic"}

        with tempfile.TemporaryDirectory() as d:
            old = os.environ.get("RESEARCH_LOCAL_GUIDANCE")
            os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try:
                gateway = AtomicFeasibilityGateway()
                store = LocalArtifactStore(Path(d) / "runs")
                state = create_run_state("run", "broad runtime topic")
                state["selected_question"] = "How do measured broad runtime outcomes differ across observable conditions?"
                state["literature_cache"] = TopicAwareLiteratureProvider().search("broad runtime topic")["records"]
                state["literature_relevance"] = literature_relevance_report(state["literature_cache"], state["topic"], [])
                runtime = GenericResearchRuntime(store, Path(d) / "work", FakeLiteratureProvider(), gateway)
                node = state["dag"]["nodes"]["feasibility_analysis"]
                node["status"] = "LEASED"
                runtime.execute(state, node)
            finally:
                if old is None:
                    os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else:
                    os.environ["RESEARCH_LOCAL_GUIDANCE"] = old
            self.assertEqual(node["status"], "COMPLETED")
            self.assertTrue(all(req.task_class == "candidate_question_generation" for req in gateway.requests))
            self.assertFalse(any(req.task_class == "research_feasibility_analysis" for req in gateway.requests))
            self.assertEqual([req.semantic_task for req in gateway.requests], [
                "feasibility_operationalization", "feasibility_route_generation",
                "feasibility_requirement_generation", "feasibility_scientific_fit",
                "feasibility_route_generation", "feasibility_requirement_generation", "feasibility_scientific_fit",
            ])
            self.assertEqual(state["feasibility_input_snapshot"]["actual_datasets"], [])
            data_match = state["feasibility_resource_matches"][0]
            self.assertNotEqual(data_match["status"], "AVAILABLE_VERIFIED")
            self.assertEqual(data_match["origin"], "deterministic_route_semantics")
            self.assertEqual(data_match["availability_origin"], "deterministic_supervisor")
            self.assertEqual(state["empirical_evidence_path"], "CONDITIONAL")
            self.assertEqual(state["research_modality_plan"]["required_evidence_modalities"], ["secondary_dataset_analysis"])
            self.assertEqual(state["research_modality_plan"]["automation_closure"], "CONDITIONAL")
            self.assertEqual(state["research_spec"]["feasibility_verdict"], "PARTIAL")
            self.assertTrue(state["feasibility_unresolved_requirements"])
            self.assertTrue(all(call.get("semantic_task") for call in state["budget"]["calls"]))

    def test_feasibility_atomicization_repair_preserves_upstream_state(self):
        state = create_run_state("run", "ordinary empirical topic")
        base = TopicAwareLiteratureProvider().search("ordinary empirical topic")["records"]
        records = [{**base[index % len(base)], "identifier": f"record-{index}"} for index in range(15)]
        state["literature_cache"] = records
        state["selected_question"] = "How do measured outcomes vary across observable conditions?"
        state["candidate_evaluations"] = [{
            "question": state["selected_question"], "feasibility": 0.8, "novelty_potential": 0.5,
            "falsifiability": 0.8, "evidence_accessibility": 0.8,
            "rationale": "The validated candidate passed all atomic planning dimensions.",
        }]
        state["question_refinement_rationale"] = "The candidate passed atomic refinement and remains selected."
        for node_id in ("question_discovery", "evidence_discovery", "question_refinement"):
            state["dag"]["nodes"][node_id]["status"] = "COMPLETED"
        state["dag"]["nodes"]["feasibility_analysis"]["status"] = "WAITING_FOR_HUMAN"
        state["status"] = "WAITING_FOR_HUMAN"
        state["decisions"].append({
            "decision_id": "D71f05f65", "status": "WAITING_FOR_HUMAN",
            "why_human_is_needed": "NO_ELIGIBLE_LOCAL_MODEL task_class=research_feasibility_analysis",
            "blocked_nodes": ["feasibility_analysis"],
        })
        original_question = state["selected_question"]
        repaired = repair_recoverable_structured_generation_failures(state)
        self.assertIn("feasibility_analysis", repaired)
        self.assertEqual(len(state["literature_cache"]), 15)
        self.assertEqual(state["selected_question"], original_question)
        self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "COMPLETED")
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_FEASIBILITY_ATOMICIZATION_REPAIR")

    def test_unknown_llm_task_class_blocks_engineering_not_human(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_discovery"]
            node["llm_task_class"] = "question_discovery"
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertEqual(state["status"], "BLOCKED_ENGINEERING_REQUIRED")
            self.assertIn("UNKNOWN_LLM_TASK_CLASS", node["failure_reason"])
            self.assertFalse(state["decisions"])

    def test_capability_requirement_to_generated_skill_execution_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            store, state = run_runtime_to_completion(d)
            manifest = store.load_manifest("run")
            paths = {a["path"] for a in manifest["artifacts"]}
            self.assertIn("analysis/literature_metrics.json", paths)
            self.assertTrue(state["skill_registry"])

    def test_generated_skill_invalid_output_rejected_by_validator_followup(self):
        with tempfile.TemporaryDirectory() as d:
            store, state = run_runtime_to_completion(d, count=1)
            self.assertEqual(state["status"], "BLOCKED_INVALID_METHOD")
            self.assertTrue(state.get("unresolved_findings"))

    def test_artifact_to_claim_to_adversarial_no_followup_needed(self):
        with tempfile.TemporaryDirectory() as d:
            _, state = run_runtime_to_completion(d)
            self.assertFalse(state.get("adversarial_findings"))
            self.assertEqual(state["claim_evidence_ledger"]["claims"][0]["status"], "VERIFIED_TOOL_OUTPUT")

    def test_original_experiment_to_independent_replication_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            _, state = run_runtime_to_completion(d)
            self.assertEqual(state["replication_status"], "PASSED")
            self.assertEqual(state["replication_reports"][0]["verdict"], "PASS")

    def test_valid_metadata_substudy_does_not_imply_parent_completion(self):
        with tempfile.TemporaryDirectory() as d:
            store, state = run_runtime_to_completion(d)
            self.assertEqual(state["status"], "PARTIAL_RESEARCH")
            self.assertEqual(state["selected_objective_coverage"]["status"], "INSUFFICIENT")
            self.assertEqual(verify_research_run(store, "run")["status"], "WAITING")

    def test_missing_real_evidence_never_research_complete(self):
        with tempfile.TemporaryDirectory() as d:
            store, state = run_runtime_to_completion(d, count=0)
            self.assertNotEqual(state["status"], "RESEARCH_COMPLETE")
            self.assertEqual(verify_research_run(store, "run")["status"], "WAITING")

    def test_local_inference_insufficiency_creates_external_reasoning_decision(self):
        class InsufficientGateway:
            def generate_structured(self, *args, **kwargs):
                raise RuntimeError("no eligible local model configuration for task")

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=InsufficientGateway())
            node = state["dag"]["nodes"]["question_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            store.atomic_update_state("run", state)
            self.assertEqual(state["status"], "WAITING_FOR_HUMAN")
            self.assertTrue(state["decisions"])
            manifest_paths = {a["path"] for a in store.load_manifest("run")["artifacts"]}
            self.assertTrue(any(p.startswith("external_reasoning/question_discovery/") and p.endswith("/prompt.md") for p in manifest_paths))
            self.assertIn("external_reasoning_bundle", state["decisions"][0])

    def test_external_reasoning_retry_uses_unique_immutable_bundle(self):
        class InsufficientGateway:
            def generate_structured(self, *args, **kwargs):
                raise RuntimeError("NO_ELIGIBLE_LOCAL_MODEL task_class=candidate_question_generation")

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=InsufficientGateway())
            node = state["dag"]["nodes"]["question_discovery"]
            for _ in range(2):
                node["status"] = "LEASED"
                runtime.execute(state, node)
                node["status"] = "LEASED"
            manifest_paths = {a["path"] for a in store.load_manifest("run")["artifacts"]}
            prompts = [p for p in manifest_paths if p.startswith("external_reasoning/question_discovery/") and p.endswith("/prompt.md")]
            self.assertEqual(len(prompts), 2)

    def test_profile_root_is_separate_from_research_artifact_root(self):
        from src.local_inference import LocalInferenceManager, LocalLLMProvider
        from src.llm_gateway import ModelGateway

        class ProfileRuntime:
            name = "llama.cpp"
            def discover(self):
                return {"runtime": self.name, "available": True, "capabilities": {}}
            def supports(self, model):
                return True
            def generate(self, prompt, config):
                return {
                    "text": json.dumps({"candidate_questions": [{"question": "q"}], "search_queries": ["q"]}),
                    "duration_seconds": 0.01,
                    "model": config["id"],
                    "diagnostic": {"timing_data": {"prompt_tokens_per_second": 10, "generation_tokens_per_second": 20}},
                }

        class ProfileRuntimeRegistry:
            def __init__(self, runtime):
                self.runtime = runtime
            def discover(self):
                return [self.runtime.discover()]
            def first_supporting(self, model):
                return self.runtime

        with tempfile.TemporaryDirectory() as d:
            profile_root = Path(d) / "profile-root"
            research_root = Path(d) / "research-root"
            profile_dir = profile_root / "local_inference"
            profile_dir.mkdir(parents=True)
            profile = {
                "configurations": [{
                    "id": "llama.cpp:local-default:Q3_K_M:fp16:1024",
                    "model": {"id": "local-default", "path": "/tmp/model.gguf", "format": "gguf", "quantization": "Q3_K_M"},
                    "context": 1024,
                    "kv_quantization": {"id": "fp16"},
                    "eligible_task_classes": ["metadata_extraction", "candidate_question_generation"],
                    "measured_capability": {"structured_output": 1.0, "planning": 1.0},
                    "ram_estimate": 1,
                }],
                "routing_profile": {"task_assignments": {"candidate_question_generation": "llama.cpp:local-default:Q3_K_M:fp16:1024"}},
            }
            (profile_dir / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            manager = LocalInferenceManager(root=profile_root, runtime_registry=ProfileRuntimeRegistry(ProfileRuntime()))
            provider = LocalLLMProvider(manager=manager)
            gateway = ModelGateway(provider=provider)
            store = LocalArtifactStore(research_root)
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=gateway)
            node = state["dag"]["nodes"]["question_discovery"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertEqual(state["budget"]["calls"][0]["actual_model"], "llama.cpp:local-default:Q3_K_M:fp16:1024")
            self.assertEqual(state["budget"]["calls"][0]["actual_cost"], 0.0)
            self.assertEqual(state["budget"]["llm_usd"], 0.0)
            self.assertEqual(state["budget"]["calls"][0]["local_profile"]["path"], str(profile_dir / "profile.json"))

    def test_exception_during_external_reasoning_does_not_leave_lease(self):
        from src.research_state import finalize_leased_node_after_exception

        state = create_run_state("run", "topic")
        node = state["dag"]["nodes"]["question_discovery"]
        node["status"] = "LEASED"
        finalize_leased_node_after_exception(state, node["node_id"], "boom")
        self.assertEqual(node["status"], "FAILED")
        self.assertIsNone(node["lease"])

    def test_structured_generation_exhaustion_waits_for_human_not_failed(self):
        class ExhaustedGateway:
            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                state.setdefault("budget", {"calls": [], "llm_usd": 0.0, "strong_calls": 0}).setdefault("calls", []).append({
                    "stage": request.stage,
                    "task_class": request.task_class,
                    "actual_model": "llama.cpp:local-default:Q3_K_M:q4:1024",
                    "status": "FAILED",
                    "failure_type": "MODEL_OUTPUT_INVALID",
                    "raw_response": '{"wrong":1}',
                    "schema_errors": ["missing required key: selected_question"],
                    "actual_cost": 0.0,
                })
                raise StructuredGenerationExhausted("MODEL_OUTPUT_INVALID local structured generation exhausted", state["budget"]["calls"])

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What patterns appear in the retrieved literature metadata?", "why_interesting": "testable", "falsifiability": "metadata can refute", "local_executability": "yes"}]
            seed_relevant_literature(state)
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=ExhaustedGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            store.atomic_update_state("run", state)
            self.assertEqual(state["status"], "WAITING_FOR_HUMAN")
            self.assertEqual(node["status"], "WAITING_FOR_HUMAN")
            self.assertNotEqual(state["status"], "FAILED")
            self.assertEqual(state["budget"]["calls"][0]["actual_cost"], 0.0)
            self.assertTrue(state["decisions"])
            bundle = state["decisions"][0]["external_reasoning_bundle"]
            context_path = store.run_root("run") / bundle / "context.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertEqual(context["failed_local_attempts"][0]["raw_response"], '{"wrong":1}')
            self.assertIsNone(state.get("selected_question"))

    def test_external_reasoning_bundle_exports_actual_response_schema_and_contract_separately(self):
        class ExhaustedGateway:
            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                raise StructuredGenerationExhausted("SCHEMA_VALIDATION_FAILURE local structured generation exhausted", [])

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What patterns appear in the retrieved literature metadata?", "why_interesting": "testable", "falsifiability": "metadata can refute", "local_executability": "yes"}]
            seed_relevant_literature(state)
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=ExhaustedGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            bundle = state["decisions"][0]["external_reasoning_bundle"]
            root = store.run_root("run") / bundle
            response_schema = json.loads((root / "response_schema.json").read_text(encoding="utf-8"))
            artifact_contract = json.loads((root / "artifact_contract.json").read_text(encoding="utf-8"))
            context = json.loads((root / "context.json").read_text(encoding="utf-8"))
            prompt = (root / "prompt.md").read_text(encoding="utf-8")
            selfEqual = self.assertEqual
            selfEqual(response_schema, QUESTION_REFINEMENT_SCHEMA)
            self.assertEqual(response_schema_for_node("question_refinement"), response_schema)
            self.assertNotEqual(response_schema, artifact_contract)
            self.assertEqual(artifact_contract["outputs"], ["specification.json"])
            self.assertIn("Candidate research questions", prompt)
            self.assertIn("Relevant retrieved literature", prompt)
            self.assertEqual(len(context["candidate_questions"]), 1)
            self.assertEqual(len(context["relevant_literature"]), 2)
            self.assertNotIn("dag", context)

    def test_human_question_refinement_response_uses_same_schema_and_semantic_validation(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What patterns appear in the retrieved literature metadata?", "why_interesting": "testable", "falsifiability": "metadata can refute", "local_executability": "yes"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            runtime._external_reasoning_required(state, node, "SCHEMA_VALIDATION_FAILURE local structured generation exhausted")
            decision_id = state["decisions"][0]["decision_id"]
            invalid = json.dumps({"selected_question": "non-empty string", "candidate_evaluations": [], "rationale": "short"})
            with self.assertRaises(ValueError):
                submit_research_decision(store, state, decision_id, "A", invalid)
            self.assertEqual(state["decisions"][0]["status"], "WAITING_FOR_HUMAN")
            valid = {
                "selected_question": "What patterns appear in the retrieved literature metadata?",
                "candidate_evaluations": [{
                    "question": "What patterns appear in the retrieved literature metadata?",
                    "feasibility": 0.9,
                    "novelty_potential": 0.4,
                    "falsifiability": 0.8,
                    "evidence_accessibility": 0.9,
                    "rationale": "The retrieved metadata can be analyzed and independently checked.",
                }],
                "rationale": "This question is scoped, locally executable, and tied to retrieved metadata.",
            }
            decision = submit_research_decision(store, state, decision_id, "A", json.dumps(valid))
            self.assertEqual(decision["status"], "RESOLVED")
            self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "COMPLETED")
            self.assertEqual(state["selected_question"], valid["selected_question"])

    def test_regenerating_external_reasoning_bundle_is_immutable_and_updates_decision(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            runtime._external_reasoning_required(state, node, "SCHEMA_VALIDATION_FAILURE local structured generation exhausted")
            decision_id = state["decisions"][0]["decision_id"]
            old_bundle = state["decisions"][0]["external_reasoning_bundle"]
            result = regenerate_external_reasoning_bundle(store, state, decision_id)
            new_bundle = result["external_reasoning_bundle"]
            self.assertNotEqual(old_bundle, new_bundle)
            self.assertTrue((store.run_root("run") / old_bundle / "prompt.md").exists())
            self.assertTrue((store.run_root("run") / new_bundle / "prompt.md").exists())
            self.assertEqual(state["decisions"][0]["external_reasoning_bundle"], new_bundle)
            self.assertIn(old_bundle, state["decisions"][0]["superseded_external_reasoning_bundles"])

    def test_atomic_decision_uses_own_contract_and_installs_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "generic bounded computational topic")
            state["computational_experimental_skeleton"] = {
                "original_candidate": "What changes under bounded inputs?",
                "independent_variable": {"display_text": "input limit", "canonical_id": "input_limit",
                                         "control_type": "DIRECT_INPUT"},
                "dependent_measurement": {"measurement_kind": "output_value"},
            }
            state["measurement_kind_attempt_history"] = [{
                "measurement_kind": "output_value", "status": "UNUSABLE_SOURCE_MISMATCH"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work",
                                             literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            node["status"] = "LEASED"
            contract = computational_measurement_decision_contract(state)
            runtime._external_reasoning_required(state, node, "atomic exhaustion", contract)
            decision_id = state["decisions"][0]["decision_id"]
            prior_skeleton = json.loads(json.dumps(state["computational_experimental_skeleton"]))
            decision = submit_research_decision(
                store, state, decision_id, "A", json.dumps({"measurement_kind": "runtime"}))
            self.assertEqual(decision["status"], "RESOLVED")
            self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "PENDING")
            self.assertNotEqual(state["dag"]["nodes"]["question_refinement"]["status"], "COMPLETED")
            self.assertEqual(state["computational_experimental_skeleton"], prior_skeleton)
            self.assertEqual(state["computational_measurement_recovery"]["external_measurement_kind"], "runtime")
            self.assertEqual(state["computational_measurement_recovery"]["skeleton"], prior_skeleton)
            response_path = store.run_root("run") / decision["external_response_artifact"]
            response = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(response["origin"], "EXTERNAL_REASONING")
            self.assertEqual(response["parsed_response"], {"measurement_kind": "runtime"})
            self.assertEqual(response["continuation_target"], "question_refinement")
            self.assertNotIn("actual_model", response)
            lifecycle = state["external_decision_continuations"][0]
            self.assertEqual(lifecycle["status"], "CONTINUATION_PENDING")
            self.assertEqual([event["event"] for event in lifecycle["events"]],
                             ["RESPONSE_VALIDATED", "RESPONSE_PERSISTED"])
            self.assertEqual(lifecycle["response_artifact"], decision["external_response_artifact"])

    def test_external_continuation_budget_is_separate_from_exhausted_generation_budget(self):
        state = create_run_state("run", "topic")
        state["budget"]["calls"] = [{"status": "SUCCESS"}]
        manager = LLMBudgetManager(max_agent_iterations=1)
        ok, reason = manager.can_spend(state, 0.0, "CHEAP", "measurement_informativeness")
        self.assertFalse(ok)
        self.assertEqual(reason, "MAX_AGENT_ITERATIONS")
        state["active_external_continuation"] = {
            "status": "CONTINUATION_STARTED",
            "downstream_semantic_allowance": {"semantic_tasks": ["measurement_informativeness"],
                                               "used_semantic_tasks": []}}
        before_calls = len(state["budget"]["calls"])
        ok, reason = manager.can_spend(state, 0.0, "CHEAP", "measurement_informativeness")
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(len(state["budget"]["calls"]), before_calls)
        self.assertEqual(state["active_external_continuation"]["downstream_semantic_allowance"]["used_semantic_tasks"],
                         ["measurement_informativeness"])

    def test_external_continuation_reconciliation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "generic bounded computational topic")
            state["computational_experimental_skeleton"] = {
                "original_candidate": "What changes under bounded inputs?",
                "independent_variable": {"display_text": "input limit", "canonical_id": "input_limit",
                                         "control_type": "DIRECT_INPUT"}}
            state["measurement_kind_attempt_history"] = [{
                "measurement_kind": "output_value", "status": "UNUSABLE_SOURCE_MISMATCH"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work",
                                             literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            runtime._external_reasoning_required(
                state, node, "atomic exhaustion", computational_measurement_decision_contract(state))
            decision_id = state["decisions"][0]["decision_id"]
            submit_research_decision(store, state, decision_id, "A", json.dumps({"measurement_kind": "runtime"}))
            artifact_count = len(list((store.run_root("run") / "external_responses" / decision_id).glob("*.json")))
            state["dag"]["nodes"]["question_refinement"].update({"status": "FAILED", "failure_reason": "MAX_AGENT_ITERATIONS"})
            state.pop("computational_measurement_recovery", None)
            first = reconcile_external_decision_continuation(store, state, decision_id)
            second = reconcile_external_decision_continuation(store, state, decision_id)
            self.assertEqual(first[0]["continuation_id"], second[0]["continuation_id"])
            self.assertEqual(len(state["external_decision_continuations"]), 1)
            self.assertEqual(len(list((store.run_root("run") / "external_responses" / decision_id).glob("*.json"))), artifact_count)
            self.assertEqual(state["computational_measurement_recovery"]["external_measurement_kind"], "runtime")
            self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "PENDING")

    def test_recoverable_engineering_continuation_rechecks_current_backend(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "generic computational topic")
            state["computational_experimental_skeleton"] = {
                "original_candidate": "What changes?",
                "independent_variable": {"display_text": "input limit", "canonical_id": "input_limit",
                                         "control_type": "DIRECT_INPUT"}}
            state["measurement_kind_attempt_history"] = [{"measurement_kind": "output_value",
                                                          "status": "UNUSABLE_SOURCE_MISMATCH"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work",
                                             literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            runtime._external_reasoning_required(
                state, node, "atomic exhaustion", computational_measurement_decision_contract(state))
            decision_id = state["decisions"][0]["decision_id"]
            submit_research_decision(store, state, decision_id, "A", json.dumps({"measurement_kind": "runtime"}))
            lifecycle = state["external_decision_continuations"][0]
            node = state["dag"]["nodes"]["question_refinement"]
            lifecycle["status"] = "CONTINUATION_BLOCKED_ENGINEERING"
            lifecycle["events"].append({"event": "CONTINUATION_EXCEPTION",
                                        "failure_class": "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"})
            node.update({"status": "BLOCKED_ENGINEERING_REQUIRED", "failure_reason": "old unified failure"})
            state["status"] = "BLOCKED_ENGINEERING_REQUIRED"
            state["engineering_requests"] = [{"status": "OPEN", "problem": "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE"}]
            direct = {"backend_id": "llama.cpp:DIRECT_CLI:llama-cli", "invocation_mode": "DIRECT_CLI",
                      "backend_compatibility": "COMPATIBLE", "available": True,
                      "executable": "/new/llama-cli"}
            result = reconcile_external_decision_continuation(
                store, state, decision_id,
                backend_probe=lambda: {"status": "AVAILABLE", "selected_backend": direct,
                                       "discovered_backends": [direct], "checked_at": "now"})
            self.assertEqual(result[0]["action"], "RETRY_STARTED")
            self.assertEqual(result[0]["selected_backend"]["invocation_mode"], "DIRECT_CLI")
            self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "PENDING")
            self.assertEqual(state["computational_measurement_recovery"]["external_measurement_kind"], "runtime")
            self.assertEqual(lifecycle["continuation_id"], result[0]["continuation_id"])
            self.assertEqual(lifecycle["attempts"][0]["status"], "RETRY_READY")
            self.assertEqual(lifecycle["attempts"][0]["precondition_recheck"]["selected_backend"], direct)
            self.assertTrue(any(event.get("failure_class") == "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"
                                for event in lifecycle["events"]))

    def test_blocked_continuation_stays_blocked_when_backend_recheck_fails(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "topic")
            state["computational_experimental_skeleton"] = {
                "independent_variable": {"display_text": "input limit", "canonical_id": "input_limit"}}
            state["measurement_kind_attempt_history"] = [{"measurement_kind": "output_value",
                                                          "status": "UNUSABLE_SOURCE_MISMATCH"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work",
                                             literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            runtime._external_reasoning_required(
                state, node, "atomic exhaustion", computational_measurement_decision_contract(state))
            decision_id = state["decisions"][0]["decision_id"]
            submit_research_decision(store, state, decision_id, "A", json.dumps({"measurement_kind": "runtime"}))
            lifecycle = state["external_decision_continuations"][0]
            node = state["dag"]["nodes"]["question_refinement"]
            lifecycle["status"] = "CONTINUATION_BLOCKED_ENGINEERING"
            node.update({"status": "BLOCKED_ENGINEERING_REQUIRED", "failure_reason": "runtime unavailable"})
            result = reconcile_external_decision_continuation(
                store, state, decision_id,
                backend_probe=lambda: {"status": "UNAVAILABLE", "selected_backend": None,
                                       "discovered_backends": [], "checked_at": "now"})
            self.assertEqual(result[0]["action"], "SKIPPED_STILL_BLOCKED")
            self.assertEqual(lifecycle["attempts"][0]["status"], "SKIPPED_STILL_BLOCKED")
            self.assertEqual(node["status"], "BLOCKED_ENGINEERING_REQUIRED")

    def test_report_surfaces_external_continuation_and_node_failure(self):
        state = create_run_state("run", "topic")
        state["external_decision_continuations"] = [{
            "decision_id": "D1", "semantic_task": "task", "response_kind": "ATOMIC_SEMANTIC_RESPONSE",
            "parsed_response": {"value": "x"}, "response_validation": "PASSED",
            "response_artifact": "external_responses/D1/x.json", "continuation_type": "TYPE",
            "status": "CONTINUATION_SEMANTIC_REJECTED", "continuation_validation_result": ["invalid"]}]
        node = state["dag"]["nodes"]["question_refinement"]
        node.update({"status": "FAILED", "failure_reason": "explicit failure"})
        report = research_report(state)
        self.assertEqual(report["external_reasoning_responses"][0]["continuation_status"],
                         "CONTINUATION_SEMANTIC_REJECTED")
        self.assertEqual(report["node_failures"][0]["failure_reason"], "explicit failure")

    def test_atomic_decision_validation_failure_is_non_mutating_and_resubmittable(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "generic bounded computational topic")
            state["computational_experimental_skeleton"] = {
                "independent_variable": {"display_text": "input limit", "canonical_id": "input_limit",
                                         "control_type": "DIRECT_INPUT"}}
            state["measurement_kind_attempt_history"] = [{
                "measurement_kind": "output_value", "status": "UNUSABLE_SOURCE_MISMATCH"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work",
                                             literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            runtime._external_reasoning_required(
                state, node, "atomic exhaustion", computational_measurement_decision_contract(state))
            decision_id = state["decisions"][0]["decision_id"]
            before = json.loads(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "DECISION_RESPONSE_VALIDATION_FAILURE") as caught:
                submit_research_decision(store, state, decision_id, "A", json.dumps({"measurement_kind": "invalid"}))
            self.assertIn("ATOMIC_SEMANTIC_RESPONSE", str(caught.exception))
            self.assertNotIn("selected_question", str(caught.exception))
            self.assertEqual(state, before)
            self.assertEqual(state["decisions"][0]["status"], "WAITING_FOR_HUMAN")
            submit_research_decision(store, state, decision_id, "A", json.dumps({"measurement_kind": "runtime"}))
            self.assertEqual(state["decisions"][0]["status"], "RESOLVED")

    def test_atomic_decision_missing_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "topic")
            state["decisions"] = [{
                "decision_id": "Datomic", "status": "WAITING_FOR_HUMAN", "response_kind": "ATOMIC_SEMANTIC_RESPONSE",
                "external_reasoning_bundle": "external_reasoning/question_refinement/old",
                "blocked_nodes": ["question_refinement"]}]
            with self.assertRaisesRegex(ValueError, "DECISION_CONTRACT_ERROR"):
                submit_research_decision(store, state, "Datomic", "A", json.dumps({"measurement_kind": "runtime"}))
            self.assertEqual(state["decisions"][0]["status"], "WAITING_FOR_HUMAN")

    def test_atomic_bundle_persists_machine_readable_contract(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "generic bounded computational topic")
            state["computational_experimental_skeleton"] = {
                "independent_variable": {"display_text": "input limit", "canonical_id": "input_limit"}}
            state["measurement_kind_attempt_history"] = [{
                "measurement_kind": "output_value", "status": "UNUSABLE_SOURCE_MISMATCH"}]
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work",
                                             literature_provider=FakeLiteratureProvider(), gateway=FakeGateway())
            contract = computational_measurement_decision_contract(state)
            runtime._external_reasoning_required(
                state, state["dag"]["nodes"]["question_refinement"], "atomic exhaustion", contract)
            decision = state["decisions"][0]
            bundle_root = store.run_root("run") / decision["external_reasoning_bundle"]
            persisted = json.loads((bundle_root / "response_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, decision["response_contract"])
            self.assertEqual(persisted["response_kind"], "ATOMIC_SEMANTIC_RESPONSE")
            self.assertEqual(persisted["continuation"]["type"], "COMPUTATIONAL_MEASUREMENT_KIND_RECOVERY_V1")

    def test_structured_decoding_configuration_failure_blocks_engineering(self):
        class BrokenDecoderGateway:
            def generate_structured(self, state, request, required_keys=None, estimated_cost=0.0, max_repairs=1, schema=None, semantic_validator=None):
                attempts = [{
                    "stage": request.stage,
                    "task_class": request.task_class,
                    "actual_model": "llama.cpp:local-default:Q3_K_M:q4:1024",
                    "status": "FAILED",
                    "failure_type": "STRUCTURED_DECODING_CONFIGURATION_FAILURE",
                    "raw_response": "",
                    "stderr": "error initializing grammar sampler",
                    "actual_cost": 0.0,
                }]
                state.setdefault("budget", {"calls": [], "llm_usd": 0.0, "strong_calls": 0})["calls"].extend(attempts)
                raise StructuredDecodingConfigurationFailure("STRUCTURED_DECODING_CONFIGURATION_FAILURE local JSON-schema decoding failed", attempts)

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(Path(d) / "runs")
            state = create_run_state("run", "broad runtime topic")
            state["candidate_questions"] = [{"question": "What patterns appear in the retrieved literature metadata?", "why_interesting": "testable", "falsifiability": "metadata can refute", "local_executability": "yes"}]
            seed_relevant_literature(state)
            runtime = GenericResearchRuntime(store, work_root=Path(d) / "work", literature_provider=FakeLiteratureProvider(), gateway=BrokenDecoderGateway())
            node = state["dag"]["nodes"]["question_refinement"]
            node["status"] = "LEASED"
            runtime.execute(state, node)
            self.assertEqual(state["status"], "BLOCKED_ENGINEERING_REQUIRED")
            self.assertEqual(node["status"], "BLOCKED_ENGINEERING_REQUIRED")
            self.assertFalse(state["decisions"])

    def test_repair_recoverable_structured_failure_resets_node_and_cost(self):
        from src.research_runtime import repair_recoverable_structured_generation_failures

        state = create_run_state("run", "topic")
        node = state["dag"]["nodes"]["question_refinement"]
        node["status"] = "FAILED"
        node["failure_reason"] = "MODEL_OUTPUT_INVALID schema validation"
        state["status"] = "FAILED"
        state["budget"]["calls"].append({"stage": "question_refinement", "estimated_cost": 0.002, "status": "FAILED"})
        state["budget"]["llm_usd"] = 0.002
        repaired = repair_recoverable_structured_generation_failures(state)
        self.assertEqual(repaired, ["question_refinement"])
        self.assertEqual(node["status"], "PENDING")
        self.assertEqual(state["status"], "PLANNED_RESEARCH")
        self.assertEqual(state["budget"]["llm_usd"], 0.0)

    def test_persistent_local_runtime_failure_creates_engineering_request(self):
        with tempfile.TemporaryDirectory() as d:
            state = create_run_state("run", "generic topic")
            runtime = GenericResearchRuntime(LocalArtifactStore(Path(d) / "runs"), Path(d) / "work",
                                             FakeLiteratureProvider(), FakeGateway())
            runtime._node_question_discovery = lambda state, node: (_ for _ in ()).throw(
                LocalRuntimeInfrastructureFailure("LOCAL_INFERENCE_RUNTIME_UNAVAILABLE", [{
                    "failure_type": "TRANSIENT_LOCAL_RUNTIME_FAILURE", "actual_model": "local-q3"}]))
            node = state["dag"]["nodes"]["question_discovery"]
            runtime.execute(state, node)
            self.assertEqual(node["status"], "BLOCKED_ENGINEERING_REQUIRED")
            self.assertEqual(state["status"], "BLOCKED_ENGINEERING_REQUIRED")
            self.assertEqual(state["engineering_requests"][-1]["problem"], "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE")
            self.assertFalse(state["decisions"])

    def test_historical_variable_surface_reevaluation_recovers_without_model_call(self):
        state = create_run_state("run", "generic computational topic")
        state["status"] = "WAITING_FOR_HUMAN"
        node = state["dag"]["nodes"]["question_refinement"]
        node.update({"status": "WAITING_FOR_HUMAN", "failure_reason": "EXTERNAL_REASONING_REQUIRED: D34131d1d"})
        for raw in ("...", "input_quantity_limit", "processing_time"):
            state["budget"]["calls"].append({"status": "FAILED", "schema": {"required": ["variable"]},
                                               "parsed_response": {"variable": raw},
                                               "semantic_errors": ["old surface validator rejection"]})
        original_calls = list(state["budget"]["calls"])
        state["decisions"].append({"decision_id": "D34131d1d", "status": "WAITING_FOR_HUMAN",
            "blocked_nodes": ["question_refinement"],
            "why_human_is_needed": "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=fresh_variable_generation"})
        repaired = repair_recoverable_structured_generation_failures(state)
        self.assertIn("question_refinement", repaired)
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_SEMANTIC_SURFACE_NORMALIZATION_REPAIR")
        recovered = state["computational_measurement_recovery"]["recovered_historical_variable"]
        self.assertEqual(recovered["raw_model_value"], "input_quantity_limit")
        self.assertEqual(recovered["normalized_display_text"], "input quantity limit")
        self.assertEqual(recovered["semantic_source"], "HISTORICAL_MODEL_OUTPUT_REEVALUATED")
        self.assertEqual(state["budget"]["calls"], original_calls)
        records = state["semantic_value_reevaluations"][-1]["records"]
        self.assertEqual([r["semantic_validation_status"] for r in records], ["INVALID", "VALID", "ROLE_MISMATCH"])

    def test_semantic_revalidation_invalidates_previous_refinement_and_dependents(self):
        from src.research_runtime import repair_recoverable_structured_generation_failures

        state = create_run_state("run", "broad runtime topic")
        state["status"] = "WAITING_FOR_HUMAN"
        state["selected_question"] = {"not": "a string"}
        state["candidate_evaluations"] = []
        state["dag"]["nodes"]["question_discovery"]["status"] = "COMPLETED"
        state["dag"]["nodes"]["evidence_discovery"]["status"] = "COMPLETED"
        state["dag"]["nodes"]["question_refinement"]["status"] = "COMPLETED"
        state["dag"]["nodes"]["feasibility_analysis"]["status"] = "WAITING_FOR_HUMAN"
        state["decisions"].append({"decision_id": "D1", "status": "WAITING_FOR_HUMAN", "why_human_is_needed": "NO_ELIGIBLE_LOCAL_MODEL", "blocked_nodes": ["feasibility_analysis"]})
        repaired = repair_recoverable_structured_generation_failures(state)
        self.assertIn("question_refinement", repaired)
        self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "PENDING")
        self.assertEqual(state["dag"]["nodes"]["feasibility_analysis"]["status"], "PENDING")
        self.assertIsNone(state["selected_question"])
        self.assertEqual(state["dag"]["nodes"]["evidence_discovery"]["status"], "COMPLETED")
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_DEPENDENCY_REVALIDATION")

    def test_typed_requirement_validation_does_not_require_magic_words(self):
        requirement = {
            "requirement_type": "measurement",
            "requirement": "Observations sufficient to compare conditions across the specified design.",
        }
        self.assertEqual(validate_feasibility_requirement(requirement), [])
        self.assertTrue(validate_feasibility_requirement({"requirement": requirement["requirement"]}))
        self.assertTrue(validate_feasibility_requirement({"requirement_type": "data", "requirement": "data|measurement|software|compute|external|other"}))
        self.assertTrue(validate_feasibility_requirement({"requirement_type": "data", "requirement": "comparison_outcome_identifier"}))
        self.assertTrue(validate_feasibility_requirement({"requirement_type": "data", "requirement": "What relationship should this research investigate?"}))
        self.assertTrue(validate_feasibility_requirement({"requirement_type": "data", "requirement": "The response should identify a resource needed for the route."}))
        self.assertTrue(validate_feasibility_requirement({"requirement_type": "data", "requirement": '{"requirement_type":"data","requirement":"software"}'}))

    def test_route_inherent_requirements_and_empirical_policy_are_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = GenericResearchRuntime(LocalArtifactStore(Path(d) / "runs"), Path(d) / "work", FakeLiteratureProvider(), FakeGateway())
            primary = runtime._route_inherent_requirements("primary_measurement")
            self.assertEqual([item["requirement_type"] for item in primary], ["method", "measurement"])
            self.assertTrue(all(item["origin"] == "deterministic_route_semantics" for item in primary))
            secondary = runtime._route_inherent_requirements("secondary_data_analysis")
            self.assertEqual([item["requirement_type"] for item in secondary], ["data"])
            self.assertEqual(runtime._empirical_evidence_path("systematic_evidence_analysis", []), "NO")

    def test_typed_matching_never_verifies_a_model_proposal_without_registry_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = GenericResearchRuntime(LocalArtifactStore(Path(d) / "runs"), Path(d) / "work", FakeLiteratureProvider(), FakeGateway())
            snapshot = {"registered_capabilities": [], "registered_tools": ["python3"], "actual_datasets": [],
                        "network_provider_availability": {"web_search": False}}
            proposed = {"requirement_type": "data", "requirement": "Records for the planned comparison.",
                        "origin": "local_model_proposal", "importance": "optional"}
            match = runtime._match_feasibility_requirement(proposed, snapshot)
            self.assertEqual(match["status"], "MISSING")
            self.assertEqual(match["matched_artifact_ids"], [])
            self.assertEqual(match["evidence"]["matching_key"], "requirement_type")

    def test_typed_requirement_repair_preserves_resume_inputs_and_history(self):
        state = create_run_state("run", "ordinary empirical topic")
        state["selected_question"] = "How do ordinary empirical topic outcomes differ across two observable conditions?"
        state["literature_cache"] = [{"identifier": f"R{i}"} for i in range(15)]
        for node_id in ("question_discovery", "evidence_discovery", "question_refinement"):
            state["dag"]["nodes"][node_id]["status"] = "COMPLETED"
        state["candidate_questions"] = [{"question": state["selected_question"], "why_interesting": "Observable comparison",
                                         "falsifiability": "A null difference could count against it", "local_executability": "Requires resources"}]
        state["candidate_evaluations"] = [{"question": state["selected_question"], "feasibility": 0.8,
            "novelty_potential": 0.6, "falsifiability": 0.8, "evidence_accessibility": 0.7,
            "rationale": "The observable comparison is testable and its resource needs can be checked."}]
        state["question_refinement_rationale"] = "The candidate passed the required atomic dimensions."
        state["dag"]["nodes"]["feasibility_analysis"]["status"] = "WAITING_FOR_HUMAN"
        state["feasibility_input_snapshot"] = {"marker": "preserved"}
        state["feasibility_atomic_steps"] = [
            {"semantic_task": "feasibility_operationalization", "structured": {"observable_test": "Compare observable outcomes across two conditions."}},
            {"semantic_task": "feasibility_route_generation", "structured": {"approach": "primary_measurement", "reason": "Field observations provide the proposed comparison."}},
        ]
        state["decisions"].append({"decision_id": "D4583d8c9", "status": "WAITING_FOR_HUMAN",
            "why_human_is_needed": "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=feasibility_requirement_generation",
            "blocked_nodes": ["feasibility_analysis"]})
        repair_recoverable_structured_generation_failures(state)
        self.assertEqual(state["decisions"][-1]["status"], "INVALIDATED_TYPED_REQUIREMENT_ROUTE_SEMANTICS_REPAIR")
        self.assertEqual(state["feasibility_resume_context"]["route"]["approach"], "primary_measurement")
        self.assertEqual(state["feasibility_resume_context"]["input_snapshot"], {"marker": "preserved"})
        self.assertEqual(len(state["literature_cache"]), 15)
        self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
