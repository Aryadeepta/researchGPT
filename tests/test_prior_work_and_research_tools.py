import hashlib
import json
import os
import tempfile
import unittest
from email.message import Message
from pathlib import Path

from src.objective_coverage import selected_objective_coverage
from src.prior_work import (classify_prior_work_coverage, may_continue_without_literature,
                            normalize_research_query, novelty_claim_allowed,
                            question_scope_modalities, record_scope_refinement,
                            validate_bounded_computational_question)
from src.research_state import create_run_state
from src.research_runtime import repair_recoverable_structured_generation_failures
from src.research_runtime import (GenericResearchRuntime, legacy_falsifiability_from_testability,
                                  validate_atomic_semantic_value, validate_candidate_clarification,
                                  validate_controllable_variable_candidate)
from src.research_tools import (MultiProviderResearchSupervisor, ResearchToolRegistry,
                                classify_provider_search_outcome, provider_descriptor,
                                safe_fetch, validate_public_http_url)
from src.storage import LocalArtifactStore


class Provider:
    def __init__(self, records=None, error=None): self.records = records or []; self.error = error
    def search(self, query, limit=5):
        if self.error: raise self.error
        return {"provider": "fake", "query": query, "records": self.records[:limit]}


class Response:
    def __init__(self, body, content_type, url="https://example.org/source"):
        self.body = body; self.url = url; self.headers = Message(); self.headers["Content-Type"] = content_type
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit): return self.body[:limit]
    def geturl(self): return self.url


def public_resolver(*args, **kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


class PriorWorkTests(unittest.TestCase):
    def test_zero_results_is_unknown_not_nonexistence(self):
        result = classify_prior_work_coverage([{"status": "ZERO_RESULTS"}])
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIn("does not establish", result["interpretation"])
        self.assertEqual(result["novelty_status"], "NOT_ESTABLISHED")
        self.assertIsNone(result["novelty_score"])

    def test_unavailable_and_zero_results_are_distinct(self):
        self.assertEqual(classify_prior_work_coverage([{"status": "PROVIDER_UNAVAILABLE"}])["status"], "UNAVAILABLE")
        self.assertEqual(classify_prior_work_coverage([{"status": "ZERO_RESULTS"}])["status"], "UNKNOWN")

    def test_computation_can_continue_but_literature_modality_cannot(self):
        caps = [{"status": "AVAILABLE_VERIFIED", "produces_modalities": ["executable_computation"]}]
        allowed, assessment = may_continue_without_literature(["executable_computation"], caps)
        self.assertTrue(allowed); self.assertEqual(assessment["automation_closure"], "HIGH")
        self.assertFalse(may_continue_without_literature(["literature_metadata"], caps)[0])

    def test_universal_scope_requires_proof_and_explicit_narrowing_has_provenance(self):
        self.assertEqual(question_scope_modalities("Is the method always correct for every input?")["required_evidence_modalities"], ["formal_proof"])
        narrowed = record_scope_refinement("Is it always correct?", "Under a bounded finite regime, does it remain correct?", "executable test scope")
        self.assertEqual(narrowed["transition"], "EXPLICIT_SCOPE_REFINEMENT")
        self.assertTrue(question_scope_modalities(narrowed["refined_question"])["bounded"])

    def test_bounded_computational_question_rejects_repetition_and_accepts_comparison(self):
        bad = "What is the result under a bounded regime under a bounded regime under a bounded regime? And why?"
        self.assertTrue(validate_bounded_computational_question(bad))
        good = "Under a bounded finite input regime, how does algorithm A compare with algorithm B in runtime and correctness?"
        self.assertEqual(validate_bounded_computational_question(good), [])

    def test_novelty_fails_closed(self):
        self.assertFalse(novelty_claim_allowed({"status": "UNKNOWN"}))
        self.assertTrue(novelty_claim_allowed({"status": "SUFFICIENT"}))

    def test_query_normalization_preserves_distinct_value(self):
        raw = "bounded_integer_inputs"
        self.assertEqual(normalize_research_query(raw), "bounded integer inputs")
        self.assertNotEqual(raw, normalize_research_query(raw))

    def test_computational_result_and_novelty_are_independent(self):
        state = create_run_state("r", "generic computation")
        state["selected_question"] = "Under a bounded regime, does one configuration differ from another?"
        state["research_modality_plan"] = {"required_evidence_modalities": ["executable_computation"]}
        state["claim_evidence_ledger"]["claims"] = [{"claim_id": "C1", "status": "VERIFIED_TOOL_OUTPUT",
            "replication_status": "PASSED", "objective_relation": "DIRECT_ANSWER", "evidence_modality": "executable_computation"}]
        state["prior_work_coverage"] = classify_prior_work_coverage([{"status": "ZERO_RESULTS"}])
        coverage = selected_objective_coverage(state, {"artifacts": []})
        self.assertNotEqual(coverage["status"], "INSUFFICIENT")
        self.assertFalse(novelty_claim_allowed(state["prior_work_coverage"]))

    def test_real4_literature_dependency_decision_is_invalidated_without_history_loss(self):
        state = create_run_state("local_research_real4", "generic computational topic")
        state["status"] = "WAITING_FOR_HUMAN"
        state["dag"]["nodes"]["question_discovery"].update({"status": "WAITING_FOR_HUMAN", "failure_reason": "EXTERNAL_REASONING_REQUIRED: D2e8491b3"})
        state["guided_agent_steps"] = [{"microstep": "literature_search", "raw_query": "raw_query",
            "normalized_query": "raw query", "record_count": 0, "created_at": "t"}]
        state["decisions"] = [{"decision_id": "D2e8491b3", "status": "WAITING_FOR_HUMAN",
            "blocked_nodes": ["question_discovery"],
            "why_human_is_needed": "INSUFFICIENT_RELEVANT_EVIDENCE guided search executed but no relevant literature exists for question generation"}]
        repair_recoverable_structured_generation_failures(state)
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_LITERATURE_DEPENDENCY_REPAIR")
        self.assertEqual(state["research_source_attempts"][0]["raw_query"], "raw_query")
        self.assertEqual(state["prior_work_coverage"]["status"], "UNKNOWN")
        self.assertEqual(state["budget"]["llm_usd"], 0.0)

    def test_typed_testability_mapping_fails_closed(self):
        self.assertEqual(legacy_falsifiability_from_testability("TESTABLE")["value"], 1.0)
        self.assertEqual(legacy_falsifiability_from_testability("NOT_TESTABLE")["value"], 0.0)
        self.assertEqual(legacy_falsifiability_from_testability("UNCERTAIN")["value"], 0.0)
        self.assertFalse(legacy_falsifiability_from_testability("TESTABLE")["scientific_measurement"])

    def test_atomic_values_need_semantics_not_arbitrary_length(self):
        self.assertEqual(validate_atomic_semantic_value({"variable": "input size"}, "variable"), [])
        self.assertTrue(validate_atomic_semantic_value({"variable": "ok"}, "variable"))
        self.assertTrue(validate_atomic_semantic_value({"variable": "the JSON object schema"}, "variable"))
        self.assertTrue(validate_controllable_variable_candidate("number"))
        self.assertEqual(validate_controllable_variable_candidate("bounded input count"), [])
        self.assertTrue(validate_atomic_semantic_value({"measurement": "1000"}, "measurement"))
        self.assertTrue(validate_atomic_semantic_value({"observation": "Performance is greater."}, "observation"))
        original = "Under a bounded regime, what is the impact of varying input size on runtime?"
        self.assertTrue(validate_candidate_clarification({"question": original}, original, "bounded computation"))

    def test_D42_repair_preserves_discovery_and_attempt_history(self):
        state = create_run_state("local_research_real4", "generic computational topic")
        state["status"] = "WAITING_FOR_HUMAN"
        state["dag"]["nodes"]["evidence_discovery"]["status"] = "COMPLETED"
        state["dag"]["nodes"]["question_refinement"].update({"status": "WAITING_FOR_HUMAN", "failure_reason": "D42d2453b"})
        state["literature_cache"] = [{"identifier": str(i)} for i in range(10)]
        state["prior_work_coverage"] = {"status": "UNKNOWN", "novelty_status": "NOT_ESTABLISHED"}
        state["budget"]["calls"] = [{"stage": "question_refinement", "parsed_response": {"score": 112}, "actual_cost": 0.0}]
        state["decisions"] = [{"decision_id": "D42d2453b", "status": "WAITING_FOR_HUMAN",
            "blocked_nodes": ["question_refinement"], "why_human_is_needed": "SCHEMA_VALIDATION_FAILURE local structured generation exhausted"}]
        repair_recoverable_structured_generation_failures(state)
        self.assertEqual(state["decisions"][0]["status"], "INVALIDATED_COMPUTATIONAL_TESTABILITY_ATOMICIZATION_REPAIR")
        self.assertEqual(state["dag"]["nodes"]["evidence_discovery"]["status"], "COMPLETED")
        self.assertEqual(len(state["literature_cache"]), 10)
        self.assertEqual(state["budget"]["calls"][0]["parsed_response"]["score"], 112)

    def test_computational_refinement_uses_separate_typed_microsteps(self):
        class Gateway:
            def __init__(self): self.required = []
            def generate_structured(self, state, request, schema=None, **kwargs):
                key = schema["required"][0]; self.required.append(key)
                if key == "variable": data = {"variable": "bounded input count", "control_type": "DIRECT_INPUT"}
                elif key == "control_type": data = {"control_type": "DIRECT_INPUT"}
                elif key == "measurement": data = {"measurement": "execution time", "measurement_kind": "runtime"}
                elif key == "measurement_kind": data = {"measurement_kind": "runtime"}
                elif key == "observation": data = {"observation": "execution times differ across generated input counts"}
                elif "neutral observable" in request.prompt: data = {"assessment": "NEUTRAL_MEASUREMENT"}
                elif "materially constrain" in request.prompt: data = {"assessment": "INFORMATIVE"}
                else: data = {"assessment": "TESTABLE"}
                return {"structured": data, "model": "fake", "prompt": request.prompt}
        with tempfile.TemporaryDirectory() as root:
            state = create_run_state("r", "generic bounded computation")
            question = "Under a bounded finite regime, what is the impact of varying input count on execution time?"
            state["candidate_questions"] = [{"question": question, "why_interesting": "A controlled comparison is useful.",
                "falsifiability": "Measurements can distinguish answers.", "local_executability": "Python is available."}]
            state["candidate_evidence_contracts"] = {question: {"scope_type": "BOUNDED_COMPUTATIONAL",
                "required_evidence_modalities": ["executable_computation"]}}
            state["prior_work_coverage"] = {"status": "UNKNOWN", "novelty_status": "NOT_ESTABLISHED"}
            gateway = Gateway(); runtime = GenericResearchRuntime(LocalArtifactStore(root), work_root=Path(root) / "work",
                literature_provider=Provider(), gateway=gateway)
            node = state["dag"]["nodes"]["question_refinement"]; node["status"] = "LEASED"
            previous = os.environ.get("RESEARCH_LOCAL_GUIDANCE"); os.environ["RESEARCH_LOCAL_GUIDANCE"] = "high"
            try: runtime.execute(state, node)
            finally:
                if previous is None: os.environ.pop("RESEARCH_LOCAL_GUIDANCE", None)
                else: os.environ["RESEARCH_LOCAL_GUIDANCE"] = previous
            self.assertEqual(node["status"], "COMPLETED", node.get("failure_reason"))
            self.assertLess(gateway.required.index("variable"), gateway.required.index("measurement"))
            self.assertIn("control_type", gateway.required)
            self.assertIn("measurement_kind", gateway.required)
            self.assertNotIn("score", gateway.required)
            self.assertEqual(state["candidate_evaluations"][0]["testability_status"], "TESTABLE")
            self.assertEqual(state["candidate_evaluations"][0]["novelty_status"], "NOT_ESTABLISHED")
            self.assertTrue(state["computational_experimental_skeleton"]["not_full_experiment_contract"])


class ResearchToolTests(unittest.TestCase):
    def test_registry_rejects_model_tool_plans(self):
        registry = ResearchToolRegistry(); registry.register("search", "web_search", lambda request: request)
        with self.assertRaises(ValueError): registry.execute("search", {"query": "x", "shell": "curl x"})

    def test_multi_provider_is_bounded_and_classifies_outcomes(self):
        descriptors = [provider_descriptor(str(i), "scholarly", ["scholarly_search"], "scholarly_metadata") for i in range(3)]
        entries = [{"descriptor": descriptors[0], "provider": Provider()},
                   {"descriptor": descriptors[1], "provider": Provider(error=OSError("network"))},
                   {"descriptor": descriptors[2], "provider": Provider([{"id": "w", "title": "Relevant"}])}]
        result = MultiProviderResearchSupervisor(entries, max_provider_attempts=2).scholarly_search(
            "raw_query", "raw query", 5, lambda records: {"usable": bool(records)})
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual([x["status"] for x in result["attempts"]], ["ZERO_RESULTS", "NETWORK_OR_PROVIDER_ERROR"])

    def test_unavailable_irrelevant_and_malformed_are_distinct(self):
        unavailable = provider_descriptor("u", "scholarly", [], "scholarly_metadata", availability="UNAVAILABLE")
        available = provider_descriptor("a", "scholarly", [], "scholarly_metadata")
        self.assertEqual(classify_provider_search_outcome(unavailable)["status"], "PROVIDER_UNAVAILABLE")
        self.assertEqual(classify_provider_search_outcome(available, {"records": [{}]}, relevance={"usable": False})["status"], "RESULTS_IRRELEVANT")
        self.assertEqual(classify_provider_search_outcome(available, [])["status"], "MALFORMED_RETRIEVAL")

    def test_web_fallback_sources_are_candidates_and_attempts_bounded(self):
        descriptor = provider_descriptor("s", "scholarly", ["scholarly_search"], "scholarly_metadata")
        result = MultiProviderResearchSupervisor([{"descriptor": descriptor, "provider": Provider()}],
            web_search=lambda query, limit: {"results": [{"url": "https://example.org"}]}).scholarly_search(
                "q", "q", 2, lambda records: {"usable": False})
        self.assertEqual(result["attempts"][-1]["status"], "WEB_CANDIDATE_SOURCES")
        self.assertLessEqual(result["attempt_count"], result["bounded_attempt_limit"])

    def test_safe_fetch_persists_raw_html_hash_and_bounded_text(self):
        body = b"<html><title>T</title><body>" + b"x" * 100 + b"</body></html>"
        with tempfile.TemporaryDirectory() as root:
            store = LocalArtifactStore(root)
            result = safe_fetch("https://example.org/source", store, "r", max_extracted_chars=20,
                                opener=lambda req, timeout: Response(body, "text/html"), resolver=public_resolver)
            self.assertEqual(result["sha256"], hashlib.sha256(body).hexdigest())
            self.assertTrue(result["normalized"]["truncated"])
            self.assertLessEqual(len(result["normalized"]["relevant_extracted_text"]), 20)
            self.assertTrue(Path(store.get_artifact_path("r", result["raw_artifact"]["path"])).exists())

    def test_json_api_preserves_raw_provenance(self):
        body = json.dumps({"value": 3}).encode()
        with tempfile.TemporaryDirectory() as root:
            result = safe_fetch("https://example.org/api", LocalArtifactStore(root), "r",
                                opener=lambda req, timeout: Response(body, "application/json"), resolver=public_resolver)
            self.assertEqual(result["normalized"]["json"], {"value": 3})
            self.assertEqual(result["request_provenance"]["requested_url"], "https://example.org/api")

    def test_private_urls_rejected(self):
        for url in ("http://localhost/x", "http://127.0.0.1/x", "file:///etc/passwd"):
            with self.assertRaises(ValueError): validate_public_http_url(url)

    def test_provider_modality_and_generated_skill_non_evidence(self):
        self.assertEqual(provider_descriptor("p", "scholarly", [], "scholarly_metadata")["evidence_modality"], "literature_metadata")
        allowed, assessment = may_continue_without_literature(["literature_metadata"], [{
            "status": "CANDIDATE_CREATED", "known_url": "https://example.org", "produces_modalities": ["literature_metadata"]}])
        self.assertFalse(allowed); self.assertNotIn("literature_metadata", assessment["currently_available_modalities"])


if __name__ == "__main__": unittest.main()
