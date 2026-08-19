import json
import os
import tempfile
import unittest
from pathlib import Path

from src.executor import ResearchExecutor, prose_only_satisfies_contract
from src.ledger import EvidenceLedger
from src.llm_gateway import BudgetExceeded, LLMRequest, ModelGateway, NullLLMProvider, LLMBudgetManager
from src.paper_gate import paper_readiness_report
from src.pipeline_meta import validate_pipeline_meta_schema
from src.research_state import acquire_node_lease, complete_node, create_run_state
from src.storage import LocalArtifactStore


class ArchitectureContractTests(unittest.TestCase):
    def test_generic_source_contains_no_run_specific_experiment_semantics(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
        forbidden = (
            "subset-sum", "bounded positive integers 1..30", "Reachable specification",
            "endpoint_median_ratio", "objective_executable_experiment", "local_research_real4",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, source)

    def test_prose_alone_cannot_satisfy_experiment_contract(self):
        contract = {"requires_execution": True, "outputs": ["results.md"]}
        self.assertTrue(prose_only_satisfies_contract(contract, contract["outputs"]))

    def test_hard_coded_results_cannot_become_verified_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = EvidenceLedger(d)
            with self.assertRaises(ValueError):
                ledger.add_claim({
                    "claim": "method is faster",
                    "status": "VERIFIED_MEASUREMENT",
                    "producer": "research_executor",
                    "artifacts": ["experiments/raw.csv"],
                    "hard_coded_results": True,
                })

    def test_planner_cannot_register_measured_results(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = EvidenceLedger(d)
            with self.assertRaises(ValueError):
                ledger.add_claim({
                    "claim": "benchmark improved",
                    "status": "VERIFIED_MEASUREMENT",
                    "origin": "planner",
                    "producer": "planner",
                    "artifacts": ["experiments/raw.csv"],
                })

    def test_same_producer_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = EvidenceLedger(d)
            with self.assertRaises(ValueError):
                ledger.add_claim({
                    "claim": "tool output is valid",
                    "status": "VERIFIED_TOOL_OUTPUT",
                    "producer": "executor",
                    "validated_by": ["executor"],
                    "artifacts": ["logs/tool.log"],
                })

    def test_absent_raw_artifacts_block_empirical_claims(self):
        state = create_run_state("r1", "topic")
        ledger = {"claims": [{"claim_id": "C001", "status": "VERIFIED_MEASUREMENT", "artifacts": [], "paper_role": "main"}]}
        report = paper_readiness_report(state, ledger, {"artifacts": []})
        self.assertFalse(report["ready"])

    def test_missing_literature_evidence_blocks_strong_novelty_claims(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = EvidenceLedger(d)
            with self.assertRaises(ValueError):
                ledger.add_claim({"claim": "This is novel", "status": "SUPPORTED_BY_CITATION", "citation_ids": []})

    def test_shallow_self_designed_pipeline_fails_meta_validation(self):
        pipeline = {
            "research_question": "Does X work?",
            "hypotheses": ["X works"],
            "artifact_producing_stages": [{"name": "write", "outputs": ["paper.md"], "producer": "planner", "validators": ["planner"]}],
        }
        result = validate_pipeline_meta_schema(pipeline)
        self.assertFalse(result["valid"])
        self.assertTrue(any("no executable" in e or "validates its own" in e for e in result["errors"]))

    def test_valid_artifact_producing_pipeline_passes_meta_validation(self):
        pipeline = {
            "research_question": "Does X improve Y?",
            "hypotheses": ["X improves Y on measured benchmark"],
            "falsification_criteria": ["X fails to beat baseline on raw metric"],
            "artifact_producing_stages": [{"name": "experiment", "requires_execution": True, "produces_raw_artifacts": True, "outputs": ["experiments/raw.csv"], "producer": "executor", "validators": ["provenance_auditor"]}],
            "tool_requirements": ["python"],
            "implementation_requirements": ["reference implementation"],
            "experiment_requirements": ["baseline comparison"],
            "baselines": ["baseline"],
            "controls": ["same data split"],
            "validation_stages": [{"name": "audit", "producer": "auditor", "validates_producer": "executor"}],
            "independent_validators": ["provenance_auditor"],
            "adversarial_stages": ["methodology_review"],
            "replication": {"required": True},
            "completion_contracts": ["raw metrics exist"],
            "paper_gate": {"required": True},
            "failure_states": ["BLOCKED_MISSING_EVIDENCE"],
            "evidence_provenance": {"required": True},
        }
        self.assertTrue(validate_pipeline_meta_schema(pipeline)["valid"])

    def test_replication_failure_blocks_paper_readiness(self):
        state = create_run_state("r1", "topic")
        state["replication_status"] = "FAILED"
        report = paper_readiness_report(state, {"claims": []}, {"artifacts": [{"path": "experiments/raw.csv"}]})
        self.assertFalse(report["ready"])

    def test_unresolved_fatal_adversarial_finding_blocks_paper_readiness(self):
        state = create_run_state("r1", "topic")
        state["fatal_adversarial_findings"] = [{"severity": "fatal"}]
        report = paper_readiness_report(state, {"claims": []}, {"artifacts": [{"path": "experiments/raw.csv"}]})
        self.assertFalse(report["ready"])

    def test_paper_writers_cannot_upgrade_claims(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = EvidenceLedger(d)
            claim = ledger.add_claim({"claim": "possible improvement", "status": "HYPOTHESIS"})
            with self.assertRaises(ValueError):
                ledger.update_claim(claim["claim_id"], status="VERIFIED_TOOL_OUTPUT", updated_by="paper_writer", artifacts=["logs/tool.log"])

    def test_budget_exhaustion_yields_blocked_budget(self):
        state = create_run_state("r1", "topic")
        gateway = ModelGateway(NullLLMProvider(), LLMBudgetManager(max_run_usd=0.01, max_agent_iterations=10))
        with self.assertRaises(BudgetExceeded):
            gateway.generate(state, LLMRequest(prompt="x", stage="test"), estimated_cost=0.02)

    def test_failed_local_estimate_does_not_count_as_actual_spend(self):
        class FailingProvider:
            available = True
            def generate_structured(self, request, schema=None):
                raise RuntimeError("NO_ELIGIBLE_LOCAL_MODEL task_class=candidate_question_generation")

        state = create_run_state("r1", "topic")
        gateway = ModelGateway(FailingProvider(), LLMBudgetManager(max_agent_iterations=10))
        with self.assertRaises(RuntimeError):
            gateway.generate_structured(
                state,
                LLMRequest(prompt="{}", stage="question_discovery", task_class="candidate_question_generation"),
                required_keys=["candidate_questions"],
                estimated_cost=0.002,
            )
        self.assertEqual(state["budget"]["llm_usd"], 0.0)
        self.assertEqual(state["budget"]["estimated_llm_usd"], 0.002)
        self.assertEqual(state["budget"]["calls"][0]["actual_cost"], 0.0)

    def test_ephemeral_workers_can_resume_persisted_runs(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(d)
            state = create_run_state("r1", "topic", dag=[
                {"node_id": "a", "kind": "planning", "dependencies": [], "contract": {}},
                {"node_id": "b", "kind": "planning", "dependencies": ["a"], "contract": {}},
            ])
            leased = acquire_node_lease(state, worker_id="w1")
            complete_node(state, leased["node_id"])
            store.atomic_update_state("r1", state)
            resumed = store.load_state("r1")
            next_node = acquire_node_lease(resumed, worker_id="w2")
            self.assertEqual(next_node["node_id"], "b")

    def test_local_storage_works_without_cloud_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(d)
            path = Path(d) / "artifact.txt"
            path.write_text("raw", encoding="utf-8")
            entry = store.put_artifact("r1", path, "experiments/raw.txt", "executor")
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertEqual(store.verify_manifest("r1"), [])

    def test_optional_cloud_storage_configuration_does_not_affect_local_tests(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["RESEARCH_S3_BUCKET"] = "not-used"
            store = LocalArtifactStore(d)
            store.atomic_update_state("r1", {"ok": True})
            self.assertEqual(store.load_state("r1"), {"ok": True})

    def test_oracle_is_no_longer_required_for_normal_execution(self):
        workflow = Path(".github/workflows/research-start.yml").read_text(encoding="utf-8")
        self.assertIn("ubuntu-latest", workflow)
        self.assertNotIn("oracle", workflow.lower())
        self.assertNotIn("cloud", workflow.lower())


if __name__ == "__main__":
    unittest.main()
