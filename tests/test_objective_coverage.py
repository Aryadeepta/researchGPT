import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.objective_coverage import requirement_lifecycle, repair_research_completion_coverage, selected_objective_coverage
from src.paper_gate import paper_readiness_report
from src.research_state import create_run_state
from src.storage import LocalArtifactStore


class ObjectiveCoverageTests(unittest.TestCase):
    def _state(self):
        state = create_run_state("run", "generic topic")
        state["selected_question"] = "What observable relationship holds between condition A and outcome B?"
        state["replication_status"] = "PASSED"
        state["feasibility_routes"] = [{"approach": "primary_measurement", "selected": True, "empirical_evidence_path": "CONDITIONAL"}]
        state["capability_requirements"] = []
        return state

    def test_valid_replicated_metadata_claim_is_scoped_not_parent_completion(self):
        state = self._state()
        state["claim_evidence_ledger"]["claims"] = [{
            "claim_id": "C001", "status": "VERIFIED_TOOL_OUTPUT", "replication_status": "PASSED",
            "objective_relation": "PROCESS_OR_METADATA", "satisfies_requirement_ids": ["substudy:inventory"],
            "allowed_paper_language": "The inventory is reproducible; broader conclusions require additional evidence.",
        }]
        coverage = selected_objective_coverage(state, {"artifacts": []})
        self.assertEqual(coverage["status"], "INSUFFICIENT")
        self.assertEqual(coverage["claim_to_objective_mapping"][0]["relation"], "PROCESS_OR_METADATA")
        self.assertEqual(coverage["completed_substudies"][0]["status"], "COMPLETED_VALIDATED_REPLICATED")

    def test_explicit_compatible_direct_claim_can_cover_objective(self):
        state = self._state()
        state["empirical_evidence_path"] = "YES"
        state["feasibility_routes"] = [{"approach": "secondary_data_analysis", "selected": True, "empirical_evidence_path": "YES"}]
        state["claim_evidence_ledger"]["claims"] = [{
            "claim_id": "C1", "status": "VERIFIED_MEASUREMENT", "replication_status": "PASSED",
            "objective_relation": "DIRECT_ANSWER", "satisfies_requirement_ids": [],
            "evidence_modality": "secondary_dataset_analysis", "claim_class": "observational_association",
            "allowed_paper_language": "The validated measurement directly addresses the specified objective.",
        }]
        self.assertEqual(selected_objective_coverage(state, {"artifacts": []})["status"], "SUFFICIENT")

    def test_requirement_needs_execution_artifact_and_verified_postcondition(self):
        state = self._state()
        requirement = {"capability_id": "resolver", "purpose": "Acquire an observation", "expected_artifacts": ["resources/result.json"],
                       "validation_criteria": ["resource existence and provenance verified"],
                       "resource_requirements": {"feasibility_status": "MISSING"}}
        state["capability_requirements"] = [requirement]
        state["skill_registry"] = [{"skill": {"capability_id": "resolver", "skill_id": "resolver_v1", "implementation": {"script_path": "/tmp/resolver/run.sh"}}}]
        self.assertEqual(requirement_lifecycle(state, {"artifacts": []})[0]["status"], "RESOLVER_CREATED")
        state["execution_records"] = [{"command": "/tmp/resolver/run.sh", "exit_status": 0}]
        self.assertEqual(requirement_lifecycle(state, {"artifacts": []})[0]["status"], "RESOLVER_EXECUTED")
        self.assertFalse(requirement_lifecycle(state, {"artifacts": [{"path": "resources/result.json"}]})[0]["postconditions_verified"])
        state["requirement_validations"] = [{"requirement_id": "resolver", "status": "PASS", "verified_artifacts": ["resources/result.json"]}]
        self.assertEqual(requirement_lifecycle(state, {"artifacts": [{"path": "resources/result.json"}]})[0]["status"], "RESOLVED_VERIFIED")

    def test_package_integrity_can_coexist_with_insufficient_coverage(self):
        state = self._state()
        state["selected_objective_coverage"] = {"status": "INSUFFICIENT"}
        report = paper_readiness_report(state, {"claims": []}, {"artifacts": [{"path": "logs/raw.log"}]})
        self.assertFalse(report["ready"])
        self.assertIn("selected research objective coverage is insufficient", report["errors"])

    def test_repair_preserves_package_and_claim_while_reopening_capability_gap(self):
        with tempfile.TemporaryDirectory() as root:
            store = LocalArtifactStore(Path(root) / "runs")
            state = self._state()
            for node in state["dag"]["nodes"].values():
                node["status"] = "COMPLETED"
            state["status"] = "RESEARCH_COMPLETE"
            state["literature_cache"] = [{"identifier": str(index)} for index in range(15)]
            claim = {"claim_id": "C001", "status": "VERIFIED_TOOL_OUTPUT", "replication_status": "PASSED",
                     "allowed_paper_language": "The inventory is reproducible; broader conclusions require additional evidence."}
            state["claim_evidence_ledger"]["claims"] = [claim]
            package_path = store.run_root("run") / "packages" / "v1" / "research_package.json"
            package_path.parent.mkdir(parents=True)
            package_path.write_text(json.dumps({"schema_version": 1, "package_hash": "immutable"}) + "\n")
            before = hashlib.sha256(package_path.read_bytes()).hexdigest()
            state["research_packages"] = [{"version": 1, "package_hash": "immutable", "path": str(package_path)}]
            result = repair_research_completion_coverage(store, state)
            self.assertEqual(hashlib.sha256(package_path.read_bytes()).hexdigest(), before)
            self.assertEqual(state["claim_evidence_ledger"]["claims"][0], claim)
            self.assertEqual(len(state["literature_cache"]), 15)
            self.assertEqual(state["status"], "PARTIAL_RESEARCH")
            self.assertEqual(state["dag"]["nodes"]["capability_gap_analysis"]["status"], "PENDING")
            self.assertEqual(state["dag"]["nodes"]["question_refinement"]["status"], "COMPLETED")
            self.assertEqual(state["dag"]["nodes"]["feasibility_analysis"]["status"], "COMPLETED")
            self.assertEqual(result["earliest_reopened_node"], "capability_gap_analysis")


if __name__ == "__main__":
    unittest.main()
