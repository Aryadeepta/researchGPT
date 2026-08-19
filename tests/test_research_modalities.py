import unittest

from src.objective_coverage import selected_objective_coverage
from src.research_modalities import (assess_automation_closure, modalities_compatible,
                                     objective_required_modalities, rank_candidate_questions)
from src.research_state import create_run_state


class ResearchModalityTests(unittest.TestCase):
    def test_valid_computational_objective_cannot_fall_back_to_metadata_only(self):
        state = {"computational_experimental_skeleton": {
            "experimental_skeleton_status": "VALID", "computational_testability": "TESTABLE",
            "required_modality": "executable_computation"}}
        self.assertEqual(objective_required_modalities(state), ["executable_computation"])

    def test_high_closure_for_verified_executable_computation(self):
        result = assess_automation_closure(["executable_computation"], [{
            "capability_id": "runtime", "status": "AVAILABLE_VERIFIED", "produces_modalities": ["executable_computation"],
        }])
        self.assertEqual(result["automation_closure"], "HIGH")

    def test_conditional_closure_for_absent_secondary_dataset(self):
        result = assess_automation_closure(["secondary_dataset_analysis"], [{
            "capability_id": "analysis", "status": "AVAILABLE_VERIFIED", "produces_modalities": ["executable_computation"],
        }])
        self.assertEqual(result["automation_closure"], "CONDITIONAL")
        self.assertEqual(result["missing_modalities"], ["secondary_dataset_analysis"])

    def test_low_closure_for_unavailable_physical_measurement(self):
        self.assertEqual(assess_automation_closure(["physical_measurement"], [])["automation_closure"], "LOW")

    def test_modality_compatibility_is_structural(self):
        self.assertFalse(modalities_compatible(["physical_measurement"], ["literature_metadata"]))
        self.assertTrue(modalities_compatible(["formal_proof"], ["formal_proof"], "deductive_theorem"))

    def test_generated_skill_is_not_verified_modality_availability(self):
        generated = [{"status": "CANDIDATE_CREATED", "produces_modalities": ["formal_proof"]}]
        self.assertEqual(assess_automation_closure(["formal_proof"], generated)["automation_closure"], "LOW")
        generated[0]["status"] = "AVAILABLE_VERIFIED"
        self.assertEqual(assess_automation_closure(["formal_proof"], generated)["automation_closure"], "HIGH")

    def test_autonomous_ranking_prefers_closure_but_policy_can_retain_lower(self):
        candidates = [
            {"question_id": "interesting", "automation_closure": "LOW", "scientific_score": 1.0},
            {"question_id": "executable", "automation_closure": "HIGH", "scientific_score": 0.7},
        ]
        self.assertEqual(rank_candidate_questions(candidates)[0]["question_id"], "executable")
        policy = {"objective": "autonomous_research", "allow_partial_automation": True, "preferred_question_id": "interesting"}
        self.assertEqual(rank_candidate_questions(candidates, policy)[0]["question_id"], "interesting")

    def test_real3_style_negative_control_stays_partial(self):
        state = create_run_state("negative-control", "generic parent objective")
        state["selected_question"] = "What observable relationship exists between an intervention and a physical outcome?"
        state["feasibility_routes"] = [{"approach": "primary_measurement", "selected": True, "empirical_evidence_path": "CONDITIONAL"}]
        state["research_modality_plan"] = assess_automation_closure(["physical_measurement"], [{
            "capability_id": "metadata", "status": "AVAILABLE_VERIFIED", "produces_modalities": ["literature_metadata"],
        }])
        state["claim_evidence_ledger"]["claims"] = [{
            "claim_id": "C001", "status": "VERIFIED_TOOL_OUTPUT", "replication_status": "PASSED",
            "objective_relation": "PROCESS_OR_METADATA", "evidence_modality": "literature_metadata",
            "allowed_paper_language": "The metadata substudy is reproducible; broader conclusions require additional evidence.",
        }]
        coverage = selected_objective_coverage(state, {"artifacts": []})
        self.assertEqual(coverage["status"], "INSUFFICIENT")
        self.assertFalse(coverage["claim_to_objective_mapping"][0]["modality_compatible"])
        self.assertEqual(coverage["completed_substudies"][0]["status"], "COMPLETED_VALIDATED_REPLICATED")
        self.assertNotEqual(state.get("status"), "RESEARCH_COMPLETE")


if __name__ == "__main__":
    unittest.main()
