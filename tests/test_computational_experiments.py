import unittest

from src.computational_experiments import (
    adjudicate_executable_claim_candidates, analyze_measurements, bounded_falsification, replicate_computational_experiment,
    executable_evidence_skill, validate_claim_candidates, validate_computational_execution,
    validate_executable_evidence_protocol, validate_measurement_records,
)
from src.objective_coverage import selected_objective_coverage
from src.research_state import create_run_state


def contract():
    return {
        "contract_id": "experiment-1", "research_objective": "Compare two configurations on a bounded benchmark.",
        "claim_class": "computational_performance", "independent_variable": "configuration",
        "dependent_measurements": ["duration"],
        "input_regime": {"generator_id": "generator", "generator_version": "1", "parameters": {"size": 4}, "bounded_domain": [0, 3]},
        "correctness_criterion": {"required": True, "oracle": "reference implementation"},
        "execution_policy": {"deterministic": False, "trials": 2},
        "falsification_condition": "Any failed correctness result or effect below the declared threshold.",
        "expected_artifacts": ["experiments/measurements.json"],
        "validation_procedure": {"validator": "deterministic_measurement_validator"},
        "replication_procedure": {"mode": "reexecute_from_specification", "tolerance": 0.0},
        "claim_scope": {"scope_id": "bounded-benchmark-v1", "scope_type": "BENCHMARK_REGIME"},
        "paired_comparison": {"metric": "duration", "baseline_configuration": "baseline", "candidate_configuration": "candidate"},
        "practical_effect_criterion": {"metric": "duration", "baseline_configuration": "baseline",
                                         "candidate_configuration": "candidate", "direction": "lower_is_better", "threshold": 0.1},
    }


def measurements(candidate_correct=True):
    rows = []
    for trial, seed in (("one", 11), ("two", 12)):
        for configuration, duration in (("baseline", 10.0), ("candidate", 8.0)):
            rows.append({
                "trial_id": trial, "configuration": configuration, "measurements": {"duration": duration},
                "correctness": {"passed": candidate_correct if configuration == "candidate" else True},
                "input_provenance": {"generator_id": "generator", "generator_version": "1", "parameters": {"size": 4},
                                     "seed": seed, "bounded_domain": [0, 3]},
                "environment": {"runtime": "python", "version": "test"},
            })
    return rows


class ComputationalExperimentTests(unittest.TestCase):
    def test_non_subset_sum_capability_uses_generic_evidence_contract(self):
        skill = {
            "capability_id": "cellular_automaton_density", "capability_status": "AVAILABLE_VERIFIED",
            "produces_modalities": ["executable_computation"],
            "implementation": {"validator_path": "/tmp/validator"},
            "evidence_protocol": {
                "validation": {"command_path": "validator_path", "report_artifact": "validation/report.json"},
                "replication": {"comparison_report_field": "scientific_signature"},
                "adversarial": {"report_artifact": "adversarial/falsification.json"},
                "claims": {"candidates_artifact": "claims/candidates.json"},
            },
        }
        self.assertIs(executable_evidence_skill({"skill_registry": [{"skill": skill}]}), skill)
        self.assertEqual(validate_executable_evidence_protocol(skill), [])
        artifacts = {"results/density.json", "validation/report.json", "replication/report.json"}
        candidates = [{
            "claim_id": "C-density", "claim": "Density stabilized within the declared finite grid regime.",
            "claim_class": "deterministic_output_property", "evidence_modality": "executable_computation",
            "objective_relation": "DIRECT_ANSWER", "artifacts": ["results/density.json"],
            "validator_artifacts": ["validation/report.json", "replication/report.json"],
            "claim_scope": {"scope_id": "finite-grid-v1", "scope_type": "FINITE_GRID"},
            "limitations": ["Only the declared finite grids were executed."],
            "allowed_paper_language": "The observed finite-grid runs stabilized under the declared rule.",
            "universal_claim": False,
        }]
        self.assertEqual(validate_claim_candidates(candidates, artifacts, True, True), [])
        result = adjudicate_executable_claim_candidates(
            {"claim_candidates": candidates}, skill, artifacts, True, True)
        self.assertTrue(result["valid"])
        self.assertEqual(result["claims"][0]["producer"], "dynamic_skill:cellular_automaton_density")

    def test_performance_claim_cannot_precede_correctness_or_replication(self):
        candidate = {
            "claim_id": "C-speed", "claim": "The candidate was faster in the declared regime.",
            "claim_class": "computational_performance", "evidence_modality": "executable_computation",
            "objective_relation": "DIRECT_ANSWER", "artifacts": ["results.json"],
            "validator_artifacts": ["validation.json"], "claim_scope": {"scope_id": "scope"},
            "limitations": ["bounded"], "allowed_paper_language": "A bounded speed difference was observed.",
        }
        errors = validate_claim_candidates([candidate], {"results.json", "validation.json"}, True, False)
        self.assertTrue(any("replication" in error for error in errors))
        self.assertTrue(any("correctness prerequisites" in error for error in errors))

    def test_formal_claim_requires_integrity_metadata(self):
        candidate = {
            "claim_id": "C-proof", "claim": "The checker accepted the declared invariant.",
            "claim_class": "deductive_theorem", "evidence_modality": "formal_proof",
            "objective_relation": "SUPPORTING_EVIDENCE", "artifacts": ["proof.json"],
            "validator_artifacts": ["proof.json"], "claim_scope": {"scope_id": "formal-spec-v1"},
            "limitations": ["Applies only to the formal specification."],
            "allowed_paper_language": "The verifier accepted the declared formal statement.",
            "theorem_verifier_metadata": {"verifier": "synthetic", "verification_artifact": "proof.json",
                                          "integrity_status": "FAIL", "assumptions_disclosed": True},
        }
        errors = validate_claim_candidates([candidate], {"proof.json"}, True, True)
        self.assertTrue(any("integrity did not pass" in error for error in errors))

    def test_exit_zero_and_missing_artifact_cannot_support_claim(self):
        result = validate_computational_execution(contract(), {"exit_status": 0}, [], measurements())
        self.assertTrue(result["execution_succeeded"])
        self.assertFalse(result["measurement_artifacts_present"])
        self.assertFalse(result["claim_support"])

    def test_deterministic_statistics_and_effect_are_computed(self):
        analysis = analyze_measurements(measurements(), contract())
        summary = analysis["summaries"]["candidate"]["duration"]
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["mean"], 8.0)
        self.assertEqual(summary["median"], 8.0)
        self.assertEqual(summary["standard_deviation"], 0.0)
        self.assertEqual((summary["minimum"], summary["maximum"]), (8.0, 8.0))
        self.assertTrue(analysis["practical_effect"]["passed"])
        self.assertAlmostEqual(analysis["practical_effect"]["observed_relative_effect"], 0.2)
        self.assertEqual(analysis["paired_comparison"]["difference_summary"]["mean"], -2.0)

    def test_correctness_failure_invalidates_performance_and_becomes_counterexample(self):
        records = measurements(candidate_correct=False)
        analysis = analyze_measurements(records, contract())
        self.assertFalse(analysis["claim_support"])
        self.assertFalse(analysis["practical_effect"]["passed"])
        falsification = bounded_falsification(records, contract())
        counterexamples = falsification["counterexamples"]
        self.assertEqual(len(counterexamples), 2)
        self.assertIn("input_provenance", counterexamples[0])
        self.assertEqual(falsification["artifact_path"], "experiments/counterexamples.json")

    def test_generator_and_seed_provenance_are_mandatory(self):
        records = measurements()
        del records[0]["input_provenance"]["seed"]
        self.assertIn("record 0 lacks random seed", validate_measurement_records(records, contract()))

    def test_replication_reexecutes_from_persisted_specification(self):
        calls = []
        def execute(specification):
            calls.append(specification["contract_id"])
            return measurements()
        result = replicate_computational_experiment(contract(), measurements(), execute)
        self.assertEqual(calls, ["experiment-1"])
        self.assertTrue(result["recomputed"])
        self.assertEqual(result["status"], "PASS")

    def test_computational_claim_scope_controls_objective_coverage(self):
        state = create_run_state("run", "generic computational objective")
        state["selected_question"] = "How do two configurations compare over the declared bounded input regime?"
        state["research_modality_plan"] = {"required_evidence_modalities": ["executable_computation"]}
        state["feasibility_routes"] = [{"approach": "other", "selected": True, "empirical_evidence_path": "YES"}]
        state["computational_experiment_contracts"] = [contract()]
        claim = {"claim_id": "C1", "status": "VERIFIED_TOOL_OUTPUT", "replication_status": "PASSED",
                 "objective_relation": "DIRECT_ANSWER", "evidence_modality": "executable_computation",
                 "claim_class": "computational_performance", "experiment_contract_id": "experiment-1",
                 "claim_scope": {"scope_id": "bounded-benchmark-v1"}, "allowed_scope_ids": ["bounded-benchmark-v1"],
                 "universal_claim": False,
                 "allowed_paper_language": "Within bounded-benchmark-v1, the recorded candidate met the declared effect criterion."}
        state["claim_evidence_ledger"]["claims"] = [claim]
        self.assertEqual(selected_objective_coverage(state, {"artifacts": []})["status"], "SUFFICIENT")
        claim["allowed_scope_ids"] = []
        self.assertEqual(selected_objective_coverage(state, {"artifacts": []})["status"], "INSUFFICIENT")
        claim["allowed_scope_ids"] = ["bounded-benchmark-v1"]
        claim["universal_claim"] = True
        coverage = selected_objective_coverage(state, {"artifacts": []})
        self.assertEqual(coverage["status"], "INSUFFICIENT")
        self.assertFalse(coverage["claim_to_objective_mapping"][0]["experiment_scope_compatible"])

    def test_executable_computation_does_not_satisfy_formal_proof(self):
        state = create_run_state("run", "generic deductive objective")
        state["selected_question"] = "Does the specified proposition follow under the formal assumptions?"
        state["research_modality_plan"] = {"required_evidence_modalities": ["formal_proof"]}
        state["feasibility_routes"] = [{"approach": "other", "selected": True, "empirical_evidence_path": "YES"}]
        state["claim_evidence_ledger"]["claims"] = [{"claim_id": "C", "status": "VERIFIED_TOOL_OUTPUT",
            "objective_relation": "DIRECT_ANSWER", "evidence_modality": "executable_computation",
            "claim_class": "bounded_correctness", "replication_status": "PASSED"}]
        coverage = selected_objective_coverage(state, {"artifacts": []})
        self.assertEqual(coverage["status"], "INSUFFICIENT")
        self.assertFalse(coverage["claim_to_objective_mapping"][0]["modality_compatible"])


if __name__ == "__main__":
    unittest.main()
