import unittest

from src.computational_skeleton import (canonical_semantic_id, clarification_effective,
                                        classify_measurement_source, control_availability_for_refinement,
                                        deterministic_bounded_question, experiment_relation_coherence,
                                        finalize_skeleton_status, measurement_direction_errors,
                                        measurement_from_kind, measurement_source_compatible,
                                        normalize_semantic_value, observable_specificity_errors,
                                        semantic_display_errors, semantic_field_role, semantic_value_record,
                                        validate_variable_contract,
                                        variable_measurement_coherence)


def skeleton(control_type="DIRECT_INPUT", measurement="execution time", neutrality="NEUTRAL_MEASUREMENT", source=None):
    variable = {"display_text": "input sequence length", "canonical_id": "input_sequence_length",
                "control_type": control_type}
    if source is not None: variable["source_variable"] = source
    return {"independent_variable": variable,
            "dependent_measurement": {"display_text": measurement,
                "canonical_id": canonical_semantic_id(measurement), "measurement_kind": "runtime",
                "measurement_source": "EXECUTION_METRIC", "neutrality": neutrality,
                "measurement_observable": {"display_text": measurement, "canonical_id": canonical_semantic_id(measurement), "specificity": "SELF_DESCRIBING"},
                "informativeness": "INFORMATIVE"},
            "bounded_regime": {"scope": "bounded"}, "required_modality": "executable_computation",
            "automation_closure": "HIGH", "computational_testability": "TESTABLE",
            "not_full_experiment_contract": True}


class ComputationalSkeletonTests(unittest.TestCase):
    def test_identifier_surface_is_normalized_before_semantic_validation(self):
        self.assertEqual(normalize_semantic_value("NUMBER_OF_INPUTS")["normalized_display_text"], "number of inputs")
        self.assertFalse(semantic_display_errors("NUMBER_OF_INPUTS", "variable"))
        self.assertTrue(validate_variable_contract({"variable": "x1", "control_type": "DIRECT_INPUT"}))
        self.assertEqual(canonical_semantic_id("Input sequence length"), "input_sequence_length")

    def test_safe_surface_normalization_and_raw_immutability(self):
        cases = {"input_quantity_limit": "input quantity limit", "input-size-limit": "input size limit",
                 "inputSequenceLength": "input sequence length", "INPUT_SEQUENCE_LENGTH": "input sequence length"}
        for raw, expected in cases.items():
            record = normalize_semantic_value(raw)
            self.assertEqual(record["raw_model_value"], raw)
            self.assertEqual(record["normalized_display_text"], expected)
        self.assertEqual(normalize_semantic_value("  `input_size`  ")["normalized_display_text"], "input size")

    def test_placeholders_are_not_rescued(self):
        for value in ("...", "???", "TBD", "TODO", "variable", "parameter", "value"):
            self.assertEqual(semantic_field_role(value), "NONSEMANTIC")
            self.assertNotEqual(semantic_value_record(value)["semantic_validation_status"], "VALID")

    def test_field_role_distinguishes_controls_from_measurements(self):
        control = semantic_value_record("input_quantity_limit")
        measured = semantic_value_record("processing_time")
        self.assertEqual(control["normalized_display_text"], "input quantity limit")
        self.assertEqual(control["semantic_role_classification"], "CONTROL_CANDIDATE")
        self.assertEqual(control["semantic_validation_status"], "VALID")
        self.assertEqual(measured["normalized_display_text"], "processing time")
        self.assertEqual(measured["semantic_validation_status"], "ROLE_MISMATCH")

    def test_direct_input_passes_without_source(self):
        self.assertEqual(finalize_skeleton_status(skeleton())["experimental_skeleton_status"], "VALID")

    def test_derived_requires_direct_source(self):
        self.assertEqual(finalize_skeleton_status(skeleton("DERIVED_FROM_INPUT"))["experimental_skeleton_status"], "INVALID")
        source = {"display_text": "input sequence length", "canonical_id": "input_sequence_length", "control_type": "DIRECT_INPUT"}
        self.assertEqual(finalize_skeleton_status(skeleton("DERIVED_FROM_INPUT", source=source))["experimental_skeleton_status"], "VALID")

    def test_uncontrolled_and_invalid_fields_override_testable(self):
        self.assertEqual(finalize_skeleton_status(skeleton("UNCONTROLLED"))["experimental_skeleton_status"], "INVALID")
        self.assertEqual(finalize_skeleton_status(skeleton(measurement="increase in execution time"))["experimental_skeleton_status"], "INVALID")
        self.assertEqual(finalize_skeleton_status(skeleton(neutrality="RESULT_EMBEDDED"))["experimental_skeleton_status"], "INVALID")
        same = skeleton(); same["dependent_measurement"].update({"display_text": "input sequence length", "canonical_id": "input_sequence_length"})
        self.assertEqual(finalize_skeleton_status(same)["experimental_skeleton_status"], "INVALID")

    def test_measurement_neutrality(self):
        self.assertTrue(measurement_direction_errors("increase in runtime"))
        self.assertEqual(measurement_direction_errors("runtime"), [])

    def test_standard_measurements_are_supervisor_derived_and_uncertain_fails(self):
        runtime = measurement_from_kind("runtime")
        self.assertEqual(runtime["display_text"], "execution time")
        self.assertEqual(runtime["canonical_id"], "execution_time")
        self.assertEqual(runtime["origin"], "deterministic_standard_measurement_mapping")
        self.assertIsNone(measurement_from_kind("uncertain"))
        self.assertIsNone(measurement_from_kind("output_value"))
        self.assertIsNone(measurement_from_kind("other"))
        self.assertEqual(measurement_from_kind("other", "cache misses")["canonical_id"], "cache_misses")
        output = measurement_from_kind("output_value", observable_display="solution cardinality")
        self.assertEqual(output["measurement_kind"], "output_value")
        self.assertEqual(output["measurement_observable"]["display_text"], "solution cardinality")
        normalized_output = measurement_from_kind("output_value", observable_display="solution_cardinality")
        self.assertEqual(normalized_output["display_text"], "solution cardinality")
        self.assertEqual(normalized_output["measurement_observable"]["raw_model_value"], "solution_cardinality")

    def test_parameterized_observable_rejects_category_echoes(self):
        self.assertTrue(observable_specificity_errors("output value", "output_value"))
        self.assertTrue(observable_specificity_errors("output_value", "output_value"))
        self.assertTrue(observable_specificity_errors("result", "output_value"))
        self.assertEqual(observable_specificity_errors("solution cardinality", "output_value"), [])

    def test_neutral_distinct_but_uninformative_is_invalid(self):
        candidate = skeleton()
        candidate["dependent_measurement"]["informativeness"] = "UNINFORMATIVE"
        self.assertEqual(finalize_skeleton_status(candidate)["experimental_skeleton_status"], "INVALID")
        candidate["computational_testability"] = "TESTABLE"
        self.assertEqual(finalize_skeleton_status(candidate)["experimental_skeleton_status"], "INVALID")

    def test_correctness_defers_oracle(self):
        correctness = measurement_from_kind("correctness")
        self.assertEqual(correctness["measurement_source"], "VALIDATION_DERIVED")
        self.assertEqual(correctness["downstream_requirements"], ["correctness_oracle"])
        self.assertNotIn("oracle", correctness)

    def test_measurement_kind_and_source_are_distinct_and_compatible(self):
        runtime = measurement_from_kind("runtime")
        self.assertEqual(runtime["measurement_kind"], "runtime")
        self.assertEqual(runtime["measurement_source"], "EXECUTION_METRIC")
        self.assertTrue(measurement_source_compatible("runtime", "EXECUTION_METRIC"))
        self.assertFalse(measurement_source_compatible("runtime", "ALGORITHM_OUTPUT"))
        self.assertTrue(measurement_source_compatible("output_value", "ALGORITHM_OUTPUT"))
        self.assertFalse(measurement_source_compatible("output_value", "INPUT_PROPERTY"))

    def test_input_or_topic_description_cannot_be_algorithm_output(self):
        self.assertEqual(classify_measurement_source("bounded integer inputs"), "INPUT_PROPERTY")
        measurement = measurement_from_kind("output_value", observable_display="bounded integer inputs")
        self.assertEqual(measurement["measurement_source"], "INPUT_PROPERTY")
        self.assertFalse(measurement_source_compatible("output_value", measurement["measurement_source"]))
        candidate = skeleton()
        candidate["dependent_measurement"] = measurement
        candidate["dependent_measurement"]["informativeness"] = "INFORMATIVE"
        self.assertEqual(finalize_skeleton_status(candidate)["experimental_skeleton_status"], "INVALID")

    def test_relation_requires_execution_lineage_not_only_distinct_ids(self):
        variable = skeleton()["independent_variable"]
        invalid = measurement_from_kind("output_value", observable_display="bounded input family")
        self.assertNotEqual(variable["canonical_id"], invalid["canonical_id"])
        self.assertEqual(experiment_relation_coherence(variable, invalid), "SOURCE_MISMATCH")
        self.assertEqual(experiment_relation_coherence(variable, measurement_from_kind("runtime")),
                         "EXPERIMENT_RELATION_COHERENT")

    def test_coherence_and_proposed_control_availability(self):
        variable = {"canonical_id": "input_size"}; measurement = measurement_from_kind("runtime")
        self.assertEqual(variable_measurement_coherence(variable, measurement), "COHERENT")
        self.assertEqual(variable_measurement_coherence(variable, {"canonical_id": "input_size"}), "INCOHERENT")
        self.assertEqual(control_availability_for_refinement("ALGORITHM_PARAMETER"), "PROPOSED_NOT_VERIFIED")
        self.assertEqual(control_availability_for_refinement("UNCONTROLLED"), "NOT_AVAILABLE")

    def test_valid_components_produce_neutral_bounded_question(self):
        question = deterministic_bounded_question("input size", "execution time")
        self.assertEqual(question, "How does varying input size affect execution time under a bounded computational input regime?")
        self.assertNotIn("increase", question)

    def test_clarification_requires_post_rewrite_validity(self):
        invalid = skeleton("UNCONTROLLED")
        result = clarification_effective(["variable uncontrolled"], invalid)
        self.assertFalse(result["effective"])
        self.assertEqual(result["post_rewrite_validation"]["status"], "INVALID")
        self.assertTrue(clarification_effective(["variable uncontrolled"], skeleton())["effective"])

    def test_skeleton_is_not_full_experiment_contract(self):
        result = finalize_skeleton_status(skeleton())
        self.assertTrue(result["not_full_experiment_contract"])
        self.assertNotIn("trials", result)
        self.assertNotIn("statistical_conclusions", result)


if __name__ == "__main__": unittest.main()
