import math
import statistics
from copy import deepcopy


COMPUTATIONAL_CLAIM_CLASSES = {
    "computational_performance", "bounded_correctness", "counterexample_existence", "deterministic_output_property",
}

MEASUREMENT_RECORD_FIELDS = (
    "trial_id", "configuration", "measurements", "correctness", "input_provenance", "environment",
)

CLAIM_CANDIDATE_FIELDS = (
    "claim_id", "claim", "claim_class", "evidence_modality", "objective_relation",
    "artifacts", "validator_artifacts", "claim_scope", "limitations", "allowed_paper_language",
)


def executable_evidence_skill(state):
    """Return the verified skill advertising executable evidence, independent of its name."""
    for entry in state.get("skill_registry", []):
        skill = entry.get("skill", entry)
        if (skill.get("capability_status") == "AVAILABLE_VERIFIED"
                and "executable_computation" in skill.get("produces_modalities", [])):
            return skill
    return None


def validate_executable_evidence_protocol(skill):
    """Validate the generic hand-off between an executable skill and the core."""
    errors = []
    protocol = skill.get("evidence_protocol")
    if not isinstance(protocol, dict):
        return ["verified executable skill lacks evidence_protocol"]
    for section in ("validation", "replication", "adversarial", "claims"):
        if not isinstance(protocol.get(section), dict):
            errors.append(f"evidence_protocol missing {section} section")
    validation = protocol.get("validation", {})
    if not validation.get("command_path") or not validation.get("report_artifact"):
        errors.append("validation protocol requires command_path and report_artifact")
    replication = protocol.get("replication", {})
    if not replication.get("comparison_report_field"):
        errors.append("replication protocol requires comparison_report_field")
    adversarial = protocol.get("adversarial", {})
    if not adversarial.get("report_artifact"):
        errors.append("adversarial protocol requires report_artifact")
    claims = protocol.get("claims", {})
    if not claims.get("candidates_artifact"):
        errors.append("claims protocol requires candidates_artifact")
    return errors


def validate_claim_candidates(candidates, available_artifacts, validation_passed, replication_passed):
    """Fail closed on malformed or unverified scientific payloads supplied by a skill."""
    errors = []
    seen = set()
    if not isinstance(candidates, list) or not candidates:
        return ["claim candidates must be a non-empty list"]
    available = set(available_artifacts)
    candidate_ids = {item.get("claim_id") for item in candidates if isinstance(item, dict)}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"claim candidate {index} is not an object")
            continue
        missing = [field for field in CLAIM_CANDIDATE_FIELDS if not candidate.get(field)]
        errors.extend(f"claim candidate {index} missing {field}" for field in missing)
        claim_id = candidate.get("claim_id")
        if claim_id in seen:
            errors.append(f"duplicate claim_id: {claim_id}")
        seen.add(claim_id)
        for field in ("artifacts", "validator_artifacts", "limitations"):
            if not isinstance(candidate.get(field), list):
                errors.append(f"claim candidate {index} {field} must be a list")
        referenced = set(candidate.get("artifacts", [])) | set(candidate.get("validator_artifacts", []))
        for path in sorted(referenced - available):
            errors.append(f"claim candidate {index} references unavailable artifact: {path}")
        if not isinstance(candidate.get("claim_scope"), dict) or not candidate.get("claim_scope", {}).get("scope_id"):
            errors.append(f"claim candidate {index} requires a scoped claim_scope")
        if candidate.get("universal_claim") is True:
            errors.append(f"claim candidate {index} may not assert universal scope from bounded executable evidence")
        if not validation_passed:
            errors.append(f"claim candidate {index} lacks passing independent validation")
        if candidate.get("objective_relation") in {"DIRECT_ANSWER", "DIRECT_CONSTRAINT"} and not replication_passed:
            errors.append(f"claim candidate {index} lacks passing replication")
        metadata = candidate.get("theorem_verifier_metadata")
        if metadata is not None and (not isinstance(metadata, dict) or not metadata.get("verifier")
                                     or not metadata.get("verification_artifact")):
            errors.append(f"claim candidate {index} has incomplete theorem/verifier metadata")
        elif metadata is not None:
            if metadata.get("integrity_status") != "PASS" or metadata.get("assumptions_disclosed") is not True:
                errors.append(f"claim candidate {index} theorem/verifier integrity did not pass")
            if metadata.get("verification_artifact") not in available:
                errors.append(f"claim candidate {index} theorem/verifier artifact is unavailable")
        if candidate.get("claim_class") == "computational_performance":
            prerequisites = candidate.get("correctness_prerequisite_claim_ids", [])
            if not prerequisites or not set(prerequisites).issubset(candidate_ids):
                errors.append(f"claim candidate {index} lacks declared correctness prerequisites")
    return errors


def adjudicate_executable_claim_candidates(payload, skill, available_artifacts,
                                            validation_passed, replication_passed):
    """Validate and stamp capability-authored science without interpreting its domain."""
    claims = deepcopy(payload.get("claim_candidates", []))
    errors = validate_executable_evidence_protocol(skill)
    errors.extend(validate_claim_candidates(claims, available_artifacts,
                                            validation_passed, replication_passed))
    if errors:
        return {"valid": False, "errors": errors, "claims": []}
    for claim in claims:
        claim["status"] = claim.get("proposed_status", "VERIFIED_TOOL_OUTPUT")
        claim["origin"] = "verified_capability_claim_contract"
        claim["producer"] = f"dynamic_skill:{skill['capability_id']}"
        claim["replication_status"] = "PASSED" if replication_passed else "FAILED"
    return {"valid": True, "errors": [], "claims": claims,
            "experiment_contracts": deepcopy(payload.get("experiment_contracts", []))}


def computational_claim_scope_compatible(contract, claim):
    if claim.get("experiment_contract_id") != contract.get("contract_id"):
        return False
    if claim.get("claim_class") != contract.get("claim_class"):
        return False
    if claim.get("universal_claim"):
        return False
    scope_id = contract.get("claim_scope", {}).get("scope_id")
    return bool(scope_id and claim.get("claim_scope", {}).get("scope_id") == scope_id
                and scope_id in claim.get("allowed_scope_ids", []))


def validate_experiment_contract(contract):
    required = (
        "contract_id", "research_objective", "claim_class", "independent_variable", "dependent_measurements",
        "input_regime", "correctness_criterion", "execution_policy", "falsification_condition",
        "expected_artifacts", "validation_procedure", "replication_procedure", "claim_scope",
    )
    errors = [f"missing contract field: {key}" for key in required if not contract.get(key)]
    if contract.get("claim_class") not in COMPUTATIONAL_CLAIM_CLASSES:
        errors.append("claim_class is not supported by executable computation")
    policy = contract.get("execution_policy", {})
    if not policy.get("deterministic") and not isinstance(policy.get("trials"), int):
        errors.append("execution policy requires deterministic=true or an integer trial count")
    effect = contract.get("practical_effect_criterion")
    if effect and (effect.get("metric") not in contract.get("dependent_measurements", []) or not isinstance(effect.get("threshold"), (int, float))):
        errors.append("practical effect criterion requires an explicit measured metric and numeric threshold")
    return errors


def validate_measurement_records(records, contract):
    errors = []
    required_measurements = set(contract.get("dependent_measurements", []))
    randomized = not contract.get("execution_policy", {}).get("deterministic", False)
    for index, record in enumerate(records or []):
        for key in ("trial_id", "configuration", "measurements", "correctness", "environment"):
            if key not in record:
                errors.append(f"record {index} missing {key}")
        if required_measurements - set(record.get("measurements", {})):
            errors.append(f"record {index} missing dependent measurements")
        provenance = record.get("input_provenance", {})
        if not provenance.get("generator_id") or not provenance.get("generator_version") or "parameters" not in provenance or "bounded_domain" not in provenance:
            errors.append(f"record {index} lacks reproducible input-generator provenance")
        if randomized and "seed" not in provenance:
            errors.append(f"record {index} lacks random seed")
    return errors


def descriptive_statistics(values, confidence=None):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("at least one measurement is required")
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * stddev / math.sqrt(len(values)) if confidence == 0.95 and len(values) > 1 else None
    return {
        "sample_count": len(values), "mean": mean, "median": statistics.median(values), "standard_deviation": stddev,
        "minimum": min(values), "maximum": max(values),
        "confidence_interval": {"confidence": confidence, "lower": mean - margin, "upper": mean + margin} if margin is not None else None,
    }


def analyze_measurements(records, contract):
    errors = validate_measurement_records(records, contract)
    if errors:
        return {"valid": False, "errors": errors, "claim_support": False}
    summaries = {}
    confidence = 0.95 if contract.get("statistical_analysis", {}).get("confidence_interval") == "normal_95" else None
    configurations = sorted({record["configuration"] for record in records})
    for configuration in configurations:
        summaries[configuration] = {
            metric: descriptive_statistics([record["measurements"][metric] for record in records if record["configuration"] == configuration], confidence)
            for metric in contract["dependent_measurements"]
        }
    correctness_passed = all(bool(record["correctness"].get("passed")) for record in records)
    paired = None
    paired_design = contract.get("paired_comparison")
    if paired_design:
        metric = paired_design["metric"]
        baseline_name = paired_design["baseline_configuration"]
        candidate_name = paired_design["candidate_configuration"]
        by_trial = {}
        for record in records:
            by_trial.setdefault(record["trial_id"], {})[record["configuration"]] = record["measurements"][metric]
        differences = [values[candidate_name] - values[baseline_name] for values in by_trial.values()
                       if baseline_name in values and candidate_name in values]
        paired = {"metric": metric, "difference_definition": "candidate_minus_baseline",
                  "difference_summary": descriptive_statistics(differences, confidence)}
    effect = None
    criterion = contract.get("practical_effect_criterion")
    if criterion:
        baseline = summaries[criterion["baseline_configuration"]][criterion["metric"]]["mean"]
        candidate = summaries[criterion["candidate_configuration"]][criterion["metric"]]["mean"]
        direction = criterion.get("direction", "lower_is_better")
        observed = ((baseline - candidate) / baseline) if direction == "lower_is_better" else ((candidate - baseline) / baseline)
        effect = {"metric": criterion["metric"], "observed_relative_effect": observed,
                  "threshold": criterion["threshold"], "criterion": "relative_improvement_gte",
                  "passed": observed >= criterion["threshold"] and correctness_passed}
    counterexamples = [{"trial_id": record["trial_id"], "configuration": record["configuration"],
                        "input_provenance": deepcopy(record["input_provenance"]),
                        "correctness": deepcopy(record["correctness"])}
                       for record in records if not record["correctness"].get("passed")]
    return {"valid": True, "errors": [], "summaries": summaries, "paired_comparison": paired, "correctness_passed": correctness_passed,
            "practical_effect": effect, "counterexamples": counterexamples,
            "claim_support": correctness_passed and (effect is None or effect["passed"])}


def validate_computational_execution(contract, execution_record, artifact_paths, measurements):
    contract_errors = validate_experiment_contract(contract)
    missing = sorted(set(contract.get("expected_artifacts", [])) - set(artifact_paths or []))
    analysis = analyze_measurements(measurements, contract) if not contract_errors and not missing else None
    errors = contract_errors + ([f"missing expected artifact: {path}" for path in missing])
    if execution_record.get("exit_status") != 0:
        errors.append("experiment process did not exit successfully")
    return {"execution_succeeded": execution_record.get("exit_status") == 0, "measurement_artifacts_present": not missing,
            "analysis": analysis, "errors": errors, "claim_support": not errors and bool(analysis and analysis["claim_support"])}


def bounded_falsification(records, contract):
    counterexamples = analyze_measurements(records, contract).get("counterexamples", [])
    return {"artifact_path": "experiments/counterexamples.json",
            "artifact": {"counterexamples": counterexamples, "search_scope": deepcopy(contract.get("input_regime")), "bounded": True},
            "counterexamples": counterexamples, "search_scope": deepcopy(contract.get("input_regime")), "bounded": True}


def replicate_computational_experiment(contract, original_measurements, execute_from_spec, tolerance=0.0):
    replicated = execute_from_spec(deepcopy(contract))
    original = analyze_measurements(original_measurements, contract)
    repeated = analyze_measurements(replicated, contract)
    matched = original.get("valid") and repeated.get("valid")
    if matched:
        for configuration, metrics in original["summaries"].items():
            for metric, summary in metrics.items():
                if abs(summary["mean"] - repeated["summaries"][configuration][metric]["mean"]) > tolerance:
                    matched = False
    return {"status": "PASS" if matched else "FAIL", "recomputed": True,
            "original_analysis": original, "replicated_analysis": repeated, "tolerance": tolerance,
            "replicated_measurements": replicated}
