from copy import deepcopy


AUTOMATION_CLOSURES = {"HIGH", "CONDITIONAL", "LOW", "UNKNOWN"}

MODALITY_CATALOG = {
    "formal_proof": {
        "evidence_kind": "deductive_artifact", "supported_claim_classes": ["deductive_theorem"],
        "required_capability_classes": ["formal_proof_producer"], "verification_mechanism": "trusted_verifier_acceptance",
        "external_inputs_mandatory": False,
    },
    "executable_computation": {
        "evidence_kind": "executed_artifact", "supported_claim_classes": ["computational_result", "computational_performance", "bounded_correctness", "counterexample_existence", "deterministic_output_property"],
        "required_capability_classes": ["code_execution"], "verification_mechanism": "reexecution_and_output_validation",
        "external_inputs_mandatory": False,
    },
    "simulation": {
        "evidence_kind": "simulated_artifact", "supported_claim_classes": ["model_behavior"],
        "required_capability_classes": ["simulation_execution"], "verification_mechanism": "reexecution_and_model_validation",
        "external_inputs_mandatory": False,
    },
    "secondary_dataset_analysis": {
        "evidence_kind": "dataset_derived_artifact", "supported_claim_classes": ["observational_association", "descriptive_measurement"],
        "required_capability_classes": ["dataset_analysis"], "verification_mechanism": "dataset_provenance_and_reexecution",
        "external_inputs_mandatory": True,
    },
    "literature_metadata": {
        "evidence_kind": "scholarly_metadata", "supported_claim_classes": ["bibliographic_inventory", "metadata_property"],
        "required_capability_classes": ["literature_metadata_analysis"], "verification_mechanism": "retrieval_provenance_and_reexecution",
        "external_inputs_mandatory": True,
    },
    "physical_measurement": {
        "evidence_kind": "measurement_artifact", "supported_claim_classes": ["physical_observation", "experimental_effect"],
        "required_capability_classes": ["measurement_acquisition"], "verification_mechanism": "instrument_provenance_and_independent_validation",
        "external_inputs_mandatory": True,
    },
}

ROUTE_MODALITIES = {
    "secondary_data_analysis": ["secondary_dataset_analysis"],
    "simulation": ["simulation"],
    "primary_measurement": ["physical_measurement"],
    "controlled_experiment": ["physical_measurement"],
    "systematic_evidence_analysis": ["literature_metadata"],
}

REQUIREMENT_TYPE_MODALITIES = {
    "measurement": ["physical_measurement"], "data": ["secondary_dataset_analysis"],
    "compute": ["executable_computation"], "software": ["executable_computation"],
}


def modality_descriptor(modality_id):
    definition = MODALITY_CATALOG.get(modality_id)
    if not definition:
        return {"modality_id": modality_id, "evidence_kind": "unknown", "supported_claim_classes": [],
                "required_capability_classes": [], "verification_mechanism": "unspecified",
                "external_inputs_mandatory": None}
    return {"modality_id": modality_id, **deepcopy(definition)}


def verified_capability_modalities(capabilities):
    available = set()
    for capability in capabilities or []:
        if capability.get("status") not in {"VERIFIED", "AVAILABLE_VERIFIED"}:
            continue
        available.update(capability.get("produces_modalities", []))
    return sorted(available)


def assess_automation_closure(required_modalities, capabilities, artifacts=None):
    required = list(dict.fromkeys(required_modalities or []))
    available = set()
    for capability in capabilities or []:
        if capability.get("status") not in {"VERIFIED", "AVAILABLE_VERIFIED"}:
            continue
        for modality in capability.get("produces_modalities", []):
            definition = MODALITY_CATALOG.get(modality, {})
            if not definition.get("external_inputs_mandatory") or capability.get("can_generate_external_evidence_autonomously"):
                available.add(modality)
    artifacts = artifacts or []
    artifact_modalities = {item.get("evidence_modality") for item in artifacts if item.get("verification_status") in {"VERIFIED", "AVAILABLE_VERIFIED"}}
    available.update(item for item in artifact_modalities if item)
    missing = [item for item in required if item not in available]
    unknown = [item for item in required if item not in MODALITY_CATALOG]
    if not required:
        closure = "UNKNOWN"
    elif not missing:
        closure = "HIGH"
    elif unknown:
        closure = "UNKNOWN"
    elif all(MODALITY_CATALOG[item]["external_inputs_mandatory"] for item in missing):
        closure = "CONDITIONAL" if all(item == "secondary_dataset_analysis" for item in missing) else "LOW"
    else:
        closure = "LOW"
    return {
        "required_evidence_modalities": required,
        "currently_available_modalities": sorted(available),
        "missing_modalities": missing, "automation_closure": closure,
        "assessment_origin": "deterministic_modality_capability_matching",
    }


def rank_candidate_questions(candidates, policy=None):
    policy = policy or {"objective": "autonomous_research", "allow_partial_automation": False}
    closure_rank = {"HIGH": 3, "CONDITIONAL": 2, "UNKNOWN": 1, "LOW": 0}
    ranked = sorted(deepcopy(candidates), key=lambda item: (
        closure_rank.get(item.get("automation_closure", "UNKNOWN"), 1)
        if policy.get("objective", "autonomous_research") == "autonomous_research" else 0,
        float(item.get("scientific_score", 0.0)),
    ), reverse=True)
    preferred = policy.get("preferred_question_id")
    if policy.get("allow_partial_automation") and preferred:
        ranked.sort(key=lambda item: item.get("question_id") == preferred, reverse=True)
    return ranked


def modalities_compatible(required_modalities, evidence_modalities, claim_class=None):
    required = set(required_modalities or [])
    evidence = set(evidence_modalities or [])
    if required and not required.issubset(evidence):
        return False
    if claim_class:
        return any(claim_class in MODALITY_CATALOG.get(item, {}).get("supported_claim_classes", []) for item in evidence)
    return not required or bool(evidence)


def objective_required_modalities(state):
    """Project mandatory evidence modalities from persisted objective semantics."""
    plan = state.get("research_modality_plan") or {}
    required = list(plan.get("required_evidence_modalities") or [])
    skeleton = state.get("computational_experimental_skeleton") or {}
    if (skeleton.get("experimental_skeleton_status") == "VALID"
            and skeleton.get("computational_testability") == "TESTABLE"
            and skeleton.get("required_modality")):
        required.append(skeleton["required_modality"])
    if not required:
        selected = next((item for item in state.get("feasibility_routes", []) if item.get("selected")), None)
        required.extend(ROUTE_MODALITIES.get((selected or {}).get("approach"), []))
    return list(dict.fromkeys(required))
