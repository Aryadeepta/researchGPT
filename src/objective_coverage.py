from copy import deepcopy
import hashlib
import json
import tempfile
from pathlib import Path

from src.research_state import now_iso
from src.research_modalities import ROUTE_MODALITIES, modalities_compatible
from src.computational_experiments import computational_claim_scope_compatible


VALID_CLAIM_STATUSES = {"VERIFIED_TOOL_OUTPUT", "VERIFIED_MEASUREMENT", "SUPPORTED_BY_CITATION"}
DIRECT_RELATIONS = {"DIRECT_ANSWER", "DIRECT_CONSTRAINT", "PARTIAL_ANSWER"}


def requirement_lifecycle(state, manifest):
    paths = {item.get("path") for item in manifest.get("artifacts", [])}
    executed_commands = " ".join(item.get("command", "") for item in state.get("execution_records", []))
    skills = {entry.get("skill", {}).get("capability_id"): entry for entry in state.get("skill_registry", [])}
    lifecycle = []
    for requirement in state.get("capability_requirements", []):
        capability_id = requirement.get("capability_id")
        expected = requirement.get("expected_artifacts") or requirement.get("required_outputs", [])
        skill = skills.get(capability_id)
        executed = bool(skill and skill.get("skill", {}).get("implementation", {}).get("script_path") in executed_commands)
        produced = bool(expected) and all(path in paths for path in expected)
        mandatory = (bool(requirement.get("mandatory")) or
                     requirement.get("resource_requirements", {}).get("feasibility_status") in {"MISSING", "UNKNOWN", "DISCOVERABLE"})
        postcondition = next((item for item in state.get("requirement_validations", [])
                              if item.get("requirement_id") in {requirement.get("requirement_id"), capability_id}), None)
        verified = bool(executed and produced and postcondition and postcondition.get("status") == "PASS"
                        and set(expected).issubset(set(postcondition.get("verified_artifacts", []))))
        if not mandatory and executed and produced and state.get("replication_status") == "PASSED":
            verified = verified or any(item.get("status") == "PASS" for item in state.get("validation_reports", []))
        if verified:
            status = "RESOLVED_VERIFIED"
        elif executed:
            status = "RESOLVER_EXECUTED"
        elif skill:
            status = "RESOLVER_CREATED"
        else:
            status = "MISSING"
        lifecycle.append({
            "requirement_id": capability_id, "purpose": requirement.get("purpose"),
            "mandatory": mandatory,
            "status": status, "resolver_skill_id": skill.get("skill", {}).get("skill_id") if skill else None,
            "resolver_executed": executed, "expected_artifacts": expected,
            "verified_artifacts": expected if verified else [],
            "postconditions_verified": verified,
            "provenance": {"requirements": "state.capability_requirements", "skills": "state.skill_registry",
                           "execution": "state.execution_records", "artifacts": "artifact_manifest"},
        })
    return lifecycle


def classify_claim_objective_relation(claim):
    explicit = claim.get("objective_relation")
    if explicit:
        return explicit
    allowed = (claim.get("allowed_paper_language") or "").lower()
    if "require additional evidence" in allowed or "requires additional evidence" in allowed:
        return "PROCESS_OR_METADATA"
    return "UNCLASSIFIED"


def selected_objective_coverage(state, manifest):
    lifecycle = requirement_lifecycle(state, manifest)
    route = next((item for item in state.get("feasibility_routes", []) if item.get("selected")), None)
    modality_plan = state.get("research_modality_plan", {})
    required_modalities = modality_plan.get("required_evidence_modalities") or ROUTE_MODALITIES.get(route.get("approach") if route else None, [])
    claims = []
    direct_valid = []
    for claim in state.get("claim_evidence_ledger", {}).get("claims", []):
        relation = classify_claim_objective_relation(claim)
        evidence_modalities = claim.get("evidence_modalities") or ([claim["evidence_modality"]] if claim.get("evidence_modality") else [])
        experiment_contract = next((item for item in state.get("computational_experiment_contracts", [])
                                    if item.get("contract_id") == claim.get("experiment_contract_id")), None)
        experiment_scope_compatible = computational_claim_scope_compatible(experiment_contract, claim) if experiment_contract else True
        satisfies = deepcopy(claim.get("satisfies_requirement_ids", []))
        if not satisfies:
            claim_artifacts = set(claim.get("artifacts", []))
            satisfies = [item.get("capability_id") for item in state.get("capability_requirements", [])
                         if claim_artifacts and claim_artifacts.issubset(set(item.get("expected_artifacts", [])))]
        mapping = {
            "claim_id": claim.get("claim_id"), "relation": relation,
            "claim_status": claim.get("status"), "replication_status": claim.get("replication_status"),
            "satisfies_requirement_ids": satisfies,
            "allowed_paper_language": claim.get("allowed_paper_language"),
            "evidence_modalities": evidence_modalities,
            "modality_compatible": modalities_compatible(required_modalities, evidence_modalities, claim.get("claim_class")),
            "experiment_scope_compatible": experiment_scope_compatible,
            "coverage_eligible": relation in DIRECT_RELATIONS and claim.get("status") in VALID_CLAIM_STATUSES
                                 and modalities_compatible(required_modalities, evidence_modalities, claim.get("claim_class"))
                                 and experiment_scope_compatible,
            "provenance": "explicit_claim_contract" if claim.get("objective_relation") else "fail_closed_legacy_scope_classification",
        }
        claims.append(mapping)
        if mapping["coverage_eligible"]:
            direct_valid.append(mapping)
    unresolved = [item for item in lifecycle if item["mandatory"] and item["status"] != "RESOLVED_VERIFIED"]
    required_modality = route.get("empirical_evidence_path") if route else state.get("empirical_evidence_path", "UNKNOWN")
    modality_satisfied = required_modality == "YES" or bool(direct_valid)
    sufficient = bool(state.get("selected_question")) and bool(direct_valid) and not unresolved and modality_satisfied
    reasons = []
    if not direct_valid:
        reasons.append("no validated claim directly answers or constrains the selected objective")
    if unresolved:
        reasons.append("mandatory feasibility requirements lack verified postconditions")
    if not modality_satisfied:
        reasons.append("required empirical evidence modality is not satisfied")
    return {
        "status": "SUFFICIENT" if sufficient else "INSUFFICIENT",
        "selected_question": state.get("selected_question"),
        "active_route": route.get("approach") if route else None,
        "required_empirical_evidence_path": required_modality,
        "required_evidence_modalities": required_modalities,
        "claim_to_objective_mapping": claims,
        "requirement_lifecycle": lifecycle,
        "unresolved_mandatory_requirements": unresolved,
        "completed_substudies": [{"substudy_id": "literature_metadata_inventory", "status": "COMPLETED_VALIDATED_REPLICATED",
                                  "claim_ids": [item["claim_id"] for item in claims if item["relation"] == "PROCESS_OR_METADATA"]}]
            if any(item["relation"] == "PROCESS_OR_METADATA" for item in claims) else [],
        "reasons": reasons, "computed_by": "deterministic_selected_objective_coverage_v1", "created_at": now_iso(),
    }


def objective_coverage_sufficient(state):
    return state.get("selected_objective_coverage", {}).get("status") == "SUFFICIENT"


def repair_research_completion_coverage(store, state):
    manifest = store.load_manifest(state["run_id"])
    coverage = selected_objective_coverage(state, manifest)
    package_snapshots = []
    for package in state.get("research_packages", []):
        path = Path(package.get("path", ""))
        package_snapshots.append({**deepcopy(package), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None})
    previous = {
        "run_status": state.get("status"), "node_statuses": {key: value.get("status") for key, value in state.get("dag", {}).get("nodes", {}).items()},
        "packages": package_snapshots, "claim_ledger": deepcopy(state.get("claim_evidence_ledger")),
        "validation_reports": deepcopy(state.get("validation_reports")), "replication_reports": deepcopy(state.get("replication_reports")),
        "created_at": now_iso(),
    }
    state.setdefault("completion_coverage_repair_history", []).append(previous)
    state["selected_objective_coverage"] = coverage
    state["requirement_lifecycle"] = coverage["requirement_lifecycle"]
    state["research_scope"] = {"parent_objective": state.get("selected_question"),
                               "completed_substudies": coverage["completed_substudies"],
                               "classification": "PARTIAL_SCOPED_RESEARCH"}
    if coverage["status"] != "SUFFICIENT":
        nodes = state.get("dag", {}).get("nodes", {})
        upstream = ("question_discovery", "evidence_discovery", "question_refinement", "feasibility_analysis")
        downstream = ("capability_gap_analysis", "skill_discovery_creation", "executable_artifact_dag", "research_execution",
                      "independent_validation", "adversarial_falsification", "replication", "claim_adjudication", "research_readiness")
        for node_id in upstream:
            if node_id in nodes and any(entry.get("node_statuses", {}).get(node_id) == "COMPLETED"
                                        for entry in state.get("completion_coverage_repair_history", [])):
                nodes[node_id]["status"] = "COMPLETED"
                nodes[node_id]["lease"] = None
                nodes[node_id]["failure_reason"] = None
        for node_id in downstream:
            if node_id in nodes:
                nodes[node_id]["status"] = "PENDING"
                nodes[node_id]["lease"] = None
                nodes[node_id]["failure_reason"] = None
                nodes[node_id]["attempts"] = 0
        state["status"] = "PARTIAL_RESEARCH"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = handle.name
    try:
        artifact_path = "coverage/selected_objective_coverage.json"
        if any(item.get("path") == artifact_path for item in manifest.get("artifacts", [])):
            artifact_path = f"coverage/selected_objective_coverage-{now_iso().replace(':', '').replace('.', '')}.json"
        artifact = store.put_artifact(state["run_id"], temp_path, artifact_path,
                                      "deterministic_completion_coverage_repair")
    finally:
        Path(temp_path).unlink(missing_ok=True)
    state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
    return {"status": state.get("status"), "coverage": coverage, "earliest_reopened_node": "capability_gap_analysis",
            "preserved_packages": package_snapshots, "artifact": artifact}
