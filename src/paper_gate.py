MAIN_CLAIM_STATUSES = {
    "VERIFIED_TOOL_OUTPUT",
    "VERIFIED_MEASUREMENT",
    "SUPPORTED_BY_CITATION",
}


INCOMPLETE_STATUSES = {
    "CONCEPT_ONLY",
    "PLANNED_RESEARCH",
    "PARTIAL_RESEARCH",
    "BLOCKED_MISSING_EVIDENCE",
    "BLOCKED_INVALID_METHOD",
    "BLOCKED_ADVERSARIAL_FAILURE",
    "BLOCKED_REPLICATION_FAILURE",
}


def paper_readiness_report(state, ledger, manifest):
    errors = []
    coverage = state.get("selected_objective_coverage", {})
    if coverage.get("status") != "SUFFICIENT":
        errors.append("selected research objective coverage is insufficient")
    artifacts = manifest.get("artifacts", [])
    artifact_paths = {a.get("path") for a in artifacts}
    raw_artifacts = [p for p in artifact_paths if p and (p.startswith("experiments/") or p.startswith("logs/") or "raw" in p)]

    for node in state.get("dag", {}).get("nodes", {}).values():
        contract = node.get("contract", {})
        if contract.get("requires_execution") and node.get("status") != "COMPLETED":
            errors.append(f"execution node incomplete: {node.get('node_id')}")
        for required in contract.get("raw_outputs", []):
            if required not in artifact_paths:
                errors.append(f"missing raw artifact: {required}")

    if not raw_artifacts:
        errors.append("no raw execution artifacts")

    for claim in ledger.get("claims", []):
        if claim.get("paper_role", "main") == "main" and claim.get("status") not in MAIN_CLAIM_STATUSES:
            errors.append(f"main claim lacks verified evidence: {claim.get('claim_id')}")
        if claim.get("status") == "VERIFIED_MEASUREMENT" and not claim.get("artifacts"):
            errors.append(f"measurement claim lacks artifacts: {claim.get('claim_id')}")
        if claim.get("status") == "SUPPORTED_BY_CITATION" and not claim.get("citation_ids"):
            errors.append(f"citation claim lacks citation ids: {claim.get('claim_id')}")
        if claim.get("replication_status") == "FAILED":
            errors.append(f"replication failed for claim: {claim.get('claim_id')}")
        if claim.get("fatal_adversarial_findings"):
            errors.append(f"fatal adversarial finding unresolved: {claim.get('claim_id')}")
        producer = claim.get("producer")
        for validator in claim.get("validated_by", []):
            if validator == producer:
                errors.append(f"same-producer validation rejected: {claim.get('claim_id')}")

    if state.get("replication_status") == "FAILED":
        errors.append("replication failed")
    if state.get("fatal_adversarial_findings"):
        errors.append("unresolved fatal adversarial findings")

    return {"ready": not errors, "errors": errors, "status": "PAPER_READY" if not errors else "BLOCKED_MISSING_EVIDENCE"}


def assert_paper_ready(state, ledger, manifest):
    report = paper_readiness_report(state, ledger, manifest)
    if not report["ready"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report
