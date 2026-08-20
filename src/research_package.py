import hashlib
import json
from pathlib import Path

from src.paper_gate import paper_readiness_report
from src.objective_coverage import objective_coverage_sufficient
from src.research_state import now_iso


FORMAL_TRUST_LEVELS = {
    "KERNEL_ONLY", "NATIVE_DECIDE", "EXTERNAL_CERTIFICATE_CHECKED", "COMPUTATIONAL_ONLY",
}
FORMAL_EXACT_CLAIM_ALLOWED_TRUST = {"KERNEL_ONLY", "NATIVE_DECIDE", "EXTERNAL_CERTIFICATE_CHECKED"}


def stable_hash(data):
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_research_package(state, manifest, readiness=None):
    readiness = readiness or paper_readiness_report(state, state.get("claim_evidence_ledger", {"claims": []}), manifest)
    if not readiness["ready"]:
        return None, {"status": "BLOCKED_RESEARCH_NOT_READY", "reasons": readiness["errors"]}
    version = len(state.get("research_packages", [])) + 1
    package = {
        "schema_version": 1,
        "package_id": f"RP-{state['run_id']}",
        "run_id": state["run_id"],
        "version": version,
        "created_at": now_iso(),
        "research_question": state.get("research_spec", {}).get("research_question") or state.get("topic"),
        "research_specification": state.get("research_spec", {}),
        "hypotheses": state.get("research_spec", {}).get("hypotheses", []),
        "research_dag": state.get("dag", {}),
        "claim_evidence_ledger": state.get("claim_evidence_ledger", {"claims": []}),
        "verified_literature": state.get("literature_cache", []),
        "artifact_manifest": manifest,
        "raw_artifacts": [a for a in manifest.get("artifacts", []) if "/raw/" in a.get("path", "") or a.get("path", "").startswith(("execution/", "evidence/"))],
        "derived_artifacts": [a for a in manifest.get("artifacts", []) if a.get("path", "").startswith(("analysis/", "figures/", "tables/"))],
        "analysis_outputs": [a for a in manifest.get("artifacts", []) if a.get("path", "").startswith("analysis/")],
        "figures": [a for a in manifest.get("artifacts", []) if a.get("path", "").startswith("figures/")],
        "tables": [a for a in manifest.get("artifacts", []) if a.get("path", "").startswith("tables/")],
        "experimental_provenance": state.get("execution_records", []),
        "validation_reports": state.get("validation_reports", []),
        "adversarial_findings": state.get("adversarial_findings", []),
        "replication_reports": state.get("replication_reports", []),
        "known_limitations": state.get("known_limitations", []),
        "unresolved_findings": state.get("unresolved_findings", []),
        "research_readiness_report": readiness,
    }
    package["package_hash"] = stable_hash(package)
    return package, {"status": "PACKAGED"}


def write_research_package(store, state):
    manifest = store.load_manifest(state["run_id"])
    package, report = build_research_package(state, manifest)
    if not package:
        return None, report
    run_root = Path(store.run_root(state["run_id"]))
    package_dir = run_root / "packages" / f"v{package['version']}"
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "research_package.json"
    package_path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.setdefault("research_packages", []).append({
        "package_id": package["package_id"],
        "version": package["version"],
        "package_hash": package["package_hash"],
        "path": str(package_path),
        "created_at": package["created_at"],
    })
    state["status"] = "RESEARCH_COMPLETE" if objective_coverage_sufficient(state) else "PARTIAL_RESEARCH"
    return package, report


def load_research_package(store, run_id, version=None):
    run_root = Path(store.run_root(run_id))
    packages_root = run_root / "packages"
    if version is None:
        versions = sorted(packages_root.glob("v*/research_package.json"))
        if not versions:
            return None
        path = versions[-1]
    else:
        path = packages_root / f"v{version}" / "research_package.json"
    if not path.exists():
        return None
    package = json.loads(path.read_text(encoding="utf-8"))
    expected = package.get("package_hash")
    actual_payload = dict(package)
    actual_payload.pop("package_hash", None)
    actual = stable_hash(actual_payload)
    return package if expected == actual else None


def verify_research_package(store, run_id, version=None):
    package = load_research_package(store, run_id, version)
    if not package:
        return {"status": "FAIL", "reasons": ["missing or corrupt ResearchPackage"]}
    manifest_failures = store.verify_manifest(run_id)
    if manifest_failures:
        return {"status": "FAIL", "reasons": manifest_failures}
    formal_failures = _verify_formal_claim_evidence(store, run_id, package)
    if formal_failures:
        return {"status": "FAIL", "reasons": formal_failures}
    if not package.get("research_readiness_report", {}).get("ready"):
        return {"status": "WAITING", "reasons": package.get("research_readiness_report", {}).get("errors", [])}
    return {"status": "PASS", "package": {"package_id": package["package_id"], "version": package["version"], "hash": package["package_hash"]}}


def _verify_formal_claim_evidence(store, run_id, package):
    """Enforce that formal exact claims name durable, self-verifying evidence.

    Artifact paths are deliberately repository-relative manifest keys.  In
    particular, a scratch path can never be the evidence reference.
    """
    failures = []
    manifest = {entry.get("path"): entry for entry in package.get("artifact_manifest", {}).get("artifacts", [])}
    for claim in package.get("claim_evidence_ledger", {}).get("claims", []):
        modalities = set(claim.get("evidence_modalities", []))
        if "formal_proof" not in modalities:
            continue
        claim_id = claim.get("claim_id", "<unknown>")
        evidence = claim.get("formal_evidence")
        if not isinstance(evidence, dict):
            failures.append(f"{claim_id}: missing formal evidence metadata")
            continue
        source = evidence.get("artifact_path")
        expected_sha = evidence.get("artifact_sha256")
        if not source or not expected_sha:
            failures.append(f"{claim_id}: formal artifact path/hash missing")
            continue
        if source.startswith("/") or "/tmp/" in source or source not in manifest:
            failures.append(f"{claim_id}: formal artifact is not a durable manifest artifact")
            continue
        artifact = Path(store.get_artifact_path(run_id, source))
        if not artifact.is_file():
            failures.append(f"{claim_id}: formal artifact missing")
            continue
        actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_sha != expected_sha or manifest[source].get("sha256") != expected_sha:
            failures.append(f"{claim_id}: formal artifact checksum mismatch")
        metadata_path = evidence.get("verifier_metadata_artifact")
        if not metadata_path or metadata_path.startswith("/") or "/tmp/" in metadata_path or metadata_path not in manifest:
            failures.append(f"{claim_id}: verifier metadata is not a durable manifest artifact")
            continue
        try:
            metadata = json.loads(Path(store.get_artifact_path(run_id, metadata_path)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"{claim_id}: verifier metadata missing or invalid")
            continue
        trust = metadata.get("verifier_trust")
        if trust not in FORMAL_TRUST_LEVELS:
            failures.append(f"{claim_id}: verifier trust classification missing or invalid")
        if metadata.get("verifier") != "Lean" or metadata.get("exit_code") != 0:
            failures.append(f"{claim_id}: formal verifier did not pass")
        if metadata.get("input_sha256") != expected_sha:
            failures.append(f"{claim_id}: verifier input does not match durable artifact")
        required = ("command", "executable", "executable_version", "stdout_artifact", "stderr_artifact", "axioms_artifact", "claim_obligation_sha256")
        if any(not metadata.get(key) for key in required):
            failures.append(f"{claim_id}: incomplete verifier metadata")
        for key in ("stdout_artifact", "stderr_artifact", "axioms_artifact"):
            value = metadata.get(key)
            if value and value not in manifest:
                failures.append(f"{claim_id}: {key} is not durable")
        if claim.get("claim_class") == "bounded_correctness" and trust not in FORMAL_EXACT_CLAIM_ALLOWED_TRUST:
            failures.append(f"{claim_id}: verifier trust level is not allowed for an exact formal claim")
    return failures
