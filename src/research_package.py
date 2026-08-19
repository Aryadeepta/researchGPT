import hashlib
import json
from pathlib import Path

from src.paper_gate import paper_readiness_report
from src.objective_coverage import objective_coverage_sufficient
from src.research_state import now_iso


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
    if not package.get("research_readiness_report", {}).get("ready"):
        return {"status": "WAITING", "reasons": package.get("research_readiness_report", {}).get("errors", [])}
    return {"status": "PASS", "package": {"package_id": package["package_id"], "version": package["version"], "hash": package["package_hash"]}}
