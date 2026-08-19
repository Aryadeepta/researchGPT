from src.decisions import list_open_decisions
from src.paper_gate import paper_readiness_report
from src.research_package import verify_research_package


def verify_research_run(store, run_id):
    state = store.load_state(run_id)
    if not state:
        return {"status": "FAIL", "reasons": ["missing run state"]}
    manifest_failures = store.verify_manifest(run_id)
    if manifest_failures:
        return {"status": "FAIL", "reasons": manifest_failures}
    if list_open_decisions(state):
        return {"status": "WAITING", "reasons": ["waiting for human decision"], "decisions": list_open_decisions(state)}
    status = state.get("status")
    if status == "RESEARCH_COMPLETE":
        package_report = verify_research_package(store, run_id)
        return package_report if package_report["status"] == "PASS" else {"status": "WAITING", "reasons": package_report.get("reasons", [])}
    if status in ("WAITING_FOR_HUMAN", "BLOCKED_BUDGET", "BLOCKED_EXTERNAL_RESOURCE", "BLOCKED_ENGINEERING_REQUIRED"):
        return {"status": "WAITING", "reasons": [status]}
    if status and status.startswith("BLOCKED"):
        return {"status": "WAITING", "reasons": [status]}
    readiness = paper_readiness_report(state, state.get("claim_evidence_ledger", {"claims": []}), store.load_manifest(run_id))
    return {"status": "WAITING", "reasons": readiness["errors"] or [status]}
