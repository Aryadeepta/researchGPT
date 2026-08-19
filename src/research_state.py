from copy import deepcopy
from datetime import datetime, timedelta, timezone
from uuid import uuid4


TERMINAL_NODE_STATUSES = {
    "COMPLETED",
    "BLOCKED_BUDGET",
    "BLOCKED_MISSING_LLM",
    "BLOCKED_EXTERNAL_RESOURCE",
    "BLOCKED_ENGINEERING_REQUIRED",
    "BLOCKED_MISSING_EVIDENCE",
    "BLOCKED_INVALID_METHOD",
    "BLOCKED_REPLICATION_FAILURE",
    "WAITING_FOR_HUMAN",
    "FAILED",
    "STOPPED",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def default_research_dag(topic):
    return [
        {"node_id": "question_discovery", "kind": "planning", "llm_task_class": "candidate_question_generation", "dependencies": [], "contract": {"outputs": ["research_spec.json"], "allow_placeholder_outputs": True}},
        {"node_id": "evidence_discovery", "kind": "evidence", "dependencies": ["question_discovery"], "contract": {"capability_required": True, "outputs": ["evidence/discovery.json"]}},
        {"node_id": "question_refinement", "kind": "planning", "llm_task_class": "candidate_question_generation", "dependencies": ["evidence_discovery"], "contract": {"outputs": ["specification.json"], "allow_placeholder_outputs": True}},
        {"node_id": "feasibility_analysis", "kind": "planning", "llm_task_class": "research_feasibility_analysis", "dependencies": ["question_refinement"], "contract": {"outputs": ["feasibility.json"], "allow_placeholder_outputs": True}},
        {"node_id": "capability_gap_analysis", "kind": "planning", "dependencies": ["feasibility_analysis"], "contract": {"outputs": ["capabilities/requirements.json"], "allow_placeholder_outputs": True}},
        {"node_id": "skill_discovery_creation", "kind": "capability", "dependencies": ["capability_gap_analysis"], "contract": {"outputs": ["skills/registry.json"], "allow_placeholder_outputs": True}},
        {"node_id": "executable_artifact_dag", "kind": "planning", "dependencies": ["skill_discovery_creation"], "contract": {"outputs": ["artifact_dag.json"], "allow_placeholder_outputs": True}},
        {"node_id": "research_execution", "kind": "execution", "dependencies": ["executable_artifact_dag"], "contract": {"capability_required": True, "requires_execution": True, "raw_outputs": ["execution/provenance.json"]}},
        {"node_id": "independent_validation", "kind": "validation", "dependencies": ["research_execution"], "contract": {"independent_validator": True, "outputs": ["validation/report.json"], "allow_placeholder_outputs": True}},
        {"node_id": "adversarial_falsification", "kind": "adversarial", "dependencies": ["independent_validation"], "contract": {"outputs": ["adversarial/findings.json"], "allow_placeholder_outputs": True}},
        {"node_id": "replication", "kind": "replication", "dependencies": ["adversarial_falsification"], "contract": {"requires_execution": True, "outputs": ["replication/report.json"]}},
        {"node_id": "claim_adjudication", "kind": "validation", "dependencies": ["replication"], "contract": {"outputs": ["claims/adjudication.json"], "allow_placeholder_outputs": True}},
        {"node_id": "research_readiness", "kind": "gate", "dependencies": ["claim_adjudication"], "contract": {"research_gate": True, "outputs": ["readiness/report.json"], "allow_placeholder_outputs": True}},
    ]


def create_run_state(run_id, topic, dag=None):
    dag = dag or default_research_dag(topic)
    nodes = {}
    for node in dag:
        node_id = node["node_id"]
        nodes[node_id] = {
            **deepcopy(node),
            "status": "PENDING",
            "lease": None,
            "attempts": 0,
            "failure_reason": None,
            "artifacts": [],
            # Artifact production and verifier acceptance are intentionally
            # separate dimensions.  Completion alone never upgrades trust.
            "verification_state": "GENERATED_UNVERIFIED",
            "verification_evidence": [],
            "verification_history": [],
            "updated_at": now_iso(),
        }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "topic": topic,
        "status": "PLANNED_RESEARCH",
        "dag": {"nodes": nodes},
        "research_spec": {},
        "claim_evidence_ledger": {"claims": []},
        "artifact_manifest": {"artifacts": []},
        "budget": {"llm_usd": 0.0, "strong_calls": 0, "calls": []},
        "decisions": [],
        "decision_history": [],
        "engineering_requests": [],
        "notifications_sent": [],
        "dag_generation": 1,
        "research_packages": [],
        "execution_records": [],
        "validation_reports": [],
        "adversarial_findings": [],
        "replication_reports": [],
        "known_limitations": [],
        "unresolved_findings": [],
        "literature_cache": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def dependencies_satisfied(state, node):
    nodes = state["dag"]["nodes"]
    return all(nodes[dep]["status"] == "COMPLETED" for dep in node.get("dependencies", []))


def ready_nodes(state):
    return [
        node
        for node in state["dag"]["nodes"].values()
        if node["status"] == "PENDING" and dependencies_satisfied(state, node)
    ]


def recover_expired_leases(state):
    recovered = []
    now = datetime.now(timezone.utc)
    for node in state["dag"]["nodes"].values():
        if node.get("status") != "LEASED":
            continue
        lease = node.get("lease") or {}
        expires = lease.get("expires_at")
        try:
            expired = datetime.fromisoformat(expires) <= now if expires else True
        except ValueError:
            expired = True
        if expired:
            node["status"] = "PENDING"
            node["lease"] = None
            node["failure_reason"] = "recovered expired lease"
            node["updated_at"] = now_iso()
            recovered.append(node["node_id"])
    if recovered:
        state["updated_at"] = now_iso()
        if state.get("status") == "PLANNED_RESEARCH":
            state["status"] = "PLANNED_RESEARCH"
    return recovered


def acquire_node_lease(state, worker_id=None, ttl_seconds=900):
    recover_expired_leases(state)
    worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    for node in ready_nodes(state):
        node["status"] = "LEASED"
        node["lease"] = {"worker_id": worker_id, "expires_at": expires_at.isoformat()}
        node["attempts"] = int(node.get("attempts", 0)) + 1
        node["updated_at"] = now_iso()
        state["updated_at"] = now_iso()
        return node
    return None


def complete_node(state, node_id, artifacts=None):
    node = state["dag"]["nodes"][node_id]
    node["status"] = "COMPLETED"
    node["lease"] = None
    node["failure_reason"] = None
    node["artifacts"] = artifacts or node.get("artifacts", [])
    node["updated_at"] = now_iso()
    state["updated_at"] = now_iso()
    if all(n["status"] == "COMPLETED" for n in state["dag"]["nodes"].values()):
        from src.objective_coverage import objective_coverage_sufficient
        state["status"] = "RESEARCH_COMPLETE" if objective_coverage_sufficient(state) else "PARTIAL_RESEARCH"


def record_verification(state, node_id, status, evidence_references=None, verifier=None, reason=None):
    """Append-only verifier event; failed evidence remains visible for repair."""
    if status not in {"VERIFYING", "VERIFIED", "VERIFICATION_FAILED", "REPAIR_REQUESTED", "REPLAN_REQUESTED"}:
        raise ValueError(f"invalid verification status: {status}")
    node = state["dag"]["nodes"][node_id]
    event = {"status": status, "evidence_references": list(evidence_references or []),
             "verifier": verifier, "reason": reason, "timestamp": now_iso()}
    node.setdefault("verification_history", []).append(event)
    node["verification_state"] = status
    node["verification_evidence"] = list(evidence_references or node.get("verification_evidence", []))
    if verifier:
        node["verifier"] = verifier
    node["updated_at"] = now_iso()
    state["updated_at"] = now_iso()
    return event


def block_node(state, node_id, status, reason):
    if status not in TERMINAL_NODE_STATUSES:
        raise ValueError(f"invalid blocked status: {status}")
    node = state["dag"]["nodes"][node_id]
    node["status"] = status
    node["lease"] = None
    node["failure_reason"] = reason
    node["updated_at"] = now_iso()
    state["status"] = status
    state["updated_at"] = now_iso()
    if status == "FAILED":
        state.setdefault("node_failures", []).append({
            "node_id": node_id, "failure_class": "NODE_EXECUTION_FAILURE",
            "failure_reason": reason or "terminal failure without an explicit reason",
            "triggering_operation": "block_node", "relevant_artifact_refs": [],
            "recoverability": "UNKNOWN", "timestamp": now_iso(),
        })


def release_node_lease(state, node_id):
    node = state["dag"]["nodes"][node_id]
    if node["status"] == "LEASED":
        node["status"] = "PENDING"
        node["lease"] = None
        node["updated_at"] = now_iso()
        state["updated_at"] = now_iso()


def finalize_leased_node_after_exception(state, node_id, reason, retryable=False):
    node = state["dag"]["nodes"][node_id]
    if node.get("status") == "LEASED":
        node["status"] = "PENDING" if retryable else "FAILED"
        node["lease"] = None
        node["failure_reason"] = reason
        node["updated_at"] = now_iso()
        state["status"] = "PLANNED_RESEARCH" if retryable else "FAILED"
        state["updated_at"] = now_iso()
        if not retryable:
            state.setdefault("node_failures", []).append({
                "node_id": node_id, "failure_class": "UNEXPECTED_EXECUTOR_EXCEPTION",
                "failure_reason": reason or "terminal failure without an explicit reason",
                "triggering_operation": "finalize_leased_node_after_exception",
                "relevant_artifact_refs": [], "recoverability": "UNKNOWN", "timestamp": now_iso(),
            })
