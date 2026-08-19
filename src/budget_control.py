from uuid import uuid4

from src.research_state import now_iso


INFRASTRUCTURE_FAILURE_TYPES = {
    "TRANSIENT_LOCAL_RUNTIME_FAILURE",
    "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE",
}


def agent_iteration_usage(state):
    """Count semantic attempts; infrastructure-only failures do not consume this budget."""
    return sum(
        1 for call in state.get("budget", {}).get("calls", [])
        if call.get("failure_type") not in INFRASTRUCTURE_FAILURE_TYPES
    )


def authorized_limit(state, budget_type, initial_limit):
    limits = [
        int(item["new_limit"])
        for item in state.get("budget_authorizations", [])
        if item.get("budget_type") == budget_type
    ]
    return limits[-1] if limits else int(initial_limit)


def record_budget_block(state, node_id, budget_type, limit, usage, triggering_operation, reason):
    record = {
        "budget_block_id": f"B{uuid4().hex[:8]}",
        "budget_type": budget_type,
        "budget_scope": "RUN_LIFETIME",
        "limit_at_block": int(limit),
        "usage_at_block": int(usage),
        "node_id": node_id,
        "triggering_operation": triggering_operation,
        "timestamp": now_iso(),
        "recoverability": "REQUIRES_EXPLICIT_BUDGET_EXTENSION",
        "reason": reason,
    }
    state.setdefault("budget_blocks", []).append(record)
    state.setdefault("budget_events", []).append({
        "event": "BUDGET_BLOCKED", "timestamp": record["timestamp"],
        "budget_block_id": record["budget_block_id"], "node_id": node_id,
        "budget_type": budget_type,
    })
    return record


def reconcile_legacy_agent_iteration_blocks(state, initial_limit):
    """Add provenance for pre-feature blocks without erasing or rewriting them."""
    recorded = {b.get("node_id") for b in state.get("budget_blocks", [])
                if b.get("budget_type") == "agent_iterations"}
    created = []
    for node_id, node in state.get("dag", {}).get("nodes", {}).items():
        if (node.get("status") == "BLOCKED_BUDGET"
                and node.get("failure_reason") == "MAX_AGENT_ITERATIONS"
                and node_id not in recorded):
            created.append(record_budget_block(
                state, node_id, "agent_iterations", initial_limit,
                agent_iteration_usage(state), node.get("llm_task_class") or node_id,
                "MAX_AGENT_ITERATIONS"))
    return created


def extend_budget(state, budget_type, new_limit, reason, source="USER_DELEGATED_CODEX"):
    if budget_type != "agent_iterations":
        raise ValueError(f"unsupported budget type: {budget_type}")
    if not reason or not reason.strip():
        raise ValueError("budget extension reason must be non-empty")
    usage = agent_iteration_usage(state)
    matching_blocks = [b for b in state.get("budget_blocks", []) if b.get("budget_type") == budget_type]
    if state.get("budget_authorizations"):
        matching_auth = [a for a in state["budget_authorizations"] if a.get("budget_type") == budget_type]
    else:
        matching_auth = []
    previous_limit = int(matching_auth[-1]["new_limit"]) if matching_auth else (
        int(matching_blocks[-1]["limit_at_block"]) if matching_blocks else 0
    )
    new_limit = int(new_limit)
    if new_limit <= previous_limit:
        raise ValueError(f"new limit must exceed previous authorized limit {previous_limit}")
    timestamp = now_iso()
    authorization = {
        "extension_id": f"E{uuid4().hex[:8]}", "run_id": state.get("run_id"),
        "budget_type": budget_type, "previous_limit": previous_limit, "new_limit": new_limit,
        "usage_at_authorization": usage, "additional_headroom": max(0, new_limit - usage),
        "timestamp": timestamp, "source": source, "reason": reason.strip(),
    }
    state.setdefault("budget_authorizations", []).append(authorization)
    state.setdefault("budget_events", []).append({
        "event": "BUDGET_EXTENSION_AUTHORIZED", "timestamp": timestamp,
        "extension_id": authorization["extension_id"], "budget_type": budget_type,
        "previous_limit": previous_limit, "new_limit": new_limit,
    })
    reopened = []
    state["budget_events"].append({
        "event": "BUDGET_PRECONDITION_RECHECK", "timestamp": now_iso(),
        "extension_id": authorization["extension_id"], "budget_type": budget_type,
        "authorized_limit": new_limit, "current_usage": usage,
        "result": "PASSED" if new_limit > usage else "STILL_EXHAUSTED",
    })
    if new_limit > usage:
        blocked_node_ids = {
            item.get("node_id") for item in matching_blocks
            if item.get("recoverability") == "REQUIRES_EXPLICIT_BUDGET_EXTENSION"
        }
        for node_id in blocked_node_ids:
            node = state.get("dag", {}).get("nodes", {}).get(node_id)
            if node and node.get("status") == "BLOCKED_BUDGET":
                node.update({"status": "PENDING", "lease": None, "failure_reason": None,
                             "updated_at": now_iso()})
                reopened.append(node_id)
        if reopened and state.get("status") == "BLOCKED_BUDGET":
            remaining = any(n.get("status") == "BLOCKED_BUDGET"
                            for n in state.get("dag", {}).get("nodes", {}).values())
            if not remaining:
                state["status"] = "PLANNED_RESEARCH"
    state["updated_at"] = now_iso()
    return {"authorization": authorization, "reopened_nodes": sorted(reopened),
            "current_usage": usage, "status": state.get("status")}
