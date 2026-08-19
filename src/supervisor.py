import os
from uuid import uuid4

from src.decisions import list_open_decisions
from src.executor import ResearchExecutor
from src.notifier import notifier_from_env
from src.research_package import write_research_package
from src.research_runtime import invalidate_stale_local_routing_decisions, validate_node_llm_task_class
from src.research_state import acquire_node_lease, finalize_leased_node_after_exception, recover_expired_leases


TERMINAL_OR_WAITING = {
    "RESEARCH_COMPLETE",
    "WAITING_FOR_HUMAN",
    "BLOCKED_BUDGET",
    "BLOCKED_MISSING_LLM",
    "BLOCKED_EXTERNAL_RESOURCE",
    "BLOCKED_ENGINEERING_REQUIRED",
    "BLOCKED_MISSING_EVIDENCE",
    "BLOCKED_INVALID_METHOD",
    "BLOCKED_REPLICATION_FAILURE",
    "FAILED",
    "STOPPED",
}


class ResearchSupervisor:
    def __init__(self, store, notifier=None):
        self.store = store
        self.notifier = notifier or notifier_from_env()

    def run_until_waiting(self, run_id, max_nodes=50, lease_ttl=900):
        state = self.store.load_state(run_id)
        if not state:
            raise SystemExit(f"run {run_id} does not exist")
        invalidate_stale_local_routing_decisions(state)
        recover_expired_leases(state)
        for node in state.get("dag", {}).get("nodes", {}).values():
            error = validate_node_llm_task_class(node)
            if error:
                state["status"] = "BLOCKED_ENGINEERING_REQUIRED"
                node["status"] = "BLOCKED_ENGINEERING_REQUIRED"
                node["failure_reason"] = error
                self.store.atomic_update_state(run_id, state)
                self.notifier.notify_transition(state, "BLOCKED_ENGINEERING_REQUIRED", {"node_id": node["node_id"], "reason": error})
                return state, 0
        worker_id = os.environ.get("GITHUB_RUN_ID") or f"supervisor-{uuid4().hex[:8]}"
        executor = ResearchExecutor(self.store)
        executed = 0
        while executed < max_nodes:
            if list_open_decisions(state):
                state["status"] = "WAITING_FOR_HUMAN"
                self.store.atomic_update_state(run_id, state)
                self.notifier.notify_transition(state, "HUMAN_REQUIRED", list_open_decisions(state)[0])
                break
            if state.get("status") in TERMINAL_OR_WAITING and state.get("status") != "RESEARCH_COMPLETE":
                self.notifier.notify_transition(state, state.get("status"), {"run_id": run_id})
                break
            node = acquire_node_lease(state, worker_id=worker_id, ttl_seconds=lease_ttl)
            if not node:
                package, report = write_research_package(self.store, state)
                if package:
                    self.store.atomic_update_state(run_id, state)
                    self.notifier.notify_transition(state, "RESEARCH_COMPLETE", package)
                else:
                    state["status"] = report["status"]
                    self.store.atomic_update_state(run_id, state)
                break
            self.store.atomic_update_state(run_id, state)
            try:
                executor.execute_node(state, node)
            except Exception as exc:
                finalize_leased_node_after_exception(state, node["node_id"], f"unexpected executor exception: {exc}")
            self.store.atomic_update_state(run_id, state)
            executed += 1
            if state.get("status") in TERMINAL_OR_WAITING:
                self.notifier.notify_transition(state, state.get("status"), {"node_id": node["node_id"], "reason": node.get("failure_reason")})
                break
        return state, executed
