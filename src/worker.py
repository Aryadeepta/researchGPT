import argparse
import os
from uuid import uuid4

from src.executor import ResearchExecutor
from src.research_state import acquire_node_lease, create_run_state, finalize_leased_node_after_exception, recover_expired_leases
from src.storage import artifact_store_from_env
from src.supervisor import ResearchSupervisor


def load_or_create(store, run_id, topic=None):
    state = store.load_state(run_id)
    if state:
        return state
    if not topic:
        raise SystemExit(f"run {run_id} does not exist")
    state = create_run_state(run_id, topic)
    store.atomic_update_state(run_id, state)
    return state


def run_worker(args):
    store = artifact_store_from_env()
    state = load_or_create(store, args.run_id, args.topic)
    worker_id = os.environ.get("GITHUB_RUN_ID") or f"local-{uuid4().hex[:8]}"
    executor = ResearchExecutor(store)
    executed = 0
    recover_expired_leases(state)
    while executed < args.max_nodes:
        node = acquire_node_lease(state, worker_id=worker_id, ttl_seconds=args.lease_ttl)
        if not node:
            break
        store.atomic_update_state(args.run_id, state)
        try:
            executor.execute_node(state, node)
        except Exception as exc:
            finalize_leased_node_after_exception(state, node["node_id"], f"unexpected executor exception: {exc}")
        store.atomic_update_state(args.run_id, state)
        executed += 1
        if state.get("status", "").startswith("BLOCKED") or state.get("status") == "FAILED":
            break
    print(f"run_id={args.run_id} status={state.get('status')} executed_nodes={executed}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "resume", "worker", "supervise", "status"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--max-nodes", type=int, default=int(os.environ.get("RESEARCH_MAX_NODES_PER_WORKER", "1")))
    parser.add_argument("--lease-ttl", type=int, default=900)
    args = parser.parse_args()
    if args.command == "status":
        state = artifact_store_from_env().load_state(args.run_id)
        print(state.get("status", "missing") if state else "missing")
        return 0
    if args.command == "supervise":
        state, executed = ResearchSupervisor(artifact_store_from_env()).run_until_waiting(args.run_id, max_nodes=args.max_nodes, lease_ttl=args.lease_ttl)
        print(f"run_id={args.run_id} status={state.get('status')} executed_nodes={executed}")
        return 0
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
