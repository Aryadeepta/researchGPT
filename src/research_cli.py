import argparse
import json

from src.decisions import list_open_decisions
from src.objective_coverage import repair_research_completion_coverage
from src.research_package import write_research_package
from src.research_runtime import (
    invalidate_stale_local_routing_decisions,
    llm_attempts_for_stage,
    regenerate_external_reasoning_bundle,
    repair_invalid_evidence_relevance,
    repair_recoverable_structured_generation_failures,
    reconcile_external_decision_continuation,
    submit_research_decision,
)
from src.research_state import create_run_state, recover_expired_leases
from src.storage import artifact_store_from_env
from src.supervisor import ResearchSupervisor
from src.verification import verify_research_run
from src.worker import run_worker
from src.budget_control import extend_budget, reconcile_legacy_agent_iteration_blocks
from src.llm_gateway import LLMBudgetManager
from src.run_inspection import (export_graph, graph_dot, graph_mermaid, graph_summary,
                                provenance_manifest, replay_dry_run)


def print_json(data):
    print(json.dumps(data, indent=2, sort_keys=True))


def research_report(state):
    nodes = state.get("dag", {}).get("nodes", {})
    return {
        "run_id": state.get("run_id"), "status": state.get("status"), "topic": state.get("topic"),
        "selected_question": state.get("selected_question") or state.get("research_spec", {}).get("research_question"),
        "literature_records": len(state.get("literature_cache", [])),
        "current_nodes": {k: v.get("status") for k, v in nodes.items() if v.get("status") != "COMPLETED"},
        "skills": [entry.get("skill", {}).get("skill_id") for entry in state.get("skill_registry", [])],
        "execution_records": state.get("execution_records", []),
        "claims": state.get("claim_evidence_ledger", {}).get("claims", []),
        "adversarial_findings": state.get("adversarial_findings", []),
        "replication_status": state.get("replication_status"),
        "selected_objective_coverage": state.get("selected_objective_coverage"),
        "requirement_lifecycle": state.get("requirement_lifecycle", []),
        "llm_usage": state.get("budget", {}),
        "llm_attempt_summary": {stage: llm_attempts_for_stage(state, stage)
            for stage in sorted({call.get("stage") for call in state.get("budget", {}).get("calls", []) if call.get("stage")})},
        "research_packages": state.get("research_packages", []), "open_decisions": list_open_decisions(state),
        "decision_history_summary": [{
            "decision_id": item.get("decision_id"), "status": item.get("status"),
            "response_kind": item.get("response_kind"), "semantic_task": (item.get("response_contract") or {}).get("semantic_task"),
            "selected_option": item.get("selected_option"), "resolved_at": item.get("resolved_at"),
            "external_response_artifact": item.get("external_response_artifact"), "continuation_id": item.get("continuation_id"),
        } for item in state.get("decision_history", [])],
        "external_reasoning_responses": [{
            "decision_id": item.get("decision_id"), "semantic_task": item.get("semantic_task"),
            "response_kind": item.get("response_kind"), "parsed_response": item.get("parsed_response"),
            "response_validation": item.get("response_validation"), "response_artifact": item.get("response_artifact"),
            "continuation_type": item.get("continuation_type"), "continuation_status": item.get("status"),
            "continuation_validation_result": item.get("continuation_validation_result"),
        } for item in state.get("external_decision_continuations", [])],
        "node_failures": state.get("node_failures", []) + [{
            "node_id": node_id, "failure_class": "LEGACY_NODE_FAILURE", "failure_reason": item.get("failure_reason"),
            "triggering_operation": "node_execution", "relevant_artifact_refs": [], "recoverability": "UNKNOWN",
            "timestamp": item.get("updated_at"),
        } for node_id, item in nodes.items() if item.get("status") == "FAILED"
          and not any(f.get("node_id") == node_id and f.get("failure_reason") == item.get("failure_reason")
                      for f in state.get("node_failures", []))],
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "resume", "worker", "supervise", "status", "decisions", "verify", "package", "skills", "report", "graph", "inspect", "provenance", "replay", "repair-routing-decisions", "repair-structured-generation", "repair-external-reasoning", "repair-external-continuation", "repair-evidence-relevance", "repair-research-completion-coverage"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--topic")
        p.add_argument("--max-nodes", type=int, default=1)
        p.add_argument("--lease-ttl", type=int, default=900)
        p.add_argument("--decision-id")
        p.add_argument("--format", choices=["json", "summary", "mermaid", "dot"], default="json")
        if name == "replay":
            p.add_argument("--dry-run", action="store_true", help="required: inspect hashes and ordering without execution")
    extend = sub.add_parser("extend-budget")
    extend.add_argument("--run-id", required=True)
    extend.add_argument("--budget-type", required=True, choices=["agent_iterations"])
    extend.add_argument("--new-limit", required=True, type=int)
    extend.add_argument("--reason", required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--run-id", required=True)
    decide.add_argument("--decision-id", required=True)
    decide.add_argument("--option")
    decide.add_argument("--text")
    decide.add_argument("--response-file")
    args = parser.parse_args()
    store = artifact_store_from_env()

    if args.command == "start":
        if not args.topic:
            raise SystemExit("--topic is required for start")
        if not store.load_state(args.run_id):
            store.atomic_update_state(args.run_id, create_run_state(args.run_id, args.topic))
        return run_worker(args)
    if args.command == "extend-budget":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        reconcile_legacy_agent_iteration_blocks(state, LLMBudgetManager().max_agent_iterations)
        try:
            result = extend_budget(state, args.budget_type, args.new_limit, args.reason)
        except ValueError as exc:
            raise SystemExit(str(exc))
        store.atomic_update_state(args.run_id, state)
        print_json(result)
        return 0
    if args.command in ("resume", "worker"):
        return run_worker(args)
    if args.command == "supervise":
        state, executed = ResearchSupervisor(store).run_until_waiting(args.run_id, max_nodes=args.max_nodes, lease_ttl=args.lease_ttl)
        print(f"run_id={args.run_id} status={state.get('status')} executed_nodes={executed}")
        return 0
    if args.command == "status":
        state = store.load_state(args.run_id)
        print(state.get("status", "missing") if state else "missing")
        return 0
    if args.command in ("graph", "inspect"):
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        graph = export_graph(state, store.load_manifest(args.run_id))
        if args.format == "mermaid":
            print(graph_mermaid(graph))
        elif args.format == "dot":
            print(graph_dot(graph))
        else:
            print_json(graph_summary(graph) if args.format == "summary" else graph)
        return 0
    if args.command == "provenance":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        print_json(provenance_manifest(state, store.load_manifest(args.run_id)))
        return 0
    if args.command == "replay":
        if not args.dry_run:
            raise SystemExit("replay is read-only; pass --dry-run")
        if getattr(args, "format", "json") not in ("json", "summary"):
            raise SystemExit("replay supports JSON only")
        print_json(replay_dry_run(store, args.run_id))
        return 0
    if args.command == "decisions":
        print_json(list_open_decisions(store.load_state(args.run_id) or {}))
        return 0
    if args.command == "repair-routing-decisions":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        invalidated = invalidate_stale_local_routing_decisions(state)
        repaired_structured = repair_recoverable_structured_generation_failures(state)
        recovered_leases = recover_expired_leases(state)
        store.atomic_update_state(args.run_id, state)
        print_json({
            "run_id": args.run_id,
            "invalidated_decision_ids": invalidated,
            "repaired_structured_generation_nodes": repaired_structured,
            "recovered_leases": recovered_leases,
            "llm_usd": state.get("budget", {}).get("llm_usd"),
            "status": state.get("status"),
        })
        return 0
    if args.command == "repair-structured-generation":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        repaired_structured = repair_recoverable_structured_generation_failures(state)
        recovered_leases = recover_expired_leases(state)
        store.atomic_update_state(args.run_id, state)
        print_json({
            "run_id": args.run_id,
            "repaired_structured_generation_nodes": repaired_structured,
            "recovered_leases": recovered_leases,
            "llm_usd": state.get("budget", {}).get("llm_usd"),
            "literature_records": len(state.get("literature_cache", [])),
            "open_decisions": list_open_decisions(state),
            "status": state.get("status"),
        })
        return 0
    if args.command == "repair-external-reasoning":
        if not args.decision_id:
            raise SystemExit("--decision-id is required")
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        result = regenerate_external_reasoning_bundle(store, state, args.decision_id)
        store.atomic_update_state(args.run_id, state)
        print_json(result)
        return 0
    if args.command == "repair-external-continuation":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        result = reconcile_external_decision_continuation(store, state, args.decision_id)
        store.atomic_update_state(args.run_id, state)
        if any(item.get("action") == "RETRY_STARTED" for item in result):
            ResearchSupervisor(store).run_until_waiting(args.run_id, max_nodes=1, lease_ttl=args.lease_ttl)
            final_state = store.load_state(args.run_id)
            for item in result:
                lifecycle = next((entry for entry in final_state.get("external_decision_continuations", [])
                                  if entry.get("continuation_id") == item.get("continuation_id")), {})
                node = final_state.get("dag", {}).get("nodes", {}).get(lifecycle.get("continuation_target"), {})
                item["resulting_continuation_status"] = lifecycle.get("status")
                item["resulting_node_status"] = node.get("status")
                item["action"] = ({"CONTINUATION_APPLIED": "RETRY_SUCCEEDED",
                                   "CONTINUATION_SEMANTIC_REJECTED": "RETRY_SEMANTIC_REJECTED",
                                   "CONTINUATION_BLOCKED_ENGINEERING": "RETRY_BLOCKED_ENGINEERING"}
                                  .get(lifecycle.get("status"), item["action"]))
        print_json(result)
        return 0
    if args.command == "repair-evidence-relevance":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        result = repair_invalid_evidence_relevance(state)
        store.atomic_update_state(args.run_id, state)
        print_json(result)
        return 0
    if args.command == "repair-research-completion-coverage":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        result = repair_research_completion_coverage(store, state)
        store.atomic_update_state(args.run_id, state)
        print_json(result)
        return 0
    if args.command == "decide":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        text = args.text
        if args.response_file:
            with open(args.response_file, "r", encoding="utf-8") as f:
                text = f.read()
        decision = submit_research_decision(store, state, args.decision_id, args.option, text)
        store.atomic_update_state(args.run_id, state)
        print_json(decision)
        return 0
    if args.command == "verify":
        print_json(verify_research_run(store, args.run_id))
        return 0
    if args.command == "package":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        package, report = write_research_package(store, state)
        store.atomic_update_state(args.run_id, state)
        print_json(package or report)
        return 0
    if args.command == "skills":
        root = store.run_root(args.run_id) / "skills"
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []
        print_json(files)
        return 0
    if args.command == "report":
        state = store.load_state(args.run_id)
        if not state:
            raise SystemExit(f"run {args.run_id} does not exist")
        print_json(research_report(state))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
