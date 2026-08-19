"""Read-only public inspection for ResearchGPT execution records.

These functions deliberately derive views from the existing run state and
artifact manifest.  They neither advance DAG nodes nor rewrite provenance.
"""
from __future__ import annotations

from collections import defaultdict, deque


def _artifact_paths(values):
    return [item.get("path") if isinstance(item, dict) else item for item in (values or [])
            if (item.get("path") if isinstance(item, dict) else item)]


def _verification_state(node):
    # Completion is execution state, not a trust assertion.  Only an explicit
    # verifier result may turn this field into VERIFIED.
    return node.get("verification_state") or "UNVERIFIED"


def export_graph(state, manifest):
    """Return a stable, typed DAG view derived from a run's authoritative state."""
    nodes = state.get("dag", {}).get("nodes", {})
    by_path = {entry.get("path"): entry for entry in manifest.get("artifacts", []) if entry.get("path")}
    children = defaultdict(list)
    for node_id, node in nodes.items():
        for dependency in node.get("dependencies", []):
            children[dependency].append(node_id)
    exported = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        paths = _artifact_paths(node.get("artifacts"))
        upstream_paths = []
        for dependency in node.get("dependencies", []):
            upstream_paths.extend(_artifact_paths(nodes.get(dependency, {}).get("artifacts")))
        exported.append({
            "node_id": node_id,
            "node_type": node.get("kind", "unknown"),
            "semantic_role": node.get("semantic_role", node.get("kind", "unknown")),
            "declared_inputs": node.get("contract", {}).get("inputs", []),
            "declared_outputs": node.get("contract", {}).get("outputs", []) + node.get("contract", {}).get("raw_outputs", []),
            "dependencies": sorted(node.get("dependencies", [])),
            "downstream": sorted(children[node_id]),
            "status": node.get("status", "UNKNOWN"),
            "executor": (node.get("lease") or {}).get("worker_id") or node.get("executor"),
            "verifier": node.get("verifier"),
            "generated_artifacts": [{"path": path, "sha256": by_path.get(path, {}).get("sha256")} for path in paths],
            "verification_state": _verification_state(node),
            "verification_evidence_references": node.get("verification_evidence", []),
            "cost_resource_metadata": node.get("cost_resource_metadata", {}),
            "timestamps": {key: node[key] for key in ("created_at", "updated_at") if node.get(key)},
            "attempts": node.get("attempts", 0),
            "replans": node.get("replans", []),
            "upstream_artifact_hashes": [{"path": path, "sha256": by_path.get(path, {}).get("sha256")} for path in sorted(set(upstream_paths))],
        })
    return {"schema_version": 1, "run_id": state.get("run_id"), "topic": state.get("topic"),
            "run_status": state.get("status"), "nodes": exported,
            "edges": [{"from": dep, "to": node_id} for node_id, node in sorted(nodes.items()) for dep in sorted(node.get("dependencies", []))]}


def graph_summary(graph):
    counts = defaultdict(int)
    for node in graph["nodes"]:
        counts[node["status"]] += 1
    return {"run_id": graph["run_id"], "topic": graph["topic"], "run_status": graph["run_status"],
            "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"]), "status_counts": dict(sorted(counts.items()))}


def graph_mermaid(graph):
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        label = f'{node["node_id"]}\\n{node["node_type"]} · {node["status"]} · {node["verification_state"]}'.replace('"', "'")
        lines.append(f'  {node["node_id"]}["{label}"]')
    lines.extend(f'  {edge["from"]} --> {edge["to"]}' for edge in graph["edges"])
    return "\n".join(lines) + "\n"


def graph_dot(graph):
    lines = ["digraph research_run {", "  rankdir=LR;"]
    for node in graph["nodes"]:
        label = f'{node["node_id"]}\\n{node["node_type"]}\\n{node["status"]} / {node["verification_state"]}'.replace('"', "'")
        lines.append(f'  "{node["node_id"]}" [label="{label}"];')
    lines.extend(f'  "{edge["from"]}" -> "{edge["to"]}";' for edge in graph["edges"])
    return "\n".join(lines + ["}"]) + "\n"


def provenance_manifest(state, manifest):
    graph = export_graph(state, manifest)
    claims = state.get("claim_evidence_ledger", {}).get("claims", [])
    return {"schema_version": 1, "kind": "replayable_execution_manifest", "run_id": state.get("run_id"),
            "objective": state.get("topic"), "graph": graph, "artifacts": manifest.get("artifacts", []),
            "execution_records": state.get("execution_records", []), "validation_reports": state.get("validation_reports", []),
            "node_failures": state.get("node_failures", []), "decisions": state.get("decision_history", []),
            "claims": claims, "final_claim_evidence": [claim_trace(claim, manifest) for claim in claims]}


def claim_trace(claim, manifest):
    by_path = {entry.get("path"): entry for entry in manifest.get("artifacts", [])}
    refs = _artifact_paths(claim.get("artifacts")) + _artifact_paths(claim.get("validator_artifacts"))
    return {"claim_id": claim.get("claim_id"), "claim": claim.get("claim"), "status": claim.get("status"),
            "validated_by": claim.get("validated_by", []),
            "evidence": [{"path": ref, "sha256": by_path.get(ref, {}).get("sha256"),
                          "kind": "validator" if ref in _artifact_paths(claim.get("validator_artifacts")) else "artifact"}
                         for ref in refs]}


def replay_dry_run(store, run_id):
    """Verify stored hashes and reconstruct a dependency order; never executes workers/models."""
    state = store.load_state(run_id)
    if not state:
        return {"run_id": run_id, "status": "MISSING_RUN", "model_execution": False}
    manifest = store.load_manifest(run_id)
    graph = export_graph(state, manifest)
    dependencies = {node["node_id"]: set(node["dependencies"]) for node in graph["nodes"]}
    ready = deque(sorted(node_id for node_id, deps in dependencies.items() if not deps))
    order = []
    while ready:
        node_id = ready.popleft(); order.append(node_id)
        for child, deps in dependencies.items():
            deps.discard(node_id)
            if not deps and child not in order and child not in ready:
                ready.append(child)
    cycles = sorted(set(dependencies) - set(order))
    failures = store.verify_manifest(run_id)
    return {"run_id": run_id, "status": "OK" if not failures and not cycles else "INVALID_PROVENANCE",
            "model_execution": False, "verifier_execution": False, "planned_dependency_order": order,
            "cycles": cycles, "artifact_integrity_failures": failures}
