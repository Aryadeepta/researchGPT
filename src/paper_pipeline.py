import json
from pathlib import Path
from uuid import uuid4

from src.decisions import DecisionEngine, list_open_decisions, submit_decision
from src.research_package import load_research_package, stable_hash, verify_research_package
from src.research_state import now_iso


def create_paper_state(paper_id, research_run_id, research_package):
    package_ref = {
        "package_id": research_package["package_id"],
        "version": research_package["version"],
        "package_hash": research_package["package_hash"],
        "research_run_id": research_run_id,
    }
    return {
        "schema_version": 1,
        "paper_id": paper_id,
        "research_run_id": research_run_id,
        "status": "PLANNED_PAPER",
        "research_package_ref": package_ref,
        "dag": {
            "nodes": {
                "package_validation": {"node_id": "package_validation", "status": "PENDING", "dependencies": []},
                "paper_blueprint": {"node_id": "paper_blueprint", "status": "PENDING", "dependencies": ["package_validation"]},
                "section_drafting": {"node_id": "section_drafting", "status": "PENDING", "dependencies": ["paper_blueprint"]},
                "claim_evidence_lint": {"node_id": "claim_evidence_lint", "status": "PENDING", "dependencies": ["section_drafting"]},
                "paper_package": {"node_id": "paper_package", "status": "PENDING", "dependencies": ["claim_evidence_lint"]},
            }
        },
        "decisions": [],
        "decision_history": [],
        "notifications_sent": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


class PaperStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def paper_root(self, paper_id):
        return self.root / paper_id

    def state_path(self, paper_id):
        return self.paper_root(paper_id) / "paper_state.json"

    def load(self, paper_id):
        path = self.state_path(paper_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, state):
        root = self.paper_root(state["paper_id"])
        root.mkdir(parents=True, exist_ok=True)
        self.state_path(state["paper_id"]).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ready_paper_nodes(state):
    nodes = state["dag"]["nodes"]
    ready = []
    for node in nodes.values():
        if node["status"] != "PENDING":
            continue
        if all(nodes[d]["status"] == "COMPLETED" for d in node.get("dependencies", [])):
            ready.append(node)
    return ready


class PaperPipeline:
    def __init__(self, research_store, paper_store, notifier=None):
        self.research_store = research_store
        self.paper_store = paper_store
        self.notifier = notifier
        self.decision_engine = DecisionEngine()

    def start(self, paper_id, research_run_id, package_version=None):
        research_state = self.research_store.load_state(research_run_id) or {}
        verification = verify_research_package(self.research_store, research_run_id, package_version)
        coverage = research_state.get("selected_objective_coverage", {})
        if verification["status"] != "PASS" or coverage.get("status") != "SUFFICIENT":
            state = {
                "schema_version": 1,
                "paper_id": paper_id,
                "research_run_id": research_run_id,
                "status": "BLOCKED_RESEARCH_NOT_READY",
                "failure_reason": verification.get("reasons", []) + ([] if coverage.get("status") == "SUFFICIENT" else ["selected research objective coverage is insufficient; package may contain only a scoped substudy"]),
                "decisions": [],
                "decision_history": [],
                "notifications_sent": [],
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
            self.paper_store.save(state)
            return state
        package = load_research_package(self.research_store, research_run_id, package_version)
        state = create_paper_state(paper_id, research_run_id, package)
        self.paper_store.save(state)
        return state

    def resume(self, paper_id, max_nodes=10):
        state = self.paper_store.load(paper_id)
        if not state:
            raise SystemExit(f"paper {paper_id} does not exist")
        if state.get("status", "").startswith("BLOCKED") or state.get("status") == "PAPER_COMPLETE":
            return state
        package = load_research_package(self.research_store, state["research_run_id"], state["research_package_ref"]["version"])
        if not package or package["package_hash"] != state["research_package_ref"]["package_hash"]:
            state["status"] = "BLOCKED_RESEARCH_NOT_READY"
            state["failure_reason"] = "ResearchPackage missing or hash mismatch"
            self.paper_store.save(state)
            return state
        executed = 0
        while executed < max_nodes:
            nodes = ready_paper_nodes(state)
            if not nodes:
                break
            node = nodes[0]
            self._execute_node(state, node, package)
            executed += 1
            self.paper_store.save(state)
            if state.get("status") == "WAITING_FOR_HUMAN":
                break
        if all(n.get("status") == "COMPLETED" for n in state.get("dag", {}).get("nodes", {}).values()):
            state["status"] = "PAPER_COMPLETE"
            self._write_paper_package(state, package)
            self.paper_store.save(state)
        return state

    def _execute_node(self, state, node, package):
        if node["node_id"] == "package_validation":
            node["status"] = "COMPLETED"
        elif node["node_id"] == "paper_blueprint":
            state["blueprint"] = {"sections": ["abstract", "introduction", "methods", "results", "limitations", "conclusion"], "style": "generic_academic"}
            node["status"] = "COMPLETED"
        elif node["node_id"] == "section_drafting":
            state["draft"] = {"source_package": state["research_package_ref"], "sections": {s: "" for s in state["blueprint"]["sections"]}}
            node["status"] = "COMPLETED"
        elif node["node_id"] == "claim_evidence_lint":
            claims = package.get("claim_evidence_ledger", {}).get("claims", [])
            invalid = [c.get("claim_id") for c in claims if c.get("status") in ("UNVERIFIED", "CONTRADICTED")]
            if invalid:
                state["status"] = "BLOCKED_RESEARCH_NOT_READY"
                state["failure_reason"] = f"paper cannot use unverified/contradicted claims: {invalid}"
            else:
                node["status"] = "COMPLETED"
        elif node["node_id"] == "paper_package":
            node["status"] = "COMPLETED"
        state["updated_at"] = now_iso()

    def _write_paper_package(self, state, package):
        paper_package = {
            "schema_version": 1,
            "paper_package_id": f"PP-{state['paper_id']}",
            "paper_id": state["paper_id"],
            "created_at": now_iso(),
            "research_package_ref": state["research_package_ref"],
            "draft": state.get("draft", {}),
            "build_artifacts": [],
            "research_revision_requests": state.get("research_revision_requests", []),
        }
        paper_package["package_hash"] = stable_hash(paper_package)
        path = self.paper_store.paper_root(state["paper_id"]) / "paper_package.json"
        path.write_text(json.dumps(paper_package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state["paper_package"] = {"path": str(path), "package_hash": paper_package["package_hash"]}

    def decisions(self, paper_id):
        return list_open_decisions(self.paper_store.load(paper_id) or {})

    def decide(self, paper_id, decision_id, option=None, free_text=None):
        state = self.paper_store.load(paper_id)
        submit_decision(state, decision_id, option, free_text)
        self.paper_store.save(state)
        return state


def verify_paper(paper_store, paper_id):
    state = paper_store.load(paper_id)
    if not state:
        return {"status": "FAIL", "reasons": ["missing paper state"]}
    if state.get("status") == "BLOCKED_RESEARCH_NOT_READY":
        reason = state.get("failure_reason")
        return {"status": "WAITING", "reasons": reason if isinstance(reason, list) else [reason]}
    package_path = paper_store.paper_root(paper_id) / "paper_package.json"
    if state.get("status") != "PAPER_COMPLETE" or not package_path.exists():
        return {"status": "WAITING", "reasons": [state.get("status")]}
    package = json.loads(package_path.read_text(encoding="utf-8"))
    payload = dict(package)
    expected = payload.pop("package_hash", None)
    if stable_hash(payload) != expected:
        return {"status": "FAIL", "reasons": ["paper package hash mismatch"]}
    return {"status": "PASS", "paper_package": {"paper_id": paper_id, "hash": expected, "research_package_ref": package["research_package_ref"]}}
