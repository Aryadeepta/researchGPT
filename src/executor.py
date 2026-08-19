import csv
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

from src.research_state import block_node, complete_node
from src.research_state import now_iso


EXECUTABLE_SUFFIXES = {".py", ".sh", ".bash", ".c", ".cpp", ".rs", ".go", ".java", ".jl", ".R", ".sv", ".v"}
PROSE_SUFFIXES = {".md", ".txt", ".tex", ".rst"}


def is_executable_artifact(path):
    return Path(path).suffix in EXECUTABLE_SUFFIXES


def contract_has_raw_artifacts(contract):
    return bool(contract.get("raw_outputs") or contract.get("metrics") or contract.get("logs"))


def prose_only_satisfies_contract(contract, artifacts):
    if not contract.get("requires_execution"):
        return False
    if not artifacts:
        return True
    return all(Path(a).suffix in PROSE_SUFFIXES for a in artifacts)


class ResearchExecutor:
    def __init__(self, store, work_root=None):
        self.store = store
        self.work_root = Path(work_root or os.environ.get("RESEARCH_WORK_ROOT", "/tmp/researchGPT-worker"))
        self.work_root.mkdir(parents=True, exist_ok=True)

    def execute_node(self, state, node, producer="research_executor"):
        from src.research_runtime import GenericResearchRuntime

        if GenericResearchRuntime(self.store, self.work_root).execute(state, node):
            return state
        contract = node.get("contract", {})
        outputs = list(contract.get("outputs", [])) + list(contract.get("raw_outputs", []))
        if prose_only_satisfies_contract(contract, outputs):
            block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", "executable contract cannot be satisfied by prose-only outputs")
            return state
        if contract.get("hard_coded_results"):
            block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "hard-coded expected results are not measurements")
            return state
        if node.get("kind") == "planning" and contract_has_raw_artifacts(contract):
            block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "planner nodes cannot register measured results")
            return state

        run_dir = self.work_root / state["run_id"] / node["node_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        produced = []

        command = contract.get("command")
        if command:
            started_at = now_iso()
            result = subprocess.run(command, cwd=run_dir, shell=True, capture_output=True, text=True, timeout=int(contract.get("timeout_seconds", 600)))
            ended_at = now_iso()
            log_path = run_dir / "execution.log"
            log_path.write_text(f"returncode={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}", encoding="utf-8")
            produced.append(self.store.put_artifact(state["run_id"], log_path, f"logs/{node['node_id']}.log", producer))
            state.setdefault("execution_records", []).append({
                "execution_id": f"X{uuid4().hex[:8]}",
                "node_id": node["node_id"],
                "command": command,
                "working_directory": str(run_dir),
                "inputs": contract.get("inputs", []),
                "outputs": list(contract.get("outputs", [])) + list(contract.get("raw_outputs", [])),
                "start_time": started_at,
                "end_time": ended_at,
                "exit_status": result.returncode,
                "stdout_artifact": f"logs/{node['node_id']}.log",
                "environment": {"python": os.sys.version.split()[0], "platform": os.name},
            })
            if result.returncode != 0:
                block_node(state, node["node_id"], "FAILED", f"command failed with exit code {result.returncode}")
                return state

        for rel in outputs:
            path = run_dir / rel
            if not path.exists():
                if contract.get("allow_placeholder_outputs") and not contract.get("requires_execution"):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"node_id": node["node_id"], "status": "planned"}, indent=2), encoding="utf-8")
                else:
                    block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", f"required artifact missing: {rel}")
                    return state
            if path.is_file() and path.stat().st_size == 0:
                block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", f"required artifact is empty: {rel}")
                return state
            produced.append(self.store.put_artifact(state["run_id"], path, rel, producer))

        state.setdefault("artifact_manifest", {"artifacts": []}).setdefault("artifacts", []).extend(produced)
        complete_node(state, node["node_id"], artifacts=produced)
        return state


def derive_statistics_from_csv(raw_csv, out_json):
    values = []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for value in row.values():
                try:
                    values.append(float(value))
                    break
                except (TypeError, ValueError):
                    continue
    if not values:
        raise ValueError("no numeric values found")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n": len(values), "mean": mean, "variance": variance}, f, indent=2)
