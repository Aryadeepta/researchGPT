"""Bounded, verifier-gated proof engineering primitives.

This is deliberately independent from a language model.  A model can propose a
``ProofRecoveryPlan`` but this module validates and executes one semantic tool
call at a time.  Computation is scouting evidence only; a claim is promoted
only after its declared checker reports success.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class ObligationClass(str, Enum):
    SYMBOLIC_LOCAL = "SYMBOLIC_LOCAL"
    FINITE_DECIDABLE = "FINITE_DECIDABLE"
    FINITE_COMBINATORIAL = "FINITE_COMBINATORIAL"
    SAT_LIKE = "SAT_LIKE"
    SMT_LIKE = "SMT_LIKE"
    ARITHMETIC = "ARITHMETIC"
    COMPUTATIONAL_IDENTITY = "COMPUTATIONAL_IDENTITY"
    ENUMERATIVE_LOWER_BOUND = "ENUMERATIVE_LOWER_BOUND"
    LIBRARY_PREMISE_MISSING = "LIBRARY_PREMISE_MISSING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Capability:
    stable_id: str
    implementation_kind: str
    inputs: list[str]
    outputs: list[str]
    permissions: dict[str, str]
    resource_bounds: dict[str, int]
    side_effects: str = "none"
    verifier_requirements: list[str] = field(default_factory=list)


def available_capabilities(workspace: str | Path, lean: str | None = None) -> list[Capability]:
    """Return only capabilities backed by this local machine.

    The manifest is data sent to a planner; it is not an authority grant.
    """
    lean_bin = lean if lean and Path(lean).exists() else shutil.which("lean")
    base = [
        Capability("artifact_read", "supervisor", ["path"], ["bytes"], {"filesystem": "workspace"}, {"max_bytes": 1_000_000}),
        Capability("artifact_write", "supervisor", ["relative_path", "bytes"], ["artifact_hash"], {"filesystem": "workspace"}, {"max_bytes": 1_000_000}, "workspace-write"),
        Capability("python_exec", "argv", ["argv"], ["stdout", "stderr", "artifacts"], {"filesystem": "workspace", "network": "none"}, {"timeout_seconds": 30, "max_output_bytes": 65536}, "bounded-exec"),
        Capability("shell_exec", "argv", ["argv"], ["stdout", "stderr", "artifacts"], {"filesystem": "workspace", "network": "none"}, {"timeout_seconds": 30, "max_output_bytes": 65536}, "bounded-exec"),
        Capability("finite_enumeration", "python", ["generator"], ["certificate"], {"filesystem": "workspace", "network": "none"}, {"timeout_seconds": 30, "max_cases": 100000}, "workspace-write", ["explicit checker"]),
        Capability("generate_lean_source", "supervisor", ["source"], ["artifact_hash"], {"filesystem": "workspace"}, {"max_bytes": 2_000_000}, "workspace-write", ["lean_check"]),
        Capability("split_obligation", "supervisor", ["case_count", "chunk_size"], ["partition"], {"filesystem": "none"}, {"max_chunks": 256}),
        Capability("generate_certificate", "supervisor", ["claim", "payload"], ["certificate"], {"filesystem": "workspace"}, {"max_bytes": 2_000_000}, "workspace-write", ["verify_certificate"]),
        Capability("verify_certificate", "checker", ["certificate"], ["verification"], {"filesystem": "workspace"}, {"timeout_seconds": 60}),
    ]
    if lean_bin:
        base.extend([
            Capability("lean_check", "argv", ["source"], ["diagnostics"], {"filesystem": "workspace", "network": "none"}, {"timeout_seconds": 90, "max_output_bytes": 65536}, "bounded-exec", ["Lean exit 0"]),
            Capability("lean_compile", "argv", ["source"], ["diagnostics"], {"filesystem": "workspace", "network": "none"}, {"timeout_seconds": 90, "max_output_bytes": 65536}, "bounded-exec", ["Lean exit 0"]),
            Capability("inspect_lean_goal", "lean-probe", ["goal"], ["goal_state"], {"filesystem": "workspace"}, {"timeout_seconds": 30}),
        ])
    for solver, cid in (("z3", "smt_solve"), ("cvc5", "smt_solve"), ("cadical", "sat_solve"), ("kissat", "sat_solve"), ("minisat", "sat_solve")):
        if shutil.which(solver):
            base.append(Capability(cid, f"{solver}-argv", ["encoding"], ["solver_result"], {"filesystem": "workspace", "network": "none"}, {"timeout_seconds": 30}, "bounded-exec", ["certificate checker for proof promotion"]))
    return base


def capability_context(capabilities: list[Capability]) -> dict[str, Any]:
    return {"available_capabilities": [asdict(c) for c in capabilities],
            "trust_boundary": "Model and tool output are untrusted until the declared verifier accepts an artifact."}


@dataclass
class ObligationAssessment:
    obligation_class: str
    estimated_size: int | None
    reasoning: str
    source: str
    selected_recovery_strategy: str


def classify_obligation(goal: str, *, metadata: dict[str, Any] | None = None, diagnostics: str = "") -> ObligationAssessment:
    metadata = metadata or {}; text = (goal + " " + diagnostics).lower()
    size = metadata.get("candidate_count") or metadata.get("estimated_size")
    if metadata.get("enumerative_lower_bound"):
        cls, why, strategy = ObligationClass.ENUMERATIVE_LOWER_BOUND, "explicit finite lower-bound metadata", "finite-partition"
    elif size is not None or any(x in text for x in ("fintype", "finset", "list.all", "decidable")):
        cls = ObligationClass.FINITE_COMBINATORIAL if (size or 0) > metadata.get("small_finite_limit", 256) else ObligationClass.FINITE_DECIDABLE
        why, strategy = "finite cardinality/decidability signal", "generated-lean" if cls == ObligationClass.FINITE_DECIDABLE else "finite-partition"
    elif any(x in text for x in ("unsat", "boolean", "cnf")):
        cls, why, strategy = ObligationClass.SAT_LIKE, "Boolean/UNSAT signal", "sat-certificate"
    elif any(x in text for x in ("linarith", "int", "nat", "≤", "<", "+")):
        cls, why, strategy = ObligationClass.ARITHMETIC, "arithmetic syntax", "arithmetic-certificate"
    elif "unknown constant" in text or "failed to synthesize" in text:
        cls, why, strategy = ObligationClass.LIBRARY_PREMISE_MISSING, "Lean library diagnostic", "library-inspection"
    else:
        cls, why, strategy = ObligationClass.SYMBOLIC_LOCAL, "local symbolic goal", "typed-context-closure"
    return ObligationAssessment(cls.value, int(size) if size is not None else None, why, "goal-shape+metadata+diagnostics", strategy)


@dataclass
class ProofRecoveryPlan:
    obligation_id: str
    hypothesis: str
    action: str
    capability: str
    expected_artifact: str
    verifier: str
    cost_bound: dict[str, int]
    success_condition: str


@dataclass
class ProofRecoveryObservation:
    exit_status: int | None
    artifact: str | None
    diagnostics: str
    verifier_status: str
    measured_progress: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    capability: str
    argv: list[str]
    call_sha256: str
    exit_status: int
    stdout_path: str
    stderr_path: str
    stdout_sha256: str
    stderr_sha256: str
    produced_artifacts: dict[str, str]
    elapsed_seconds: float


class BoundedToolSupervisor:
    """Workspace-only argv executor with persisted, normalized observations."""
    forbidden = {"rm", "sudo", "env", "printenv", "curl", "wget", "git", "sh", "bash"}
    def __init__(self, workspace: str | Path, artifact_dir: str | Path, capabilities: list[Capability] | None = None):
        self.workspace = Path(workspace).resolve(); self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(artifact_dir).resolve(); self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.capabilities = {c.stable_id: c for c in (capabilities or available_capabilities(self.workspace))}
        self.calls: list[ToolCallRecord] = []; self._active = False
    def _validate(self, capability: str, argv: list[str]) -> None:
        if self._active: raise RuntimeError("ONE_SEMANTIC_TOOL_CALL_AT_A_TIME")
        if capability not in self.capabilities: raise ValueError("CAPABILITY_UNAVAILABLE")
        if not argv or not all(isinstance(x, str) and x for x in argv): raise ValueError("INVALID_ARGV")
        if argv[0] in self.forbidden or "/" in argv[0] and not Path(argv[0]).exists(): raise ValueError("COMMAND_REJECTED")
        if capability == "python_exec" and argv[0] != "python3": raise ValueError("PYTHON3_REQUIRED")
        if any(x in {"/", ".."} or x.startswith("/") and not str(self.workspace) in x for x in argv[1:]): raise ValueError("WORKSPACE_SCOPE_REJECTED")
    def execute(self, capability: str, argv: list[str], *, timeout_seconds: int | None = None, produced_paths: list[str] | None = None) -> ToolCallRecord:
        self._validate(capability, argv); self._active = True
        try:
            bound = self.capabilities[capability].resource_bounds
            timeout = min(timeout_seconds or bound.get("timeout_seconds", 30), bound.get("timeout_seconds", 30))
            started = time.monotonic()
            try: cp = subprocess.run(argv, cwd=self.workspace, text=True, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired as ex: cp = subprocess.CompletedProcess(argv, 124, ex.stdout or "", (ex.stderr or "") + "\nTIMEOUT")
            stamp = f"tool-{len(self.calls):04d}"; stdout = str(cp.stdout or "")[:bound.get("max_output_bytes", 65536)]; stderr = str(cp.stderr or "")[:bound.get("max_output_bytes", 65536)]
            op, ep = self.artifact_dir / f"{stamp}.stdout", self.artifact_dir / f"{stamp}.stderr"; op.write_text(stdout); ep.write_text(stderr)
            artifacts = {}
            for raw in produced_paths or []:
                path = (self.workspace / raw).resolve()
                if self.workspace not in path.parents or not path.is_file(): continue
                artifacts[str(path.relative_to(self.workspace))] = sha256_file(path)
            record = ToolCallRecord(capability, argv, sha256_bytes(json.dumps([capability, argv], separators=(",", ":")).encode()), cp.returncode, str(op), str(ep), sha256_file(op), sha256_file(ep), artifacts, time.monotonic()-started)
            self.calls.append(record)
            with (self.artifact_dir / "tool_calls.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(asdict(record), sort_keys=True)+"\n")
            return record
        finally: self._active = False


@dataclass
class ProofCertificate:
    format: str
    producer: str
    claim_hash: str
    payload_path: str
    payload_hash: str
    checker_id: str
    checker_version: str
    verification_result: str = "UNVERIFIED"


@dataclass
class ProofDagNode:
    node_id: str
    proposition: str
    fingerprint: str
    dependencies: list[str] = field(default_factory=list)
    partition_metadata: dict[str, Any] = field(default_factory=dict)
    producer: str = ""
    verifier: str = ""
    artifact_hash: str = ""
    status: str = "UNVERIFIED"
    resource_use: dict[str, Any] = field(default_factory=dict)


class ProofDag:
    def __init__(self): self.nodes: dict[str, ProofDagNode] = {}
    def add(self, node: ProofDagNode) -> ProofDagNode:
        if node.node_id in self.nodes or node.node_id in node.dependencies or any(d not in self.nodes for d in node.dependencies): raise ValueError("INVALID_DAG_DEPENDENCY")
        self.nodes[node.node_id] = node; return node
    def verify(self, node_id: str, *, verifier: str, artifact_hash: str) -> bool:
        node = self.nodes[node_id]
        if any(self.nodes[d].status != "VERIFIED" for d in node.dependencies): return False
        node.status, node.verifier, node.artifact_hash = "VERIFIED", verifier, artifact_hash; return True


def deterministic_partition(case_count: int, chunk_size: int, *, max_chunks: int = 256) -> list[dict[str, int]]:
    if case_count < 0 or chunk_size <= 0: raise ValueError("INVALID_PARTITION")
    parts = [{"index": i, "start": start, "end": min(case_count, start + chunk_size)} for i, start in enumerate(range(0, case_count, chunk_size))]
    if len(parts) > max_chunks: raise ValueError("PARTITION_BUDGET_EXCEEDED")
    if sum(x["end"]-x["start"] for x in parts) != case_count or any(parts[i]["end"] != parts[i+1]["start"] for i in range(len(parts)-1)): raise AssertionError("PARTITION_INCOMPLETE")
    return parts


def split_partition(part: dict[str, int], *, min_chunk_size: int = 1) -> list[dict[str, int]]:
    if part["end"] - part["start"] <= min_chunk_size: return []
    mid = (part["start"] + part["end"]) // 2
    return [{**part, "end": mid, "split": True}, {**part, "start": mid, "split": True}]


RECOVERY_LADDER = ("direct-lean", "typed-context", "blueprint-decomposition", "computational-scout", "generated-formal-artifact", "solver-certificate", "local-specialist")


def recovery_stages(assessment: ObligationAssessment, capabilities: list[Capability], previous_failures: list[str] | None = None) -> list[dict[str, str]]:
    have = {c.stable_id for c in capabilities}; failed = set(previous_failures or []); result=[]
    for stage in RECOVERY_LADDER:
        needed = {"computational-scout": "python_exec", "generated-formal-artifact": "generate_lean_source", "solver-certificate": "verify_certificate"}.get(stage)
        applicable = (not needed or needed in have) and stage not in failed
        if assessment.obligation_class == ObligationClass.SYMBOLIC_LOCAL.value and stage in {"computational-scout", "solver-certificate"}: applicable = False
        result.append({"stage": stage, "status": "APPLICABLE" if applicable else "SKIPPED", "reason": "capability/class/budget"})
    return result
