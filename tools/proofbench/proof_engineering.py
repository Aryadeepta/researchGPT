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


# The objects below deliberately describe *recovery work*, not proof evidence.
# They are small enough to be emitted even when no SAT implementation is present.
class ProofTerminal(str, Enum):
    VERIFIED = "VERIFIED"
    ACTIONABLE_HANDOFF = "ACTIONABLE_HANDOFF"
    PROOF_RECOVERY_EXHAUSTED = "PROOF_RECOVERY_EXHAUSTED"


@dataclass
class ConstraintProvenance:
    source: str
    method: str
    inputs: list[str] = field(default_factory=list)
    verification_status: str = "SCOUTED_UNTRUSTED"


@dataclass
class BooleanConstraint:
    kind: str  # clause, at_most, at_least, implication, contradiction
    literals: list[int] = field(default_factory=list)
    bound: int | None = None
    constraint_id: str = ""
    provenance: ConstraintProvenance = field(default_factory=lambda: ConstraintProvenance("unknown", "unknown"))


@dataclass
class FiniteBooleanModel:
    source_obligation_hash: str
    variables: dict[str, str]  # stable id -> human meaning
    constraints: list[BooleanConstraint]
    provenance: list[ConstraintProvenance] = field(default_factory=list)
    encoding_version: str = "finite-boolean-model/v1"

    def canonical(self) -> dict[str, Any]:
        return {"source_obligation_hash": self.source_obligation_hash, "variables": dict(sorted(self.variables.items())),
                "constraints": [asdict(c) for c in self.constraints], "encoding_version": self.encoding_version}
    def digest(self) -> str: return sha256_bytes(json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode())


def _lit(var: str, value: bool, ids: dict[str, int]) -> int:
    return ids[var] if value else -ids[var]


def propagate_constraints(model: FiniteBooleanModel) -> tuple[dict[str, bool], list[BooleanConstraint], list[dict[str, Any]]]:
    """Bounded unit/cardinality propagation. Results remain scouting facts."""
    ids = {v: i + 1 for i, v in enumerate(model.variables)}; names = {i: v for v, i in ids.items()}
    values: dict[str, bool] = {}; trace: list[dict[str, Any]] = []; changed = True
    while changed:
        changed = False
        for c in model.constraints:
            if c.kind == "clause":
                live=[]; satisfied=False
                for literal in c.literals:
                    val=values.get(names[abs(literal)])
                    if val is None: live.append(literal)
                    elif val == (literal > 0): satisfied=True; break
                if satisfied: continue
                if not live: trace.append({"constraint_id":c.constraint_id,"kind":"contradiction"}); continue
                if len(live)==1:
                    var, val=names[abs(live[0])], live[0]>0
                    if var not in values:
                        values[var]=val; changed=True; trace.append({"constraint_id":c.constraint_id,"variable":var,"value":val,"method":"unit-propagation"})
            elif c.kind in {"at_most", "at_least"} and c.bound is not None:
                true_count=sum(values.get(names[abs(x)]) == (x>0) for x in c.literals)
                unknown=[x for x in c.literals if names[abs(x)] not in values]
                if c.kind=="at_most" and true_count==c.bound:
                    for x in unknown:
                        var=names[abs(x)]; values[var]=not (x>0); changed=True; trace.append({"constraint_id":c.constraint_id,"variable":var,"value":values[var],"method":"cardinality-propagation"})
                if c.kind=="at_least" and true_count+len(unknown)==c.bound:
                    for x in unknown:
                        var=names[abs(x)]; values[var]=x>0; changed=True; trace.append({"constraint_id":c.constraint_id,"variable":var,"value":values[var],"method":"cardinality-propagation"})
    residual=[]
    for c in model.constraints:
        if c.kind != "clause": residual.append(c); continue
        if not any(values.get(names[abs(x)]) == (x>0) for x in c.literals):
            residual.append(BooleanConstraint("clause", [x for x in c.literals if names[abs(x)] not in values], c.bound, c.constraint_id, c.provenance))
    return values, residual, trace


def dimacs_export(model: FiniteBooleanModel, directory: str | Path) -> dict[str, Any]:
    """Export a deterministic CNF with a polynomial Sinz sequential counter.

    CNF is an advisory solver input only.  In particular this routine never
    creates a ProofDag node or changes a verifier status.
    """
    root=Path(directory); root.mkdir(parents=True, exist_ok=True)
    ids={v:i+1 for i,v in enumerate(model.variables)}; meanings=dict(model.variables); clauses=[]
    def fresh(label: str) -> int:
        i=len(ids)+1; name=f"aux_card_{label}_{i}"; ids[name]=i; meanings[name]="sequential-counter auxiliary"; return i
    def at_most(lits: list[int], bound: int, label: str) -> None:
        # Sinz 2005.  Literal polarity is retained, so this also works for a
        # cardinality over signed literals.
        m=len(lits)
        if bound < 0: clauses.append([]); return
        if bound >= m: return
        if bound == 0: clauses.extend([[-x] for x in lits]); return
        s={(i,j):fresh(f"{label}_{i}_{j}") for i in range(1,m) for j in range(1,bound+1)}
        clauses.append([-lits[0], s[(1,1)]])
        for i in range(2,m):
            clauses.append([-lits[i-1], s[(i,1)]])
            clauses.append([-s[(i-1,1)], s[(i,1)]])
            for j in range(2,bound+1):
                clauses.append([-lits[i-1], -s[(i-1,j-1)], s[(i,j)]])
                clauses.append([-s[(i-1,j)], s[(i,j)]])
            # This is the essential sequential-counter overflow guard.  It
            # applies at *every* intermediate input, not just the last one:
            # once the first i-1 inputs already contain k truths, x_i cannot
            # be true.  Omitting it admits an overflow which is then hidden
            # by unconstrained later auxiliary variables.
            clauses.append([-lits[i-1], -s[(i-1,bound)]])
        clauses.append([-lits[-1], -s[(m-1,bound)]])
    for c in model.constraints:
        if c.kind=="clause": clauses.append(list(c.literals))
        elif c.kind=="at_most" and c.bound is not None: at_most(c.literals, c.bound, c.constraint_id or "atmost")
        elif c.kind=="at_least" and c.bound is not None: at_most([-x for x in c.literals], len(c.literals)-c.bound, c.constraint_id or "atleast")
    cnf=root/"residual.cnf"; cnf.write_text("c finite-boolean-model/v2\n"+f"p cnf {len(ids)} {len(clauses)}\n"+"".join(" ".join(map(str,c))+" 0\n" for c in clauses))
    mapping=root/"variable_map.json"; mapping.write_text(json.dumps({str(i):{"id":v,"meaning":meanings[v]} for v,i in ids.items()}, indent=2, sort_keys=True)+"\n")
    manifest={"obligation_hash":model.source_obligation_hash,"cnf_sha256":sha256_file(cnf),"variable_map_sha256":sha256_file(mapping),"variable_count":len(ids),"clause_count":len(clauses),"encoding_version":"finite-boolean-model/v2","cardinality_encoding":"sinz-sequential-counter","producer":"proof_engineering","solver_status":"UNRUN"}
    (root/"sat_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); return {**manifest,"cnf_path":str(cnf),"variable_map_path":str(mapping)}


def additive_basis_model(n: int, forbidden: set[int], max_selected: int, obligation_hash: str) -> FiniteBooleanModel:
    """Generic finite sum-coverage encoding; no theorem or benchmark identifiers."""
    # n is inclusive: x_n and coverage-n are part of the original obligation.
    variables={f"x_{i}": f"select element {i}" for i in range(n + 1)}; cs=[]
    def add(kind,lits,bound,cid,method): cs.append(BooleanConstraint(kind,lits,bound,cid,ConstraintProvenance("finite-checker",method,[cid],"SCOUTED_UNTRUSTED")))
    for i in sorted(forbidden): add("clause",[-(i+1)],None,f"forbidden-{i}","problem-metadata")
    aux=0
    for target in range(n + 1):
        pairs=[]
        for i in range(target+1):
            j=target-i
            if i>j or j>n: continue
            # Same-variable pair is a literal; distinct pairs use a standard
            # Tseitin auxiliary that is explicit in the portable model.
            if i==j: pairs.append(i+1)
            else:
                # Standard Tseitin representation of xi AND xj.  These names
                # are stable and make both DIMACS and a human handoff precise.
                name=f"pair_{target}_{i}_{j}"; variables[name]=f"representation {target} = {i} + {j}"; lit=len(variables)
                # Variables are inserted in numeric order, so lit is its DIMACS id.
                add("clause",[-lit,i+1],None,f"pair-left-{target}-{i}-{j}","tseitin-and")
                add("clause",[-lit,j+1],None,f"pair-right-{target}-{i}-{j}","tseitin-and")
                add("clause",[lit,-(i+1),-(j+1)],None,f"pair-back-{target}-{i}-{j}","tseitin-and")
                pairs.append(lit)
        add("clause",sorted(set(pairs)),None,f"coverage-{target}","finite-pair-enumeration")
    add("at_most",list(range(1,n+2)),max_selected,"cardinality", "problem-metadata")
    return FiniteBooleanModel(obligation_hash,variables,cs)


def readable_deduction_trace(model: FiniteBooleanModel, facts: dict[str, bool], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render constraint-linked deductions without benchmark-specific prose."""
    out=list(trace)
    names=list(model.variables)
    for c in model.constraints:
        if not c.constraint_id.startswith("coverage-") or c.kind != "clause": continue
        alternatives=[]; covered=False
        for lit in c.literals:
            name=names[abs(lit)-1]
            if name.startswith("pair_"):
                _, target, left, right=name.split("_")
                lv,rv=facts.get(f"x_{left}"),facts.get(f"x_{right}")
                if lv is True and rv is True: covered=True; break
                if lv is False or rv is False: continue
                option=[x for x,v in ((f"x_{left}",lv),(f"x_{right}",rv)) if v is not True]
                alternatives.append(sorted(option))
            elif facts.get(name) is True: covered=True; break
            elif facts.get(name) is not False: alternatives.append([name])
        if not covered and alternatives:
            unique=sorted({tuple(x) for x in alternatives})
            structured=[list(x) for x in unique]
            out.append({"constraint_id":c.constraint_id,"kind":"coverage-alternatives","alternatives":structured,
                        # Compatibility convenience only; semantic consumers
                        # must use alternatives so conjunctions are retained.
                        "variables":sorted({v for option in structured for v in option}),
                        "source":"finite model simplification","verification_status":"SCOUTED_UNTRUSTED"})
    return out


def residual_model(model: FiniteBooleanModel, facts: dict[str, bool]) -> FiniteBooleanModel:
    """Substitute propagated values and discard satisfied constraints safely."""
    old_names=list(model.variables); old_id={name:i+1 for i,name in enumerate(old_names)}
    kept: list[BooleanConstraint]=[]; live_names:set[str]=set()
    for c in model.constraints:
        if c.kind == "clause":
            live=[]; satisfied=False
            for lit in c.literals:
                name=old_names[abs(lit)-1]; value=facts.get(name)
                if value is None: live.append(lit)
                elif value == (lit > 0): satisfied=True; break
            if satisfied: continue
            kept.append(BooleanConstraint("clause", live, None, c.constraint_id, c.provenance))
            live_names.update(old_names[abs(x)-1] for x in live)
        elif c.kind == "at_most" and c.bound is not None:
            true_count=sum(facts.get(old_names[abs(x)-1]) == (x > 0) for x in c.literals)
            live=[x for x in c.literals if old_names[abs(x)-1] not in facts]
            kept.append(BooleanConstraint("at_most", live, c.bound-true_count, c.constraint_id, c.provenance)); live_names.update(old_names[abs(x)-1] for x in live)
        else: kept.append(c)
    # Re-number literals after eliminating irrelevant variables.
    ordered=[x for x in old_names if x in live_names]; new_id={x:i+1 for i,x in enumerate(ordered)}
    for c in kept:
        c.literals=[(1 if x > 0 else -1)*new_id[old_names[abs(x)-1]] for x in c.literals]
    return FiniteBooleanModel(model.source_obligation_hash, {x:model.variables[x] for x in ordered}, kept, model.provenance, model.encoding_version)


@dataclass
class BottleneckObligation:
    statement: str; formal_statement: str; why_it_blocks_parent: str; residual_variables: list[str]; estimated_complexity: int; failed_approaches: list[str]; candidate_approaches: list[str]; relevant_artifacts: list[str]


def create_handoff_bundle(directory: str | Path, *, obligation_id: str, obligation_hash: str, goal: str, classification: str, model: FiniteBooleanModel | None, verified_prefix: list[str], dag_state: Any, attempts: list[Any], diagnostics: str, capabilities: list[Capability]) -> dict[str, Any]:
    root=Path(directory); root.mkdir(parents=True,exist_ok=True); facts={}; trace=[]; sat={}
    if model:
        facts,residual,trace=propagate_constraints(model); trace=readable_deduction_trace(model,facts,trace)
        reduced=residual_model(model, facts); sat=dimacs_export(reduced,root)
        (root/"finite_boolean_model.json").write_text(json.dumps(model.canonical(),indent=2,sort_keys=True)+"\n")
        (root/"residual_boolean_model.json").write_text(json.dumps(reduced.canonical(),indent=2,sort_keys=True)+"\n")
        (root/"deduction_trace.json").write_text(json.dumps(trace,indent=2,sort_keys=True)+"\n")
    remaining=sorted(reduced.variables) if model else []
    bottleneck=BottleneckObligation("Show the residual Boolean model is UNSAT under its stated constraints." if model else "Prove the remaining Lean goal.",goal,"Direct bounded proof recovery was exhausted.",remaining,2**len(remaining),[getattr(a,"strategy",str(a)) for a in attempts], ["run a SAT solver and retain a checkable proof certificate","supply a proved Lean lemma"], [sat.get("cnf_path","")]).__dict__
    (root/"bottleneck.json").write_text(json.dumps(bottleneck,indent=2,sort_keys=True)+"\n")
    available=sorted({c.implementation_kind.split("-")[0] for c in capabilities}); tools={"available_now":available,"useful_missing":["proof-producing SAT solver / certificate checker"] if not any(x in available for x in ("cadical","kissat","minisat")) else [],"suggested_next_action":"Run a SAT solver on residual.cnf and preserve a proof certificate; import a Lean lemma or checked certificate through the resume manifest."}
    (root/"suggested_tools.json").write_text(json.dumps(tools,indent=2,sort_keys=True)+"\n")
    advisory="# Codex advisory (untrusted)\n\nExact bottleneck: `"+bottleneck["statement"]+"`\n\nLean goal:\n```lean\n"+goal+"\n```\n\nAny proposal must enter the normal verifier pipeline; it is not evidence.\nArtifacts: residual.cnf, variable_map.json, deduction_trace.json, bottleneck.json.\n"
    (root/"codex_advisory.md").write_text(advisory)
    hashes={p.name:sha256_file(p) for p in root.iterdir() if p.is_file()}
    bundle={"terminal_state":ProofTerminal.ACTIONABLE_HANDOFF.value,"obligation_id":obligation_id,"obligation_hash":obligation_hash,"formal_goal":goal,"classification":classification,"verifier_trust_requirement":"Lean or an explicitly checked certificate","verified_partial_proof_prefix":verified_prefix,"proof_dag_state":dag_state,"strategies_attempted":[getattr(a,"strategy",str(a)) for a in attempts],"failure_diagnostics":diagnostics,"derived_facts":facts,"deduction_trace":trace,"estimated_residual_search_size":2**len(remaining),"bottleneck":bottleneck,"solver_artifacts":sat,"original_model_hash":model.digest() if model else "","residual_model_hash":reduced.digest() if model else "","tools":tools,"resume":{"accepted_inputs":["proved Lean lemma","solver certificate","corrected formalization","approved engineering action","external reasoning note"],"checkpoint":"handoff_bundle.json","validation_required":True},"hashes":hashes,"reason":"DIRECT_PROOF_PATHS_EXHAUSTED_ACTIONABLE_MODEL_AVAILABLE"}
    (root/"handoff_bundle.json").write_text(json.dumps(bundle,indent=2,sort_keys=True)+"\n")
    (root/"handoff.md").write_text(f"# Actionable proof handoff\n\nTerminal: ACTIONABLE_HANDOFF\n\n{bottleneck['statement']}\n\nResume from `handoff_bundle.json`; all proposed input requires independent validation.\n")
    return bundle
