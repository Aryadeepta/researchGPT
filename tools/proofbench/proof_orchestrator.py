#!/usr/bin/env python3
"""Verified, bounded, public-only Lean tactic orchestration for ProofBench.

Models suggest one tactic at a time.  This module never regards a suggestion,
plan, or incomplete Lean probe as evidence: only ``proof_gym.validate`` is a
completed proof check.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, heapq
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.llm_gateway import LLMRequest
from tools.proofbench.proof_engine import FORBIDDEN
from tools.proofbench.proof_gym import GymCase, make_cases, resolve_lean, source, validate, lean_run
from tools.proofbench.v9_controller import v9_local_provider
from tools.proofbench.kimina_specialist import KiminaMicroProofSolver, extract_candidates, public_kimina_prompt
from tools.proofbench.proof_task import ProofTask
from tools.proofbench.proof_engineering import (
    ProofDag, ProofDagNode, available_capabilities, capability_context,
    ProofTerminal, additive_basis_model, create_handoff_bundle,
    classify_obligation, recovery_stages,
)

TACTIC_SCHEMA={"type":"object","properties":{"tactic":{"type":"string"}},"required":["tactic"],"additionalProperties":False}
MAX_TACTIC_CHARS=256
HIDDEN_MARKERS=("hidden", "oracle", "held-out", "heldout", "controller secret")

class ProbeOutcome(str, Enum):
    FINAL_PASS="FINAL_PASS"; VALID_NEW_GOAL_STATE="VALID_NEW_GOAL_STATE"; NO_PROGRESS_SAME_GOAL="NO_PROGRESS_SAME_GOAL"
    LEAN_SYNTAX_OR_TACTIC_FAILURE="LEAN_SYNTAX_OR_TACTIC_FAILURE"; INTEGRITY_FAILURE="INTEGRITY_FAILURE"; INFRASTRUCTURE_FAILURE="INFRASTRUCTURE_FAILURE"

def bounded(s, n=1600): return str(s or "")[-n:]
def normalize_goal(text):
    text=re.sub(r"^.*?\.lean:\d+:\d+:\s*(?:error|warning):\s*", "", str(text), flags=re.M)
    text=re.sub(r"\b\d+:\d+\b", "#:#", text)
    return re.sub(r"\s+", " ", text).strip()
def goal_fingerprint(text): return hashlib.sha256(normalize_goal(text).encode()).hexdigest()[:20]
def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as f: f.write(json.dumps(obj,sort_keys=True)+"\n")

LEAN_ERROR_HEADER = re.compile(r"\berror(?:\([^)]*\))?:")
PROBE_TERMINAL_SENTINEL = "Failed: `fail` tactic was invoked"

def probe_failure_has_only_terminal_sentinel(out):
    """True only when Lean failed solely because of our deliberate final `fail`.

    Search probes intentionally end in `trace_state; fail` so that an incomplete
    but otherwise valid tactic prefix exposes its authoritative goal state.
    Any other Lean error means the candidate prefix itself was invalid and its
    printed goal text is diagnostic context, never a search transition.
    """
    headers = [
        line
        for line in str(out).splitlines()
        if LEAN_ERROR_HEADER.search(line)
    ]
    return bool(headers) and all(
        PROBE_TERMINAL_SENTINEL in line
        for line in headers
    )

@dataclass
class GoalSnapshot:
    raw: str; normalized: str; goals: list[str]; fingerprint: str; diagnostic: str=""
@dataclass
class ProofPrefix:
    tactics: list[str]=field(default_factory=list)
    def text(self): return "\n".join("  "+x.replace("\n", "\n  ") for x in self.tactics)
@dataclass
class ProofState:
    prefix: ProofPrefix; goal: GoalSnapshot; depth: int=0
@dataclass
class TacticCandidate: text: str; source: str; detail: str=""
@dataclass(frozen=True)
class DeterministicStrategy:
    """A generic proposal source.  Lean, not this metadata, judges success."""
    strategy_id: str
    tactics: tuple[str, ...]
    priority: int
@dataclass
class TacticAttempt:
    tactic: str; source: str; outcome: str; diagnostic: str=""; before: str=""; after: str=""; strategy: str=""
@dataclass
class ProofCheckpoint:
    checkpoint_id: int; parent_id: int|None; prefix: ProofPrefix; goal: GoalSnapshot; reason: str
@dataclass
class ProofPlanStep: kind: str; text: str
@dataclass
class ProofPlan: strategy: str; steps: list[ProofPlanStep]; model: str=""
@dataclass
class SearchNode:
    node_id: int; parent_id: int|None; state: ProofState; produced_by: str=""; source: str="initial"; status: str="frontier"; diagnostic: str=""
@dataclass
class BlueprintNode:
    """A small, proof-local DAG node.  It is evidence, never an assumption."""
    node_id: str; proposition: str; fingerprint: str; dependencies: list[str]
    status: str; source_strategy: str; proof_prefix: list[str]=field(default_factory=list)
    verifier_status: str="UNVERIFIED"
    partition_metadata: dict=field(default_factory=dict)
    producer: str="lean-tactic"
    artifact_hash: str=""
    resource_use: dict=field(default_factory=dict)

class TypedContextClosure:
    """Bounded syntactic applications of actual local names.

    Types are deliberately not guessed from text: generated terms are merely
    proposals and each one must pass the Lean transition gate.  Enumerating
    short application trees still exposes the useful typed chains Lean can
    elaborate, without becoming a theorem synthesizer.
    """
    max_application_depth=3
    max_candidates=64
    name_pattern=re.compile(r"(?:^|\n)\s*([A-Za-z_][\w']*)\s*:")
    def expressions(self, state):
        names=[]
        for name in self.name_pattern.findall(state.goal.raw):
            if name not in {"warning", "error"} and name not in names: names.append(name)
        # Hypotheses are generally more productive heads than parameters; this
        # is a generic lexical ordering, not a statement/case heuristic.
        hypotheses=sorted((n for n in names[:20] if n.startswith("h")), reverse=True)
        names=hypotheses + sorted(n for n in names[:20] if not n.startswith("h"))
        atoms=[]
        for name in names:
            atoms.extend((name, f"{name}.1", f"{name}.2", f"{name}.mp", f"{name}.mpr", f"{name}.1.1", f"{name}.1.2", f"{name}.2.1", f"{name}.2.2"))
        terms={0: list(dict.fromkeys(atoms))}
        all_terms=list(terms[0]); seen=set(all_terms)
        for depth in range(1, self.max_application_depth + 1):
            previous=[]
            for lower in range(depth): previous.extend(terms[lower])
            generated=[]
            for fn in names:
                for arg in previous:
                    expr=f"{fn} ({arg})"
                    if expr not in seen:
                        seen.add(expr); generated.append(expr)
            # Stable, short-first ordering bounds the combinatorial frontier.
            terms[depth]=generated[:self.max_candidates]
            all_terms.extend(terms[depth])
        # First use the observed local type annotations to prioritize terms
        # whose codomain is the observed target.  This remains proposal-only:
        # Lean checks every expression below, including all parsing edge cases.
        annotations={}
        for name, typ in re.findall(r"(?:^|\n)\s*([A-Za-z_][\w']*)\s*:\s*([^\n]+)", state.goal.raw):
            annotations.setdefault(name, typ.strip())
        target=state.goal.raw.rsplit("⊢", 1)[-1].strip().splitlines()[0].strip()
        known=[]
        for name in names:
            typ=annotations.get(name, "")
            if typ: known.append((name, typ, 0))
            if " ∧ " in typ:
                left,right=typ.split(" ∧ ",1); known.extend(((f"{name}.1",left,0),(f"{name}.2",right,0)))
            if " ↔ " in typ:
                left,right=typ.split(" ↔ ",1); known.extend(((f"{name}.mp",f"{left} → {right}",0),(f"{name}.mpr",f"{right} → {left}",0)))
        typed=[]; used=set()
        for depth in range(self.max_application_depth+1):
            for expr, typ, expr_depth in list(known):
                if expr_depth==depth and typ==target and expr not in used:
                    typed.append((depth,expr)); used.add(expr)
            if depth==self.max_application_depth: break
            for fn, fn_type, _ in list(known):
                bits=[x.strip() for x in fn_type.split(" → ")]
                if len(bits)<2: continue
                inputs,output=bits[:-1],bits[-1]
                pools=[]
                for required in inputs:
                    pool=[(e,d) for e,t,d in known if t==required and d<=depth]
                    if not pool: pools=[]; break
                    pools.append(pool[:12])
                if not pools: continue
                import itertools
                for combo in itertools.product(*pools):
                    expr=fn+" "+" ".join(f"({e})" for e,_ in combo); expr_depth=1+max(d for _,d in combo)
                    if expr_depth<=self.max_application_depth and not any(e==expr for e,_,_ in known): known.append((expr,output,expr_depth))
        # Exact local facts/projections first, then shallow applications.  The
        # candidate cap applies to expressions, independent of Lean outcomes.
        ranked=list(typed)
        for depth in range(self.max_application_depth + 1):
            ranked.extend((depth, x) for x in terms[depth])
        for depth in range(self.max_application_depth + 1):
            ranked.extend((depth, expr) for expr in terms[depth])
        # Typed target terms precede bounded blind fallback terms.
        ordered=[]; seen_expr=set()
        for depth,expr in typed + [(d,e) for d,e in ranked if (d,e) not in typed]:
            if expr not in seen_expr: ordered.append((depth,expr)); seen_expr.add(expr)
        return ordered[:self.max_candidates]
    def candidates(self, state):
        exact=[TacticCandidate(f"exact {expr}", "deterministic", f"context.closure.depth-{depth}")
               for depth, expr in self.expressions(state)]
        # `apply` asks Lean to make the premise an explicit obligation.  The
        # resulting goal is represented by an UNVERIFIED blueprint node and
        # cannot support its parent until a later Lean-accepted transition
        # closes it; this is the intentionally small wishful-lemma mechanism.
        names=[]
        for name in self.name_pattern.findall(state.goal.raw):
            if name not in names and name not in {"warning", "error"}: names.append(name)
        apply=[TacticCandidate(f"apply {name}", "deterministic", "context.blueprint.apply") for name in names[:16]]
        return exact[:self.max_candidates-16] + apply

class ProofPlanner(Protocol):
    def plan(self, public_residual: dict) -> ProofPlan|None: ...

class NoopProofPlanner:
    calls=0
    def plan(self, public_residual): return None

def public_residual(case: GymCase, state: ProofState, rejected: list[TacticAttempt]) -> dict:
    value={"theorem":case.theorem,"declaration":case.declaration,"goal":state.goal.normalized,
           "prefix":bounded(state.prefix.text(),1200),"rejected":[{"tactic":a.tactic,"diagnostic":bounded(a.diagnostic,300)} for a in rejected[-3:]]}
    raw=json.dumps(value).lower()
    if any(marker in raw for marker in HIDDEN_MARKERS): raise ValueError("REMOTE_RESIDUAL_PRIVATE_MARKER")
    return value

class CodexProofPlanner:
    """Opt-in, disposable public capsule planner.  It returns text only, never edits a candidate."""
    def __init__(self, enabled=None, max_luna=1, max_terra=1):
        self.enabled=os.environ.get("PROOFBENCH_ORCH_ENABLE_REMOTE","0")=="1" if enabled is None else enabled
        self.remaining={"luna":max_luna,"terra":max_terra}; self.calls={"luna":0,"terra":0}
    def plan(self, residual):
        if not self.enabled: return None
        model=next((m for m in ("luna","terra") if self.remaining[m]>0),None)
        if not model: return None
        # Validate/scrub before material reaches the disposable process.
        payload=public_residual_from_dict(residual)
        codex=shutil.which("codex")
        if not codex: return None
        self.remaining[model]-=1; self.calls[model]+=1
        with tempfile.TemporaryDirectory(prefix="proofbench-orch-public-") as d:
            p=Path(d); (p/"PUBLIC_INPUT.json").write_text(json.dumps(payload,sort_keys=True))
            instruction=("Read only PUBLIC_INPUT.json. Write PLAN.json containing exactly a JSON object with strategy and "
              "steps [{kind:'tactic',text:'one Lean tactic'}]. Do not inspect parent paths or modify other files.")
            cp=subprocess.run(["/usr/bin/timeout","90s",codex,"exec","--ephemeral","--skip-git-repo-check","--ignore-user-config","--ignore-rules","--sandbox","read-only","--model",{"luna":"gpt-5.6-luna","terra":"gpt-5.6-terra"}[model]],cwd=p,input=instruction,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            plan=p/"PLAN.json"
            if cp.returncode or not plan.is_file(): return None
            try: data=json.loads(plan.read_text())
            except Exception: return None
        steps=[ProofPlanStep(str(x.get("kind","")),str(x.get("text",""))) for x in data.get("steps",[]) if isinstance(x,dict)]
        if not isinstance(data.get("strategy"),str) or not steps: return None
        return ProofPlan(data["strategy"][:500],steps[:8],model)

def public_residual_from_dict(residual):
    raw=json.dumps(residual).lower()
    if any(x in raw for x in HIDDEN_MARKERS): raise ValueError("REMOTE_RESIDUAL_PRIVATE_MARKER")
    return residual

class LocalNextTacticSolver:
    def __init__(self, provider=None): self.provider=provider or v9_local_provider()
    def propose(self, case, state, rejected):
        prompt=("Return exactly JSON {\"tactic\":\"...\"}; exactly ONE Lean 4 tactic, no proof/file.\n"
          "Skill card: rfl | assumption | constructor | intro h | exact h | rw [h] | simp [h].\n"
          f"Theorem: {case.declaration}\nCurrent Lean state:\n{bounded(state.goal.normalized,2400)}\n"
          f"Validated prefix:\n{bounded(state.prefix.text(),800)}\nRejected: {json.dumps([{ 't':a.tactic,'d':bounded(a.diagnostic,180)} for a in rejected[-3:]])}")
        reply=self.provider.generate_structured(LLMRequest(prompt=prompt,stage="proofbench_next_tactic",requested_model_class="CHEAP",task_class=""),schema=TACTIC_SCHEMA)
        data=reply.get("structured",reply) if isinstance(reply,dict) else {}
        tactic=data.get("tactic","") if isinstance(data,dict) else ""
        if not isinstance(tactic,str) or not tactic.strip() or len(tactic)>MAX_TACTIC_CHARS: return None
        return TacticCandidate(tactic.strip(),"qwen",str(reply.get("model","local")) if isinstance(reply,dict) else "local")

def kimina_public_prompt(case, state, rejected):
    """Build the only Kimina input capsule, exclusively from public evidence."""
    return public_kimina_prompt(
        declaration=case.declaration,
        goal=bounded(state.goal.normalized, 2400),
        prefix=bounded(state.prefix.text(), 1200),
        rejected=[{"tactic": a.tactic, "diagnostic": bounded(a.diagnostic, 300)} for a in rejected[-3:]],
    )

class LeanGoalExtractor:
    def __init__(self, lean, workspace, timeout=30): self.lean,self.workspace,self.timeout=lean,Path(workspace),timeout
    def probe(self, case, prefix):
        if FORBIDDEN.search(prefix.text()): return ProbeOutcome.INTEGRITY_FAILURE,None,"forbidden proof construct"
        body="by\n"+prefix.text()+"\n  all_goals\n    trace_state\n    fail"
        probe=self.workspace/".orchestrator-probe.lean"; probe.write_text(source(case,body))
        try: cp=lean_run(self.lean,self.workspace,[probe.name],self.timeout)
        except (OSError, subprocess.TimeoutExpired) as ex: return ProbeOutcome.INFRASTRUCTURE_FAILURE,None,str(ex)
        finally: probe.unlink(missing_ok=True)
        out=bounded(cp.stdout,8000)
        if cp.returncode==0: return ProbeOutcome.FINAL_PASS,None,out

        # A valid incomplete search probe must fail ONLY because of the deliberate
        # terminal `fail` inserted after trace_state.  Lean can recover from an
        # invalid tactic/term and continue elaborating later commands, so merely
        # finding a printed goal is not evidence that the candidate tactic was
        # accepted.  Reject unknown identifiers/tactics, parse/type errors,
        # unsolved-goal errors caused by a bad prefix, etc. before extracting any
        # successor state.
        error_headers = [
            line for line in out.splitlines()
            if LEAN_ERROR_HEADER.search(line)
        ]
        if not error_headers:
            return ProbeOutcome.INFRASTRUCTURE_FAILURE,None,out
        if not probe_failure_has_only_terminal_sentinel(out):
            return ProbeOutcome.LEAN_SYNTAX_OR_TACTIC_FAILURE,None,out

        # If the candidate itself failed, any subsequently printed state is only
        # diagnostic context, not a transition.  The deliberate terminal
        # ``fail`` below is the sole exception.
        if re.search(r"Tactic `(?!(?:fail)`)[^`]+` failed", out):
            return ProbeOutcome.LEAN_SYNTAX_OR_TACTIC_FAILURE,None,out
        # trace_state is authoritative and consistently includes turnstiles.
        goals=re.findall(r"(?:⊢\s*[^\n]+(?:\n(?!.*?⊢)[^\n]+)*)", out)
        if not goals: return ProbeOutcome.LEAN_SYNTAX_OR_TACTIC_FAILURE,None,out
        normalized="\n---GOAL---\n".join(normalize_goal(g) for g in goals)
        snap=GoalSnapshot(out,normalized,[normalize_goal(g) for g in goals],goal_fingerprint(normalized),out)
        return ProbeOutcome.VALID_NEW_GOAL_STATE,snap,out

def goal_shape(state):
    goal = state.goal.normalized.rsplit("⊢", 1)[-1].strip()
    if "→" in goal or "∀" in goal: return "implication_or_forall"
    if "∧" in goal: return "conjunction"
    if "↔" in goal: return "iff"
    if "∃" in goal: return "existential"
    if any(token in goal for token in ("=", "≤", "<", "+", "*")): return "equality_or_arithmetic"
    return "atomic"

def task_identity(task):
    return getattr(task, "task_id", None) or getattr(task, "case_id", "anonymous-task")

class StrategyPortfolio:
    """Stable, bounded structural Lean tactics; unavailable tactics simply fail probes."""
    registry = (
        DeterministicStrategy("shape.reflexivity", ("rfl",), 0),
        DeterministicStrategy("context.assumption", ("assumption",), 1),
        DeterministicStrategy("decompose.constructor", ("constructor",), 2),
        DeterministicStrategy("decompose.intro", ("intro h",), 3),
        DeterministicStrategy("simplify.simp", ("simp",), 4),
        DeterministicStrategy("automation.aesop", ("aesop",), 5),
        DeterministicStrategy("arithmetic.norm_num", ("norm_num",), 6),
        DeterministicStrategy("arithmetic.omega", ("omega",), 7),
        DeterministicStrategy("algebra.ring", ("ring", "ring_nf", "linarith"), 8),
        DeterministicStrategy("decidable.decide", ("decide",), 9),
    )
    def candidates(self, state):
        seen, result = set(), []
        shape = goal_shape(state)
        ordered = list(self.registry)
        if shape == "conjunction":
            ordered.sort(key=lambda s: 0 if s.strategy_id == "decompose.constructor" else s.priority + 1)
        elif shape == "implication_or_forall":
            ordered.sort(key=lambda s: 0 if s.strategy_id == "decompose.intro" else s.priority + 1)
        # Cheap structural operations run before type closure; broader
        # automation is deliberately left until afterwards.
        cheap=[s for s in ordered if s.priority <= 4]
        if shape == "equality_or_arithmetic":
            cheap=[s for s in cheap if s.strategy_id != "decompose.constructor"]
        broad=[s for s in ordered if s.priority > 4]
        for strategy in cheap:
            for tactic in strategy.tactics:
                if tactic not in seen:
                    seen.add(tactic); result.append(TacticCandidate(tactic, "deterministic", strategy.strategy_id))
        for candidate in TypedContextClosure().candidates(state):
            if candidate.text not in seen:
                seen.add(candidate.text); result.append(candidate)
        for strategy in broad:
            for tactic in strategy.tactics:
                if tactic not in seen:
                    seen.add(tactic); result.append(TacticCandidate(tactic, "deterministic", strategy.strategy_id))
        return result

DEFAULT_PORTFOLIO = StrategyPortfolio()
def deterministic_candidates(state):
    return DEFAULT_PORTFOLIO.candidates(state)

class ProofOrchestrator:
    def __init__(self, lean, workspace, result_dir, max_nodes=64, max_depth=16, qwen_attempts=3, solver=None, planner=None, kimina=None, progress=False):
        self.lean,self.workspace,self.result_dir=lean,Path(workspace),Path(result_dir); self.workspace.mkdir(parents=True,exist_ok=True); self.max_nodes,self.max_depth,self.qwen_attempts=max_nodes,max_depth,qwen_attempts
        self.extractor=LeanGoalExtractor(lean,workspace); self.solver=solver; self.planner=planner or NoopProofPlanner(); self.kimina=kimina; self.progress=progress; self.events=[]; self.attempts=[]; self.checkpoints=[]; self.blueprint=[]; self.proof_dag=ProofDag(); self.capabilities=available_capabilities(self.workspace, lean); self.visited=set(); self.backtracks=0; self.validations=0; self.task_id=""
    def event(self, kind, **kw):
        self.events.append({"event":kind,**kw})
        if self.progress and kind in {"TACTIC_ACCEPTED", "TACTIC_REJECTED", "BACKTRACK", "KIMINA_INVOCATION", "KIMINA_RESPONSE_CAPTURED", "BLUEPRINT_NODE"}:
            result = "accepted" if kind == "TACTIC_ACCEPTED" else "rejected" if kind == "TACTIC_REJECTED" else kind.lower()
            if kind == "BLUEPRINT_NODE":
                print(f"[proof] task={self.task_id} blueprint={kw.get('blueprint')} status={kw.get('status')} deps={len(kw.get('dependencies',[]))} strategy={kw.get('strategy')}", file=sys.stderr)
            else: print(f"[proof] task={self.task_id} state={kw.get('node','?')} strategy={kw.get('strategy',kw.get('source','search'))} result={result}", file=sys.stderr)
    def _blueprint_node(self, proposition, dependencies, strategy, prefix, status="VERIFIED"):
        """Create an acyclic node only after a Lean-accepted transition."""
        node_id=f"bp-{len(self.blueprint)}-{goal_fingerprint(proposition)[:8]}"
        known={x.node_id for x in self.blueprint}
        if node_id in dependencies or any(dep not in known for dep in dependencies):
            self.event("BLUEPRINT_REJECTED", reason="CYCLE_OR_UNKNOWN_DEPENDENCY", strategy=strategy); return None
        node=BlueprintNode(node_id, proposition, goal_fingerprint(proposition), list(dependencies), status, strategy, list(prefix.tactics), "LEAN_ACCEPTED" if status=="VERIFIED" else "UNVERIFIED")
        self.blueprint.append(node); self.event("BLUEPRINT_NODE", blueprint=node_id, status=status.lower(), dependencies=dependencies, strategy=strategy)
        # The richer DAG is intentionally parallel to the legacy JSONL shape
        # so existing consumers remain compatible.
        self.proof_dag.add(ProofDagNode(node_id, proposition, node.fingerprint, list(dependencies), producer="lean-tactic", status="UNVERIFIED"))
        return node
    def _blueprint_verified(self, node_id, prefix, strategy):
        for node in self.blueprint:
            if node.node_id == node_id:
                node.status="VERIFIED"; node.verifier_status="LEAN_ACCEPTED"; node.proof_prefix=list(prefix.tactics)
                self.proof_dag.verify(node_id, verifier="Lean", artifact_hash=hashlib.sha256(prefix.text().encode()).hexdigest())
                self.event("BLUEPRINT_NODE", blueprint=node_id, status="verified", dependencies=node.dependencies, strategy=strategy)
                return
    def _attempt(self, case,node,cand):
        before=node.state.goal.fingerprint; prefix=ProofPrefix(node.state.prefix.tactics+[cand.text]); outcome,snap,diag=self.extractor.probe(case,prefix); self.validations+=1
        if outcome==ProbeOutcome.FINAL_PASS:
            a=TacticAttempt(cand.text,cand.source,outcome.value,bounded(diag),before,"FINAL",cand.detail); self.attempts.append(a); self.event("TACTIC_ACCEPTED",tactic=cand.text,source=cand.source,strategy=cand.detail,outcome=outcome.value,before=before,after="FINAL"); return "final",prefix,None
        if outcome==ProbeOutcome.VALID_NEW_GOAL_STATE and snap and snap.fingerprint!=before and snap.fingerprint not in self.visited:
            a=TacticAttempt(cand.text,cand.source,outcome.value,bounded(diag),before,snap.fingerprint,cand.detail); self.attempts.append(a); self.event("TACTIC_ACCEPTED",tactic=cand.text,source=cand.source,strategy=cand.detail,outcome=outcome.value,before=before,after=snap.fingerprint); return "new",prefix,snap
        reason=ProbeOutcome.NO_PROGRESS_SAME_GOAL.value if snap and snap.fingerprint==before else outcome.value
        self.attempts.append(TacticAttempt(cand.text,cand.source,reason,bounded(diag),before,snap.fingerprint if snap else "",cand.detail)); self.event("TACTIC_REJECTED",tactic=cand.text,source=cand.source,strategy=cand.detail,outcome=reason); return "reject",None,None
    def solve(self, case):
        self.result_dir.mkdir(parents=True,exist_ok=True)
        self.task_id=task_identity(case)
        # A solver may be shared across cases.  Preserve global totals in its
        # metadata, but report case-local deltas as the measurement unit.
        self._kimina_start_invocations = self.kimina.invocations if self.kimina else 0
        self._kimina_start_failures = len(self.kimina.failures) if self.kimina else 0
        initial=ProofPrefix(list(getattr(case, "initial_prefix", ())))
        outcome,goal,diag=self.extractor.probe(case,initial); self.validations+=1
        if not goal: return {"case_id":task_identity(case),"pass":False,"code":outcome.value,"diagnostic":bounded(diag)}
        self.visited.add(goal.fingerprint); root=SearchNode(0,None,ProofState(initial,goal),status="frontier"); frontier=[root]; nodes=[root]; root_bp=self._blueprint_node(goal.normalized, [], "initial.goal", initial, status="UNVERIFIED"); blueprint_for_search={0:root_bp.node_id if root_bp else None}; self.event("INITIAL_GOAL",fingerprint=goal.fingerprint,goal=goal.normalized,shape=goal_shape(root.state))
        recovery_metadata=getattr(case, "recovery_metadata", {})
        assessment=classify_obligation(goal.normalized, metadata=recovery_metadata, diagnostics=diag)
        self.event("OBLIGATION_CLASSIFIED", obligation_class=assessment.obligation_class, estimated_size=assessment.estimated_size, reasoning=assessment.reasoning, strategy=assessment.selected_recovery_strategy)
        self.events.append({"event":"CAPABILITY_MANIFEST", **capability_context(self.capabilities)})
        self.events.extend({"event":"RECOVERY_STAGE", **stage} for stage in recovery_stages(assessment, self.capabilities))
        while frontier and len(nodes)<self.max_nodes:
            # Reproducible best-first ordering over only observed structural state.
            frontier.sort(key=lambda item: (len(item.state.goal.goals), len(item.state.goal.normalized), item.state.depth, item.node_id))
            node=frontier.pop(0); node.status="expanded"
            self.event("DECOMPOSITION", node=node.node_id, strategy=f"shape.{goal_shape(node.state)}", goal_shape=goal_shape(node.state))
            if node.state.depth>=self.max_depth: self.backtracks+=1; self.event("BACKTRACK",node=node.node_id,reason="MAX_DEPTH"); continue
            rejected=[]
            deterministic_progress=False
            # Deterministic search owns the first chance at every state.  Kimina
            # is only a dead-end specialist and cannot replace a checkpoint.
            for cand in deterministic_candidates(node.state):
                self.event("DETERMINISTIC_TACTIC_PROPOSED",tactic=cand.text,source=cand.source,strategy=cand.detail)
                status,prefix,snap=self._attempt(case,node,cand); rejected=self.attempts[-3:]
                if status=="final":
                    self._blueprint_verified(blueprint_for_search.get(node.node_id), prefix, cand.detail)
                    (self.workspace/"Solution.lean").write_text(source(case,"by\n"+prefix.text()))
                    final=validate(self.lean,self.workspace,case); self.validations+=3
                    self.event("FINAL_VERIFICATION_PASS" if final.ok else "INTEGRITY_FAILURE",code=final.code)
                    return self._finish(case,final,prefix,nodes)
                if status=="new":
                    deterministic_progress=True; self.visited.add(snap.fingerprint); child=SearchNode(len(nodes),node.node_id,ProofState(prefix,snap,node.state.depth+1),cand.text,cand.source)
                    nodes.append(child); frontier.append(child); self.checkpoints.append(ProofCheckpoint(child.node_id,node.node_id,prefix,snap,"validated tactic")); parent_bp=blueprint_for_search.get(node.node_id); bp=self._blueprint_node(snap.normalized, [], cand.detail, prefix, status="UNVERIFIED"); blueprint_for_search[child.node_id]=bp.node_id if bp else None
                    if parent_bp and bp:
                        next(x for x in self.blueprint if x.node_id==parent_bp).dependencies.append(bp.node_id)
                    self.event("CHECKPOINT_CREATED",node=child.node_id)
                    # A verified structural decomposition is expanded before
                    # speculative siblings; this keeps the portfolio bounded.
                    break
            candidates=[]
            if self.kimina and not deterministic_progress:
                prompt=kimina_public_prompt(case,node.state,rejected)
                for attempt in range(self.kimina.attempts_per_goal):
                    self.event("KIMINA_INVOCATION",attempt=attempt+1,reason="deterministic_no_progress",state_fingerprint=node.state.goal.fingerprint,**self.kimina.metadata())
                    generation=self.kimina.generate(prompt)
                    self.event("KIMINA_RESPONSE_CAPTURED",
                               transcript_chars=len(generation.transcript), response_chars=len(generation.response),
                               transcript_sha256=generation.transcript_sha256,
                               isolated_response_sha256=generation.response_sha256,
                               formal_region_classification=generation.formal_region_classification,
                               formal_tail=bounded(generation.output, 800),
                               model=self.kimina.model.name, model_sha256=self.kimina.metadata().get("model_sha256"), retry=False)
                    if generation.formal_region_classification == "REASONING_TRUNCATED":
                        retry_budget=min(self.kimina.max_output_tokens * 2, 2048)
                        self.event("KIMINA_RETRY",reason="REASONING_TRUNCATED",attempt=attempt+1,budget_before=self.kimina.max_output_tokens,budget_after=retry_budget)
                        generation=self.kimina.generate(prompt + "\nReturn only a concise formal Lean continuation now.", max_output_tokens=retry_budget)
                        self.event("KIMINA_RESPONSE_CAPTURED", transcript_sha256=generation.transcript_sha256,
                                   isolated_response_sha256=generation.response_sha256,
                                   formal_region_classification=generation.formal_region_classification,
                                   response_chars=len(generation.response), retry=True, retry_budget=retry_budget,
                                   model=self.kimina.model.name)
                    if generation.status != "OK":
                        self.event("KIMINA_OUTPUT_EMPTY" if generation.status=="OUTPUT_EMPTY" else "KIMINA_GENERATION_FAILURE",status=generation.status,diagnostic=bounded(generation.diagnostic,800),returncode=generation.returncode,**self.kimina.metadata())
                        continue
                    # Keep the globally admitted Kimina set within the search
                    # validation cap: every proposed Kimina candidate below is
                    # therefore submitted to the strict Lean transition gate.
                    remaining = 24 - len(candidates)
                    if remaining <= 0:
                        break
                    for text in extract_candidates(generation.output, max_candidates=remaining):
                        candidates.append(TacticCandidate(text,"kimina","specialist.kimina"))
                        self.event("KIMINA_CANDIDATE_PROPOSED",tactic=text,source="kimina",strategy="specialist.kimina",model=self.kimina.model.name)
            if self.solver:
                for _ in range(self.qwen_attempts):
                    try: c=self.solver.propose(case,node.state,rejected)
                    except Exception as ex: self.event("INFRASTRUCTURE_FAILURE",source="qwen",diagnostic=bounded(ex)); break
                    if c: candidates.append(c)
            try:
                plan=self.planner.plan(public_residual(case,node.state,rejected))
                if plan:
                    self.event("REMOTE_PLAN_RECEIVED",model=plan.model,strategy=plan.strategy)
                    # ``have ... := by ...`` is deliberately just another
                    # untrusted, bounded Lean chunk: Lean must accept its
                    # declaration and proof before it becomes a checkpoint.
                    candidates += [TacticCandidate(s.text,"planner",plan.model) for s in plan.steps if s.kind in {"tactic", "have"}]
            except Exception as ex: self.event("INFRASTRUCTURE_FAILURE",source="planner",diagnostic=bounded(ex))
            for cand in candidates[:24]:
                self.event("LOCAL_TACTIC_PROPOSED",tactic=cand.text,source=cand.source)
                status,prefix,snap=self._attempt(case,node,cand); rejected=self.attempts[-3:]
                if status=="final":
                    self._blueprint_verified(blueprint_for_search.get(node.node_id), prefix, cand.detail)
                    (self.workspace/"Solution.lean").write_text(source(case,"by\n"+prefix.text()))
                    final=validate(self.lean,self.workspace,case); self.validations+=3
                    self.event("FINAL_VERIFICATION_PASS" if final.ok else "INTEGRITY_FAILURE",code=final.code)
                    return self._finish(case,final,prefix,nodes)
                if status=="new":
                    self.visited.add(snap.fingerprint); child=SearchNode(len(nodes),node.node_id,ProofState(prefix,snap,node.state.depth+1),cand.text,cand.source)
                    nodes.append(child); frontier.append(child); self.checkpoints.append(ProofCheckpoint(child.node_id,node.node_id,prefix,snap,"validated tactic")); parent_bp=blueprint_for_search.get(node.node_id); bp=self._blueprint_node(snap.normalized, [], cand.detail, prefix, status="UNVERIFIED"); blueprint_for_search[child.node_id]=bp.node_id if bp else None
                    if parent_bp and bp:
                        next(x for x in self.blueprint if x.node_id==parent_bp).dependencies.append(bp.node_id)
                    self.event("CHECKPOINT_CREATED",node=child.node_id)
            if not frontier: self.backtracks+=1; self.event("BACKTRACK",node=node.node_id,reason="BRANCH_EXHAUSTED")
        self.event("SEARCH_EXHAUSTED",nodes=len(nodes)); return self._finish(case,None,None,nodes,assessment=assessment,goal=goal.normalized,diagnostic=diag)
    def _finish(self,case,final,prefix,nodes,assessment=None,goal="",diagnostic=""):
        kimina = self.kimina.metadata() if self.kimina else {"enabled":False}
        if self.kimina:
            kimina = {**kimina,
                      "case_invocations": self.kimina.invocations - self._kimina_start_invocations,
                      "case_generation_failures": len(self.kimina.failures) - self._kimina_start_failures}
        artifact = source(case,"by\n"+prefix.text()) if prefix else ""
        obligation_hash=getattr(case,"source_hash",None) or hashlib.sha256((case.declaration+case.theorem).encode()).hexdigest()
        record={"case_id":task_identity(case),"level":case.level,"pass":bool(final and final.ok),"code":final.code if final else "SEARCH_EXHAUSTED","terminal_state":ProofTerminal.VERIFIED.value if final and final.ok else ProofTerminal.PROOF_RECOVERY_EXHAUSTED.value,"prefix":prefix.tactics if prefix else [],"nodes":len(nodes),"visited_states":len(self.visited),"backtracks":self.backtracks,"lean_validations":self.validations,"max_depth":max((n.state.depth for n in nodes),default=0),"kimina":kimina,"task_source_hash":obligation_hash,"artifact_sha256":hashlib.sha256(artifact.encode()).hexdigest() if artifact else "","accepted_strategies":[a.strategy for a in self.attempts if a.outcome in {ProbeOutcome.FINAL_PASS.value,ProbeOutcome.VALID_NEW_GOAL_STATE.value}]}
        if not record["pass"]:
            meta=getattr(case,"recovery_metadata",{})
            model=None
            if meta.get("finite_additive_basis"):
                spec=meta["finite_additive_basis"]
                model=additive_basis_model(int(spec["n"]),set(spec.get("forbidden",[])),int(spec["max_selected"]),obligation_hash)
            # A handoff is required for finite / SAT-style goals even without a
            # model: it still records the exact Lean residual and resume gate.
            finite=assessment and assessment.obligation_class in {"FINITE_DECIDABLE","FINITE_COMBINATORIAL","SAT_LIKE","ENUMERATIVE_LOWER_BOUND"}
            if model or finite:
                bundle=create_handoff_bundle(self.result_dir/"handoff",obligation_id=self.task_id,obligation_hash=obligation_hash,goal=goal or case.theorem,classification=assessment.obligation_class if assessment else "UNKNOWN",model=model,verified_prefix=record["prefix"],dag_state={k:asdict(v) for k,v in self.proof_dag.nodes.items()},attempts=self.attempts,diagnostics=diagnostic or "search exhausted",capabilities=self.capabilities)
                record["terminal_state"]=bundle["terminal_state"]; record["handoff_path"]=str(self.result_dir/"handoff"/"handoff_bundle.json")
                self.event("ACTIONABLE_HANDOFF",artifacts=len(bundle["hashes"]),bottleneck="residual-unsat" if model else "lean-residual")
        if self.progress: print(f"[proof] task={self.task_id} result={'PASS' if record['pass'] else 'FAIL'} nodes={record['nodes']} lean_checks={self.validations}", file=sys.stderr)
        record["blueprint_nodes"]=len(self.blueprint)
        record["capability_manifest"]=capability_context(self.capabilities)
        record["proof_dag_nodes"]=len(self.proof_dag.nodes)
        for name,items in (("events.jsonl",self.events),("attempts.jsonl",[asdict(x) for x in self.attempts]),("checkpoints.jsonl",[asdict(x) for x in self.checkpoints]),("blueprint.jsonl",[asdict(x) for x in self.blueprint])):
            for item in items: append_jsonl(self.result_dir/name,item)
        if record["pass"] and getattr(case, "output_path", None):
            output = Path(case.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(artifact)
        return record

def prove_task(task_path, *, result_root, max_nodes=64, max_depth=16, progress=False):
    """Run a standalone JSON ProofTask without any benchmark/model dependency."""
    task=ProofTask.from_json(task_path); root=Path(result_root); root.mkdir(parents=True, exist_ok=False)
    workspace=root/"workspace"; workspace.mkdir()
    result=ProofOrchestrator(resolve_lean(),workspace,root, max_nodes=max_nodes, max_depth=max_depth, progress=progress).solve(task)
    (root/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True)); return 0 if result["pass"] else 2

def main(argv=None):
    argv=list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] == "session":
        from tools.proofbench.proof_session import ProofSession
        p=argparse.ArgumentParser(prog="proof-orchestrator session")
        sub=p.add_subparsers(dest="session_command", required=True)
        start=sub.add_parser("start"); start.add_argument("--task", required=True); start.add_argument("--result-root", required=True); start.add_argument("--explain", action="store_true"); start.add_argument("--disable-closer", action="store_true"); start.add_argument("--progress", action="store_true")
        inspect=sub.add_parser("inspect"); inspect.add_argument("--session", required=True)
        resume=sub.add_parser("resume"); resume.add_argument("--session", required=True); resume.add_argument("--human-lemma", required=True); resume.add_argument("--disable-closer", action="store_true"); resume.add_argument("--progress", action="store_true")
        args=p.parse_args(argv[1:])
        if args.session_command == "start":
            session=ProofSession.from_task(args.result_root,args.task).start(explain=args.explain,allow_closer=not args.disable_closer,progress=args.progress)
        elif args.session_command == "inspect":
            session=ProofSession.open(args.session)
        else:
            session=ProofSession.open(args.session).resume(args.human_lemma,allow_closer=not args.disable_closer,progress=args.progress)
        print(json.dumps({"session_id":session.session_id,"session":str(session.path),"status":session.status.value,"metrics":session.metrics},sort_keys=True))
        return 0 if session.status.value == ProofTerminal.VERIFIED.value else 2
    if argv and argv[0] == "prove":
        p=argparse.ArgumentParser(); p.add_argument("prove"); p.add_argument("--task",required=True); p.add_argument("--result-root",required=True); p.add_argument("--max-nodes",type=int,default=64); p.add_argument("--max-depth",type=int,default=16); p.add_argument("--progress",action="store_true"); args=p.parse_args(argv)
        return prove_task(args.task,result_root=args.result_root,max_nodes=args.max_nodes,max_depth=args.max_depth,progress=args.progress)
    p=argparse.ArgumentParser(); p.add_argument("--levels",default="L1,L2,L3"); p.add_argument("--cases-per-level",type=int,default=5); p.add_argument("--max-nodes",type=int,default=64); p.add_argument("--max-depth",type=int,default=16); p.add_argument("--qwen-attempts-per-goal",type=int,default=3); p.add_argument("--result-root",required=True); p.add_argument("--qwen",action="store_true"); p.add_argument("--enable-remote",action="store_true"); p.add_argument("--kimina",action="store_true"); p.add_argument("--kimina-model"); p.add_argument("--kimina-attempts-per-goal",type=int,default=1); p.add_argument("--kimina-reasoning-budget",type=int,default=96); p.add_argument("--kimina-max-output-tokens",type=int,default=640); p.add_argument("--progress",action="store_true"); args=p.parse_args(argv)
    kimina=KiminaMicroProofSolver(model=args.kimina_model,attempts_per_goal=args.kimina_attempts_per_goal,reasoning_budget=args.kimina_reasoning_budget,max_output_tokens=args.kimina_max_output_tokens) if args.kimina else None
    root=Path(args.result_root); root.mkdir(parents=True,exist_ok=False); lean=resolve_lean(); metadata={"created":dt.datetime.now().isoformat(),"remote_enabled":args.enable_remote,"levels":args.levels,"kimina":kimina.metadata() if kimina else {"enabled":False}}; (root/"metadata.json").write_text(json.dumps(metadata,indent=2))
    results=[]; rng=__import__("random").Random(0)
    for level in args.levels.split(","):
        for case in make_cases(level,args.cases_per_level,rng):
            ws=root/"workspaces"/case.case_id; ws.mkdir(parents=True); orch=ProofOrchestrator(lean,ws,root/case.case_id,args.max_nodes,args.max_depth,args.qwen_attempts_per_goal,LocalNextTacticSolver() if args.qwen else None,CodexProofPlanner(enabled=args.enable_remote),kimina, args.progress); results.append(orch.solve(case)); append_jsonl(root/"cases.jsonl",results[-1])
    summary={"cases":len(results),"passed":sum(x["pass"] for x in results),"by_level":{l:{"passed":sum(x["pass"] for x in results if x["level"]==l),"total":sum(x["level"]==l for x in results)} for l in args.levels.split(",")},"results":results}; (root/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
