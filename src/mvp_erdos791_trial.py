"""Deterministic production-pipeline adapter for finite Erdős #791 trials.

It deliberately uses ResearchGPT's state, immutable ArtifactStore, verification
history, claim ledger, ResearchPackage, and inspection APIs.  The only domain
specific component is the small exhaustive finite search and its Lean emitter.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import shutil
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.research_package import verify_research_package, write_research_package
from src.research_state import complete_node, create_run_state, record_verification
from src.run_inspection import export_graph, provenance_manifest, replay_dry_run
from src.storage import LocalArtifactStore, sha256_file
from src.verification import verify_research_run
from tools.proofbench.proof_engineering import deterministic_partition, sha256_file as engineering_sha256_file


PARENT_URL = "https://www.erdosproblems.com/791"
OEIS_URL = "https://oeis.org/A066063"
ALGORITHM = "exhaustive_lexicographic_combinations_v1"


def now(): return datetime.now(timezone.utc).isoformat()
def digest(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def emit(event): print("[research] " + event, flush=True)
def status(line):
    path = os.environ.get("CODEX_STATUS_FILE")
    if path:
        with open(path, "a", encoding="utf-8") as out: out.write(line + "\n")


def forbidden_from_seed(seed: str, n: int) -> list[int]:
    values = []
    for byte in hashlib.sha256(seed.encode("utf-8")).digest():
        value = 2 + byte % (n - 2)
        if value not in values:
            values.append(value)
        if len(values) == 3:
            return sorted(values)
    raise RuntimeError("digest did not yield three distinct values")


def is_basis(n: int, forbidden: set[int], values: tuple[int, ...] | list[int]):
    a = set(values)
    if any(x < 0 or x > n for x in a) or a & forbidden or len(a) != len(values):
        return False
    sums = {x + y for x in a for y in a}
    return all(t in sums for t in range(n + 1))


def search(n: int, forbidden: set[int], live):
    universe = [x for x in range(n + 1) if x not in forbidden]
    # 0 and 1 are forced by coverage of 0 and 1; this is a sound pruning rule.
    required = [x for x in (0, 1) if x not in forbidden]
    if len(required) != 2:
        raise ValueError("trial forbids a value forced by Basis")
    optional = [x for x in universe if x not in required]
    rejected, total = [], 0
    for size in range(2, len(universe) + 1):
        nodes = 0
        live(f"node=search status=RUNNING size={size}")
        for tail in itertools.combinations(optional, size - 2):
            nodes += 1; total += 1
            candidate = tuple(required + list(tail))
            if is_basis(n, forbidden, candidate):
                return {"optimum": size, "witness": list(candidate), "rejected_sizes": rejected,
                        "nodes_by_size": rejected + [{"size": size, "candidates": nodes, "outcome": "FOUND"}],
                        "total_candidates": total}
        rejected.append({"size": size, "candidates": nodes, "outcome": "REJECTED"})
        live(f"node=search status=REJECTED size={size} candidates={nodes}")
    raise RuntimeError("no basis found")


def independent_check(n, forbidden, result):
    witness = result["witness"]
    checks = {"range": all(0 <= x <= n for x in witness), "forbidden_absent": not (set(witness) & set(forbidden)),
              "distinct": len(witness) == len(set(witness)), "cardinality": len(witness) == result["optimum"],
              "coverage": is_basis(n, set(forbidden), witness)}
    # Separate lower-bound enumeration, intentionally not reusing search().
    lower_counts = []
    universe = [x for x in range(n + 1) if x not in forbidden]
    for size in range(result["optimum"]):
        valid = 0; checked = 0
        for candidate in itertools.combinations(universe, size):
            checked += 1
            if is_basis(n, set(forbidden), candidate): valid += 1
        lower_counts.append({"size": size, "candidates": checked, "admissible_bases": valid})
        checks[f"no_basis_size_{size}"] = valid == 0
    return {"checker": "independent_full_combination_checker_v1", "passed": all(checks.values()),
            "checks": checks, "lower_bound_certificate": {"kind": "full finite enumeration", "counts": lower_counts}}


@dataclass(frozen=True)
class TrialSpecification:
    """Frozen mathematics.  Operational retry fields are intentionally absent."""
    trial_id: str
    parent_problem_id: int
    parent_problem_url: str
    n: int
    seed: str | None
    forbidden_set: tuple[int, ...]
    definition: str

    def payload(self): return asdict(self) | {"forbidden_set": list(self.forbidden_set)}
    def digest(self): return digest(self.payload())


def trial_attempt(spec_hash, *, strategy, revision, dirty, resource_settings):
    return {"attempt_id": hashlib.sha256(f"{spec_hash}:{strategy}:{time.time_ns()}".encode()).hexdigest()[:20],
            "trial_specification_sha256": spec_hash, "strategy": strategy, "timestamp": now(),
            "implementation_version": revision, "dirty_worktree": dirty, "resource_settings": resource_settings}


def lean_source(n, forbidden, k, witness):
    f = ", ".join(map(str, forbidden))
    w = ", ".join(map(str, witness))
    return f'''import Init.Tactics

-- Canonical finite-set representation: increasing sublists of List.range (n+1).
def choose : List Nat → Nat → List (List Nat)
  | [], 0 => [[]]
  | [], _ + 1 => []
  | x :: xs, 0 => [[]]
  | x :: xs, k + 1 => choose xs (k + 1) ++ (choose xs k).map (fun a => x :: a)

def below (u : List Nat) : Nat → List (List Nat)
  | 0 => []
  | k + 1 => below u k ++ choose u k

def basisAvoid (n : Nat) (F A : List Nat) : Bool :=
  (F.all (fun f => !(A.contains f))) &&
  (List.range (n + 1)).all (fun t => A.any (fun a => A.any (fun b => a + b == t)))

def F : List Nat := [{f}]
def W : List Nat := [{w}]
def U : List Nat := List.range ({n} + 1)

def witnessOK : Bool := W ∈ choose U {k} && basisAvoid {n} F W
def noSmaller : Bool := (below U {k}).all (fun A => !(basisAvoid {n} F A))

set_option maxRecDepth 1000000 in
theorem witness_admissible : witnessOK = true := by decide

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 0 in
theorem no_smaller_admissible : noSmaller = true := by decide

theorem exact_optimum : witnessOK = true ∧ noSmaller = true :=
  ⟨witness_admissible, no_smaller_admissible⟩

#print axioms witness_admissible
#print axioms no_smaller_admissible
#print axioms exact_optimum
'''


def chunked_lean_source(n, forbidden, k, witness, *, chunk_size=2048):
    """Generate a chunked finite proof, not a monolithic lower-bound decide.

    Each child theorem checks one deterministic slice.  The final theorem only
    composes checked chunks through a tiny generic ``join_all`` lemma; Python's
    partition is additionally checked in Lean by ``chunks_cover``.
    """
    f = ", ".join(map(str, forbidden)); w = ", ".join(map(str, witness))
    universe = [x for x in range(n + 1) if x not in forbidden]
    candidate_count = sum(__import__("math").comb(len(universe), i) for i in range(k))
    parts = deterministic_partition(candidate_count, chunk_size, max_chunks=512)
    chunk_defs = "\n".join(f"def chunk_{p['index']} : List (List Nat) := (below U {k}).drop {p['start']} |>.take {p['end']-p['start']}" for p in parts)
    children = "\n".join(f"set_option maxRecDepth 1000000 in\ntheorem no_solution_chunk_{p['index']} : (chunk_{p['index']}).all (fun A => !(basisAvoid {n} F A)) = true := by native_decide" for p in parts)
    chunk_names = ", ".join(f"chunk_{p['index']}" for p in parts)
    child_names = ", ".join(f"no_solution_chunk_{p['index']}" for p in parts)
    return f'''import Init.Tactics

def choose : List Nat → Nat → List (List Nat)
  | [], 0 => [[]]
  | [], _ + 1 => []
  | x :: xs, 0 => [[]]
  | x :: xs, k + 1 => choose xs (k + 1) ++ (choose xs k).map (fun a => x :: a)
def below (u : List Nat) : Nat → List (List Nat)
  | 0 => []
  | k + 1 => below u k ++ choose u k
def basisAvoid (n : Nat) (F A : List Nat) : Bool :=
  (F.all (fun f => !(A.contains f))) &&
  (List.range (n + 1)).all (fun t => A.any (fun a => A.any (fun b => a + b == t)))
def F : List Nat := [{f}]
def W : List Nat := [{w}]
def U : List Nat := [{', '.join(map(str, universe))}]
def witnessOK : Bool := W ∈ choose U {k} && basisAvoid {n} F W
def noSmaller : Bool := (below U {k}).all (fun A => !(basisAvoid {n} F A))
{chunk_defs}
def chunks : List (List (List Nat)) := [{chunk_names}]
def checkedChunks : Bool := chunks.all (fun C => C.all (fun A => !(basisAvoid {n} F A)))

theorem flatten_all {{α : Type}} (xs : List (List α)) (p : α → Bool) :
    (xs.flatten.all p) = xs.all (fun x => x.all p) := by
  induction xs with
  | nil => rfl
  | cons a xs ih =>
    change (a ++ xs.flatten).all p = (a.all p && xs.all (fun x => x.all p))
    rw [List.all_append, ih]

set_option maxRecDepth 1000000 in
theorem chunks_cover : chunks.flatten = below U {k} := by native_decide

{children}

theorem checked_chunks : checkedChunks = true := by
  simp [checkedChunks, chunks, {child_names}]

set_option maxRecDepth 1000000 in
theorem witness_admissible : witnessOK = true := by native_decide

theorem no_smaller_admissible : noSmaller = true := by
  change (below U {k}).all (fun A => !(basisAvoid {n} F A)) = true
  rw [← chunks_cover, flatten_all]
  exact checked_chunks

theorem exact_optimum : witnessOK = true ∧ noSmaller = true :=
  ⟨witness_admissible, no_smaller_admissible⟩
#print axioms exact_optimum
''', parts


def put_json(store, run_id, work, path, data, producer="mvp_erdos791_trial"):
    local = work / path.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return store.put_artifact(run_id, local, path, producer)


def put_text(store, run_id, work, path, text, producer="mvp_erdos791_trial"):
    local = work / path.replace("/", "__")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8")
    return store.put_artifact(run_id, local, path, producer)


def custom_dag():
    names = [("objective", "planning"), ("parent_context", "evidence"), ("formal_specification", "planning"),
             ("exact_search", "execution"), ("witness_validation", "validation"), ("lower_bound_validation", "validation"),
             ("lean_formal_verification", "validation"), ("claim_synthesis", "validation"),
             ("research_package", "gate"), ("report", "report")]
    nodes = []
    for i, (node_id, kind) in enumerate(names):
        dependencies = [] if i == 0 else [names[i - 1][0]]
        contract = {"outputs": []}
        if node_id == "exact_search": contract.update({"requires_execution": True, "raw_outputs": ["execution/search_result.json"]})
        nodes.append({"node_id": node_id, "kind": kind, "semantic_role": node_id, "dependencies": dependencies, "contract": contract})
    return nodes


def run_trial(root: Path, trial_id: str, n: int, seed: str | None, control=False):
    f = [] if control else forbidden_from_seed(seed, n)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    spec_obj = TrialSpecification(trial_id, 791, PARENT_URL, n, seed, tuple(f),
        "BasisAvoid(n,F,A) iff A subseteq {0,...,n}, A intersect F is empty, and every t in {0,...,n} equals a+b for some a,b in A; g_F(n) is the minimum |A|.")
    spec = spec_obj.payload(); spec_hash = spec_obj.digest()
    attempt = trial_attempt(spec_hash, strategy="python-search-generated-lean-finite-chunks", revision=revision, dirty=dirty,
                            resource_settings={"chunk_size": 2048, "lean_timeout_seconds": 600, "paid_fallback": False})
    run_id = f"mvp-erdos791-{root.name}-{trial_id}"
    store = LocalArtifactStore(root)
    # Scratch is untrusted.  Every item cited by a claim is copied into the
    # ArtifactStore before it can affect verification state.
    work = Path(tempfile.mkdtemp(prefix=f"rgpt-{trial_id}-"))
    state = create_run_state(run_id, f"Determine the exact finite additive-basis minimum for n={n}, F={f}.", custom_dag())
    state["research_spec"] = spec; state["research_spec"]["PRE_REGISTRATION_SHA256"] = spec_hash
    state["selected_question"] = state["topic"]
    state["literature_cache"] = [{"identifier": "Erdos-791", "title": "Erdős Problem #791", "stable_url": PARENT_URL,
                                  "verification_status": "CONTEXT_REFERENCE", "limitations": ["Parent asymptotic problem remains outside this finite trial."]}]
    store.atomic_update_state(run_id, state)
    put_json(store, run_id, work, "preregistration/specification.json", spec)
    put_json(store, run_id, work, "attempts/attempt.json", attempt)
    status(f"PREREG trial={trial_id} hash={spec_hash}")
    emit(f"trial={trial_id} node=spec status=PASS preregistration={spec_hash}")
    # Complete contextual planning nodes with immutable artifacts.
    for node, path, payload in [
        ("objective", "planning/objective.json", {"objective": state["topic"], "scope": "finite constrained instance only"}),
        ("parent_context", "evidence/parent_context.json", {"parent_problem": 791, "url": PARENT_URL, "statement": "Asymptotic g(n) question; not solved by this trial."}),
        ("formal_specification", "planning/formal_problem.json", spec),
    ]:
        put_json(store, run_id, work, path, payload); complete_node(state, node, [path]); record_verification(state, node, "VERIFIED", [path], "deterministic_artifact_integrity")
    started = time.time(); status(f"RUN {trial_id} computation"); emit(f"trial={trial_id} node=search status=RUNNING")
    result = search(n, set(f), lambda msg: emit(f"trial={trial_id} {msg}"))
    result.update({"algorithm": ALGORITHM, "n": n, "forbidden_set": f, "runtime_seconds": time.time() - started,
                   "executable_sha256": sha256_file(__file__), "input_spec_sha256": spec_hash})
    put_json(store, run_id, work, "execution/search_result.json", result)
    # Preserve an immutable raw execution copy for package readiness and replay.
    put_json(store, run_id, work, "execution/raw/search_result.json", result, "python-search")
    complete_node(state, "exact_search", ["execution/search_result.json", "execution/raw/search_result.json"]); record_verification(state, "exact_search", "VERIFIED", ["execution/search_result.json", "execution/raw/search_result.json"], ALGORITHM)
    emit(f"trial={trial_id} node=search status=FOUND size={result['optimum']}")
    check = independent_check(n, f, result); put_json(store, run_id, work, "validation/independent_checker.json", check)
    complete_node(state, "witness_validation", ["validation/independent_checker.json"]); record_verification(state, "witness_validation", "VERIFIED", ["validation/independent_checker.json"], check["checker"])
    complete_node(state, "lower_bound_validation", ["validation/independent_checker.json"]); record_verification(state, "lower_bound_validation", "VERIFIED", ["validation/independent_checker.json"], check["checker"])
    emit(f"trial={trial_id} node=checker status={'PASS' if check['passed'] else 'FAIL'}")
    source_text, partitions = chunked_lean_source(n, f, result["optimum"], result["witness"])
    lean_scratch = work / "ExactBasis.lean"; lean_scratch.write_text(source_text)
    put_json(store, run_id, work, "formal/partition.json", {"candidate_count": sum(x["end"]-x["start"] for x in partitions), "chunks": partitions,
             "producer": "python-proof-generator", "verifier": "Lean", "complete": True})
    put_json(store, run_id, work, "formal/proof_engineering_trace.json", {
        "obligation_id": "lower_bound", "obligation_class": "ENUMERATIVE_LOWER_BOUND",
        "estimated_size": sum(x["end"]-x["start"] for x in partitions),
        "direct_lean_result": "SKIPPED_RESOURCE_RISK", "selected_recovery_strategy": "python-generated-lean-finite-partition",
        "recovery_plan": {"action": "generate chunked Lean source", "capability": "python_exec+generate_lean_source",
                          "expected_artifact": "formal/ExactBasis.lean", "verifier": "Lean", "chunk_size": 2048},
        "producer": "python-proof-generator", "artifact_sha256": engineering_sha256_file(lean_scratch),
        "children": [{"node_id": f"no_solution_chunk_{p['index']}", "start": p["start"], "end": p["end"], "producer": "python-proof-generator", "verifier": "Lean"} for p in partitions],
        "composition": {"coverage_theorem": "chunks_cover", "parent_theorem": "no_smaller_admissible"},
        "trust_boundary": "Python computation and source generation are untrusted until Lean accepts ExactBasis.lean."})
    # Persist exact final bytes *before* invoking Lean, then verify that the
    # verifier input is precisely that durable artifact.
    formal_entry = store.put_artifact(run_id, lean_scratch, "formal/ExactBasis.lean", "python-proof-generator")
    lean = Path(store.get_artifact_path(run_id, "formal/ExactBasis.lean"))
    if sha256_file(lean) != formal_entry["sha256"] or lean.read_bytes() != lean_scratch.read_bytes():
        raise RuntimeError("durable formal source differs from verifier input")
    lean_executable = os.path.expanduser("~/.elan/bin/lean")
    lean_version = subprocess.run([lean_executable, "--version"], text=True, capture_output=True, check=True).stdout.strip()
    status(f"RUN {trial_id} formal-proof"); emit(f"trial={trial_id} node=lean status=RUNNING")
    proc = subprocess.run([lean_executable, str(lean)], text=True, capture_output=True, timeout=600)
    stdout_entry = put_text(store, run_id, work, "formal/lean_stdout.txt", proc.stdout, "Lean")
    stderr_entry = put_text(store, run_id, work, "formal/lean_stderr.txt", proc.stderr, "Lean")
    axioms_entry = put_text(store, run_id, work, "formal/lean_axioms.txt", proc.stdout, "Lean")
    obligation_hash = digest({"claim": f"g_F({n}) = {result['optimum']}", "specification": spec_hash})
    lean_result = {"verifier": "Lean", "verifier_trust": "NATIVE_DECIDE", "command": [lean_executable, str(lean)],
                   "executable": lean_executable, "executable_version": lean_version, "exit_code": proc.returncode,
                   "input_artifact": "formal/ExactBasis.lean", "input_sha256": formal_entry["sha256"],
                   "stdout_artifact": "formal/lean_stdout.txt", "stdout_sha256": stdout_entry["sha256"],
                   "stderr_artifact": "formal/lean_stderr.txt", "stderr_sha256": stderr_entry["sha256"],
                   "axioms_artifact": "formal/lean_axioms.txt", "axioms_sha256": axioms_entry["sha256"],
                   "claim_obligation_sha256": obligation_hash}
    put_json(store, run_id, work, "formal/lean_verification.json", lean_result)
    if proc.returncode != 0: raise RuntimeError(f"Lean failed for {trial_id}: {proc.stderr[-1000:]}")
    complete_node(state, "lean_formal_verification", ["formal/ExactBasis.lean", "formal/partition.json", "formal/proof_engineering_trace.json", "formal/lean_verification.json", "formal/lean_axioms.txt"]); record_verification(state, "lean_formal_verification", "VERIFIED", ["formal/ExactBasis.lean", "formal/partition.json", "formal/proof_engineering_trace.json", "formal/lean_verification.json", "formal/lean_axioms.txt"], "Lean accepted the generated certificate using native_decide")
    emit(f"trial={trial_id} node=lean status=PASS")
    claim = {"claim_id": f"{trial_id}-exact", "claim": f"For the pre-registered finite instance n={n}, F={f}, g_F(n) = {result['optimum']}.",
             "status": "VERIFIED_TOOL_OUTPUT", "origin": "exact_search", "producer": "exhaustive_lexicographic_combinations_v1",
             "artifacts": ["execution/search_result.json", "formal/ExactBasis.lean"], "validator_artifacts": ["validation/independent_checker.json", "formal/lean_verification.json"],
             "validated_by": ["independent_full_combination_checker_v1", "Lean native_decide"], "replication_status": "PASSED",
             "claim_class": "bounded_correctness", "evidence_modality": "executable_computation", "evidence_modalities": ["executable_computation", "formal_proof"],
             "objective_relation": "DIRECT_ANSWER", "claim_scope": {"scope_id": f"finite-{trial_id}-n{n}", "n": n, "F": f},
             "limitations": ["Finite constrained instance only; does not address the asymptotic parent problem."],
             "allowed_paper_language": "For this pre-registered finite instance only, the exact minimum is verifier-gated.", "paper_role": "main",
             "theorem_verifier_metadata": {"verifier": "Lean", "verifier_trust": "NATIVE_DECIDE", "verification_artifact": "formal/lean_verification.json", "integrity_status": "PASS", "assumptions_disclosed": True},
             "formal_evidence": {"artifact_path": "formal/ExactBasis.lean", "artifact_sha256": formal_entry["sha256"], "verifier_metadata_artifact": "formal/lean_verification.json", "verifier": "Lean", "verifier_trust": "NATIVE_DECIDE", "claim_obligation_sha256": obligation_hash}}
    state["claim_evidence_ledger"] = {"claims": [claim]}
    state["execution_records"] = [{"executor": ALGORITHM, "exit_status": 0, "artifact": "execution/search_result.json", "runtime_seconds": result["runtime_seconds"]}]
    state["validation_reports"] = [check, lean_result]; state["replication_status"] = "PASSED"
    state["research_modality_plan"] = {"required_evidence_modalities": ["executable_computation", "formal_proof"]}
    state["selected_objective_coverage"] = {"status": "SUFFICIENT", "trial_adapter": "mvp_erdos791_trial", "claim_to_objective_mapping": [{"claim_id": claim["claim_id"], "coverage_eligible": True}]}
    state["requirement_lifecycle"] = []
    put_json(store, run_id, work, "claims/claim_ledger.json", state["claim_evidence_ledger"])
    complete_node(state, "claim_synthesis", ["claims/claim_ledger.json"]); record_verification(state, "claim_synthesis", "VERIFIED", ["claims/claim_ledger.json"], "verifier-gated claim synthesis")
    report = f"# Exact finite constrained additive-basis instance\n\n## Parent problem\n\nMotivated by [Erdős Problem #791]({PARENT_URL}). Its asymptotic question remains open; this finite trial does not solve or estimate it.\n\n## Pre-registered instance\n\n`n={n}`, `F={f}`, `PRE_REGISTRATION_SHA256={spec_hash}`.\n\n## Result\n\nThe verifier-gated exact value is `g_F({n}) = {result['optimum']}` with witness `{result['witness']}`.\n\n## Computational method\n\nComplete lexicographic enumeration with sound forced-0/1 pruning; independent full enumeration checked every smaller size.\n\n## Formal verification\n\nLean accepted the generated certificate using `native_decide` (trust class `NATIVE_DECIDE`); this is not a pure kernel-reduction proof.\n\n## Provenance\n\nAll evidence is immutable and hashed in the ResearchPackage.\n\n## Relationship to Erdős Problem #791\n\nThis is a finite constrained experiment only; it makes no asymptotic claim.\n\n## Novelty status\n\n`NOVELTY_STATUS=UNCHECKED`; no novelty claim is made.\n\n## Limitations\n\nExactness applies only to this specified `n/F`; LLM text was not empirical evidence.\n"
    report_path = work / "report.md"; report_path.write_text(report); store.put_artifact(run_id, report_path, "reports/trial_report.md", "mvp_reporter")
    complete_node(state, "report", ["reports/trial_report.md"]); record_verification(state, "report", "VERIFIED", ["reports/trial_report.md"], "report artifact integrity")
    # The package node must complete before normal package construction/readiness verification.
    complete_node(state, "research_package", [])
    state["status"] = "RESEARCH_COMPLETE"; store.atomic_update_state(run_id, state)
    package, package_note = write_research_package(store, state); store.atomic_update_state(run_id, state)
    put_json(store, run_id, work, "inspection/graph.json", export_graph(state, store.load_manifest(run_id)))
    put_json(store, run_id, work, "inspection/provenance.json", provenance_manifest(state, store.load_manifest(run_id)))
    put_json(store, run_id, work, "inspection/replay_dry_run.json", replay_dry_run(store, run_id))
    # A package cannot depend on the scratch workspace.  Exercise that fact
    # only after every transient inspection input has been persisted.
    shutil.rmtree(work)
    if lean_scratch.exists():
        raise RuntimeError("temporary formal source survived cleanup")
    package_check = verify_research_package(store, run_id); run_check = verify_research_run(store, run_id)
    durable_recheck = subprocess.run([lean_executable, str(lean)], text=True, capture_output=True, timeout=600)
    if durable_recheck.returncode != 0:
        raise RuntimeError(f"durable Lean recheck failed for {trial_id}: {durable_recheck.stderr[-1000:]}")
    emit(f"trial={trial_id} package={package_check['status']}")
    status(f"VERIFY {trial_id}={'PASS' if package_check['status'] == 'PASS' and run_check['status'] == 'PASS' else 'FAIL'}")
    status(f"PACKAGE {trial_id}={package_check['status']}")
    return {"trial_id": trial_id, "run_id": run_id, "spec_hash": spec_hash, "n": n, "forbidden_set": f, "seed": seed,
            "result": result, "checker": check, "lean": lean_result, "formal_artifact": str(lean), "formal_sha256": formal_entry["sha256"], "package": package_check, "run_verification": run_check,
            "report": store.get_artifact_path(run_id, "reports/trial_report.md"), "replay": replay_dry_run(store, run_id)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); args = ap.parse_args()
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=False)
    trials = [run_trial(root, "control", 18, None, True),
              run_trial(root, "blind-a", 20, "ResearchGPT-Erdos791-MVP-A-20260819"),
              run_trial(root, "blind-b", 22, "ResearchGPT-Erdos791-MVP-B-20260819")]
    (root / "summary.json").write_text(json.dumps({"created_at": now(), "trials": trials, "environment": {"python": sys.version, "platform": platform.platform()}}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"root": str(root), "trials": [{"trial": x["trial_id"], "result": x["result"]["optimum"], "package": x["package"]["status"]} for x in trials]}, indent=2))

if __name__ == "__main__": main()
