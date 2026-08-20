#!/usr/bin/env python3
"""Public, local-only Lean calibration ladder.

Purpose:
  Measure whether the same local model used by ProofBench V9 can repair
  increasingly difficult Lean proofs when given the exact compiler diagnostic.

This is NOT qualification:
  - no hidden cases
  - no remote models
  - no paid fallback
  - no planner/action selection
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_gateway import LLMRequest
from tools.proofbench.proof_engine import FORBIDDEN
from tools.proofbench.v9_controller import (
    EDIT_SCHEMA,
    MAX_REPLACEMENT_CHARS,
    v9_local_provider,
)


LEVELS = ("L1", "L2", "L3")


@dataclass(frozen=True)
class GymCase:
    level: str
    case_id: str
    theorem: str
    declaration: str
    expected_type: str
    description: str


@dataclass
class Validation:
    ok: bool
    rank: int
    code: str
    diagnostic: str


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fp(text: str) -> str:
    normalized = re.sub(r"\d+", "#", re.sub(r"\s+", " ", str(text or "")))
    return sha(normalized)[:16]


def bounded(text: str, n: int = 3500) -> str:
    return str(text or "")[-n:]


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def resolve_lean() -> str:
    elan = Path.home() / ".elan" / "bin" / "lean"
    if elan.is_file() and os.access(elan, os.X_OK):
        return str(elan)

    found = shutil.which("lean")
    if found:
        return found

    raise RuntimeError("LEAN_RUNTIME_UNAVAILABLE")


def lean_run(
    lean: str,
    workspace: Path,
    args: list[str],
    seconds: int = 60,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    old = env.get("LEAN_PATH", "")
    env["LEAN_PATH"] = str(workspace) + (os.pathsep + old if old else "")

    return subprocess.run(
        [
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=5s",
            f"{seconds}s",
            lean,
            *args,
        ],
        cwd=workspace,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def randtag(rng) -> str:
    chars = "abcdefghjkmnpqrstuvwxyz23456789"
    return "".join(rng.choice(chars) for _ in range(7))


def make_case(level: str, index: int, rng) -> GymCase:
    tag = randtag(rng)
    theorem = f"target_{tag}"

    if level == "L1":
        typ = ("Nat", "Bool", "String", "List Nat", "Nat × Nat")[index % 5]
        x = f"x_{tag}"

        return GymCase(
            level,
            f"{level}-{index+1:03d}-{tag}",
            theorem,
            f"({x} : {typ}) : {x} = {x}",
            f"∀ ({x} : {typ}), {x} = {x}",
            "Prove a reflexive equality.",
        )

    if level == "L2":
        P, Q, R = f"P_{tag}", f"Q_{tag}", f"R_{tag}"

        variants = [
            (
                f"({P} {Q} : Prop) (hP : {P}) (hQ : {Q}) : {P} ∧ {Q}",
                f"∀ ({P} {Q} : Prop), {P} → {Q} → {P} ∧ {Q}",
                "Construct a conjunction.",
            ),
            (
                f"({P} {Q} : Prop) (h : {P} ∧ {Q}) : {Q} ∧ {P}",
                f"∀ ({P} {Q} : Prop), {P} ∧ {Q} → {Q} ∧ {P}",
                "Swap conjunction components.",
            ),
            (
                f"({P} {Q} : Prop) (hP : {P}) : {Q} → {P}",
                f"∀ ({P} {Q} : Prop), {P} → {Q} → {P}",
                "Introduce an implication and reuse an assumption.",
            ),
            (
                f"({P} {Q} {R} : Prop) "
                f"(hPQ : {P} → {Q}) (hQR : {Q} → {R}) : {P} → {R}",
                f"∀ ({P} {Q} {R} : Prop), "
                f"({P} → {Q}) → ({Q} → {R}) → {P} → {R}",
                "Compose implications.",
            ),
            (
                f"({P} {Q} : Prop) : {P} ∧ {Q} → {P}",
                f"∀ ({P} {Q} : Prop), {P} ∧ {Q} → {P}",
                "Project from a conjunction.",
            ),
        ]

        declaration, expected, description = variants[index % len(variants)]

        return GymCase(
            level,
            f"{level}-{index+1:03d}-{tag}",
            theorem,
            declaration,
            expected,
            description,
        )

    if level == "L3":
        a, b, c = f"a_{tag}", f"b_{tag}", f"c_{tag}"
        xs = f"xs_{tag}"

        variants = [
            (
                f"({a} {b} : Nat) (h : {a} = {b}) : 0 + {a} = {b}",
                f"∀ ({a} {b} : Nat), {a} = {b} → 0 + {a} = {b}",
                "Simplify and rewrite a Nat equality.",
            ),
            (
                f"({a} {b} : Nat) (h : {a} = {b}) : "
                f"Nat.succ {a} = Nat.succ {b}",
                f"∀ ({a} {b} : Nat), {a} = {b} → "
                f"Nat.succ {a} = Nat.succ {b}",
                "Transport equality through Nat.succ.",
            ),
            (
                f"({a} {b} : Nat) : {a} + {b} = {b} + {a}",
                f"∀ ({a} {b} : Nat), {a} + {b} = {b} + {a}",
                "Prove Nat addition commutativity.",
            ),
            (
                f"({a} {b} {c} : Nat) : "
                f"({a} + {b}) + {c} = {a} + ({b} + {c})",
                f"∀ ({a} {b} {c} : Nat), "
                f"({a} + {b}) + {c} = {a} + ({b} + {c})",
                "Prove Nat addition associativity.",
            ),
            (
                f"({xs} : List Nat) : {xs}.reverse.length = {xs}.length",
                f"∀ ({xs} : List Nat), {xs}.reverse.length = {xs}.length",
                "Prove a simple List reverse/length fact.",
            ),
        ]

        declaration, expected, description = variants[index % len(variants)]

        return GymCase(
            level,
            f"{level}-{index+1:03d}-{tag}",
            theorem,
            declaration,
            expected,
            description,
        )

    raise ValueError(level)


def make_cases(level: str, count: int, rng) -> list[GymCase]:
    return [make_case(level, i, rng) for i in range(count)]


def source(case: GymCase, body: str) -> str:
    # ProofTask and GymCase share the orchestrator without benchmark coupling.
    if hasattr(case, "render_source"):
        return case.render_source(body)
    return (
        "import Std\n"
        "set_option autoImplicit false\n"
        "namespace Gym\n\n"
        f"theorem {case.theorem} {case.declaration} := {body}\n\n"
        "end Gym\n"
    )


def initial_source(case: GymCase) -> str:
    # Intentionally wrong without using sorry/admit.
    return source(case, "by\n  exact 0")


def validate(
    lean: str,
    workspace: Path,
    case: GymCase,
) -> Validation:
    solution = workspace / "Solution.lean"

    if not solution.is_file():
        return Validation(False, 0, "MISSING_SOLUTION", "")

    text = solution.read_text()

    if FORBIDDEN.search(text):
        return Validation(
            False,
            0,
            "PROOF_INTEGRITY_FAILURE",
            "forbidden proof construct",
        )

    (workspace / "Solution.olean").unlink(missing_ok=True)

    cp = lean_run(
        lean,
        workspace,
        ["-o", "Solution.olean", "Solution.lean"],
    )

    if cp.returncode != 0:
        return Validation(
            False,
            1,
            "LEAN_COMPILATION_FAILURE",
            bounded(cp.stdout),
        )

    probe = workspace / ".shape.lean"
    probe.write_text(
        "import Solution\n"
        f"example : {case.expected_type} := {getattr(case, 'qualified_theorem', 'Gym.' + case.theorem)}\n"
    )

    try:
        cp = lean_run(lean, workspace, [probe.name])
    finally:
        probe.unlink(missing_ok=True)

    if cp.returncode != 0:
        return Validation(
            False,
            2,
            "THEOREM_SHAPE_FAILURE",
            bounded(cp.stdout),
        )

    probe = workspace / ".axioms.lean"
    probe.write_text(
        "import Solution\n"
        f"#print axioms {getattr(case, 'qualified_theorem', 'Gym.' + case.theorem)}\n"
    )

    try:
        cp = lean_run(lean, workspace, [probe.name])
    finally:
        probe.unlink(missing_ok=True)

    if cp.returncode != 0:
        return Validation(
            False,
            3,
            "AXIOM_CHECK_FAILURE",
            bounded(cp.stdout),
        )

    if "sorryAx" in cp.stdout:
        return Validation(
            False,
            3,
            "AXIOM_INTEGRITY_FAILURE",
            bounded(cp.stdout),
        )

    return Validation(True, 4, "PASS", bounded(cp.stdout))


def make_prompt(
    case: GymCase,
    candidate: str,
    validation: Validation,
    turn: int,
) -> str:
    obj = {
        "mode": "PUBLIC_LOCAL_ONLY_PROOF_GYM",
        "protocol": (
            "Return only a complete exact replacement for Solution.lean "
            "inside the required replacement field. "
            "Do not return prose, shell commands, paths, SHA256, or diffs. "
            "Do not use sorry, admit, axiom, sorryAx, or unsafe. "
            "Preserve the required theorem name and theorem type."
        ),
        "level": case.level,
        "case_id": case.case_id,
        "turn": turn,
        "description": case.description,
        "required_theorem": f"Gym.{case.theorem}",
        "required_type": case.expected_type,
        "diagnostic_code": validation.code,
        "diagnostic": bounded(validation.diagnostic, 2200),
        "candidate_file": "Solution.lean",
        "candidate": candidate[:5500],
    }

    prompt = json.dumps(obj, sort_keys=True)

    if len(prompt) > 12000:
        raise RuntimeError("PROOF_GYM_PROMPT_TOO_LARGE")

    return prompt


def response_meta(reply) -> dict:
    if not isinstance(reply, dict):
        return {}

    keys = (
        "model",
        "selected_configuration_id",
        "weight_quantization",
        "kv_quantization",
        "input_tokens",
        "output_tokens",
        "wall_clock_duration_seconds",
    )

    return {
        k: reply[k]
        for k in keys
        if k in reply and reply[k] is not None
    }


def run_case(
    provider,
    lean: str,
    root: Path,
    case: GymCase,
    max_edits: int,
) -> dict:
    ws = root / "cases" / case.case_id
    ws.mkdir(parents=True, exist_ok=False)

    (ws / "TASK.md").write_text(
        f"{case.description}\n\n"
        f"Required theorem:\nGym.{case.theorem} : {case.expected_type}\n"
    )

    candidate = initial_source(case)
    (ws / "Solution.lean").write_text(candidate)
    (ws / "Solution.turn0.lean").write_text(candidate)

    state = validate(lean, ws, case)

    append_jsonl(
        root / "attempts.jsonl",
        {
            "event": "INITIAL_VALIDATION",
            "case_id": case.case_id,
            "level": case.level,
            "code": state.code,
            "rank": state.rank,
            "diagnostic_fingerprint": fp(state.diagnostic),
            "diagnostic": state.diagnostic,
        },
    )

    accepted = 0
    generation_failures = 0
    best_rank = state.rank
    turns_used = 0

    for turn in range(1, max_edits + 1):
        if state.ok:
            break

        turns_used = turn
        before_text = (ws / "Solution.lean").read_text()
        before_state = state

        started = time.time()

        try:
            reply = provider.generate_structured(
                LLMRequest(
                    make_prompt(case, before_text, state, turn),
                    stage="proof-gym-edit",
                    task_class="",
                ),
                schema=EDIT_SCHEMA,
            )
        except Exception as exc:
            generation_failures += 1

            append_jsonl(
                root / "attempts.jsonl",
                {
                    "event": "GENERATION_FAILURE",
                    "case_id": case.case_id,
                    "level": case.level,
                    "turn": turn,
                    "exception_type": type(exc).__name__,
                    "exception": bounded(str(exc), 1500),
                    "rank_before": state.rank,
                    "rank_after": state.rank,
                    "progress": False,
                    "wall_seconds": round(time.time() - started, 3),
                },
            )
            continue

        payload = reply.get("structured") if isinstance(reply, dict) else None
        replacement = (
            payload.get("replacement")
            if isinstance(payload, dict)
            else None
        )

        meta = response_meta(reply)

        if not isinstance(replacement, str):
            generation_failures += 1
            append_jsonl(
                root / "attempts.jsonl",
                {
                    "event": "INVALID_REPLACEMENT",
                    "case_id": case.case_id,
                    "level": case.level,
                    "turn": turn,
                    "progress": False,
                    **meta,
                },
            )
            continue

        if len(replacement) > MAX_REPLACEMENT_CHARS:
            append_jsonl(
                root / "attempts.jsonl",
                {
                    "event": "OVERSIZE_REPLACEMENT",
                    "case_id": case.case_id,
                    "level": case.level,
                    "turn": turn,
                    "replacement_chars": len(replacement),
                    "maximum_chars": MAX_REPLACEMENT_CHARS,
                    "rank_before": state.rank,
                    "rank_after": state.rank,
                    "progress": False,
                    **meta,
                },
            )
            continue

        (ws / f"Solution.proposed{turn}.lean").write_text(replacement)

        if replacement == before_text:
            append_jsonl(
                root / "attempts.jsonl",
                {
                    "event": "IDENTICAL_REPLACEMENT",
                    "case_id": case.case_id,
                    "level": case.level,
                    "turn": turn,
                    "rank_before": state.rank,
                    "rank_after": state.rank,
                    "progress": False,
                    **meta,
                },
            )
            continue

        if FORBIDDEN.search(replacement):
            append_jsonl(
                root / "attempts.jsonl",
                {
                    "event": "FORBIDDEN_REPLACEMENT",
                    "case_id": case.case_id,
                    "level": case.level,
                    "turn": turn,
                    "rank_before": state.rank,
                    "rank_after": state.rank,
                    "progress": False,
                    **meta,
                },
            )
            continue

        accepted += 1
        (ws / "Solution.lean").write_text(replacement)
        (ws / f"Solution.turn{turn}.lean").write_text(replacement)

        state = validate(lean, ws, case)

        progress = state.rank > best_rank
        best_rank = max(best_rank, state.rank)

        append_jsonl(
            root / "attempts.jsonl",
            {
                "event": "EDIT_VALIDATED",
                "case_id": case.case_id,
                "level": case.level,
                "turn": turn,
                "code_before": before_state.code,
                "code_after": state.code,
                "rank_before": before_state.rank,
                "rank_after": state.rank,
                "progress": progress,
                "diagnostic_fingerprint": fp(state.diagnostic),
                "diagnostic": state.diagnostic,
                "candidate_sha256": sha(replacement),
                "wall_seconds": round(time.time() - started, 3),
                **meta,
            },
        )

    result = {
        "case_id": case.case_id,
        "level": case.level,
        "status": "PASS" if state.ok else "FAIL",
        "final_code": state.code,
        "final_rank": state.rank,
        "turns_used": turns_used,
        "accepted_edits": accepted,
        "generation_failures": generation_failures,
        "theorem": f"Gym.{case.theorem}",
    }

    (ws / "result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )

    append_jsonl(root / "cases.jsonl", result)
    return result


def default_root() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path.home() / "proofbench-results" / f"proof-gym-{stamp}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--levels", default="L1,L2,L3")
    ap.add_argument("--cases-per-level", type=int, default=5)
    ap.add_argument("--max-edits", type=int, default=4)
    ap.add_argument("--stop-below", type=int, default=4)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--result-root")

    args = ap.parse_args(argv)

    levels = tuple(
        x.strip()
        for x in args.levels.split(",")
        if x.strip()
    )

    if not levels or any(x not in LEVELS for x in levels):
        ap.error("levels must be drawn from L1,L2,L3")

    if args.cases_per_level <= 0:
        ap.error("--cases-per-level must be > 0")

    if args.max_edits <= 0:
        ap.error("--max-edits must be > 0")

    # Hard local-only guarantees.
    os.environ["RESEARCH_ALLOW_PAID_FALLBACK"] = "0"
    os.environ["PROOFBENCH_V9_ENABLE_REMOTE"] = "0"

    root = (
        Path(args.result_root).expanduser().resolve()
        if args.result_root
        else default_root()
    )

    # Deliberately refuse to overwrite an old experiment.
    root.mkdir(parents=True, exist_ok=False)
    (root / "cases").mkdir()
    (root / "attempts.jsonl").touch()
    (root / "cases.jsonl").touch()

    lean = resolve_lean()
    provider = v9_local_provider()

    rng = (
        random.Random(args.seed)
        if args.seed is not None
        else random.SystemRandom()
    )

    (root / "metadata.json").write_text(
        json.dumps(
            {
                "mode": "PUBLIC_LOCAL_ONLY_PROOF_GYM",
                "qualification": False,
                "hidden_cases": False,
                "remote_calls": 0,
                "paid_fallback": False,
                "levels": list(levels),
                "cases_per_level": args.cases_per_level,
                "max_edits": args.max_edits,
                "seed": args.seed,
                "lean": lean,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    print(f"PROOF_GYM_ROOT={root}", flush=True)
    print("MODE=PUBLIC_LOCAL_ONLY", flush=True)
    print("REMOTE_CALLS=0", flush=True)

    summaries = {}
    all_results = []
    stopped_after = None

    for level in levels:
        results = []

        for number, case in enumerate(
            make_cases(level, args.cases_per_level, rng),
            1,
        ):
            print(
                f"CASE_START level={level} "
                f"case={number}/{args.cases_per_level} "
                f"id={case.case_id}",
                flush=True,
            )

            result = run_case(
                provider,
                lean,
                root,
                case,
                args.max_edits,
            )

            results.append(result)
            all_results.append(result)

            print(
                f"CASE_END level={level} "
                f"id={case.case_id} "
                f"status={result['status']} "
                f"code={result['final_code']} "
                f"edits={result['accepted_edits']} "
                f"generation_failures={result['generation_failures']}",
                flush=True,
            )

        passed = sum(r["status"] == "PASS" for r in results)

        successful_edit_counts = [
            r["accepted_edits"]
            for r in results
            if r["status"] == "PASS"
        ]

        summaries[level] = {
            "passed": passed,
            "total": len(results),
            "pass_rate": passed / len(results),
            "generation_failures": sum(
                r["generation_failures"] for r in results
            ),
            "median_accepted_edits_on_pass": (
                statistics.median(successful_edit_counts)
                if successful_edit_counts
                else None
            ),
        }

        print(
            f"LEVEL_RESULT {level} "
            f"{passed}/{len(results)} "
            f"generation_failures="
            f"{summaries[level]['generation_failures']}",
            flush=True,
        )

        if args.stop_below > 0 and passed < args.stop_below:
            stopped_after = level
            print(
                f"FRONTIER_STOP level={level} "
                f"passed={passed} "
                f"threshold={args.stop_below}",
                flush=True,
            )
            break

    frontier = "ABOVE_TESTED_LEVELS"

    for level in levels:
        stat = summaries.get(level)
        if stat is None:
            break
        if stat["passed"] < stat["total"]:
            frontier = level
            break

    summary = {
        "mode": "PUBLIC_LOCAL_ONLY_PROOF_GYM",
        "qualification": False,
        "hidden_cases": False,
        "remote_calls": 0,
        "levels": summaries,
        "frontier": frontier,
        "stopped_after": stopped_after,
        "total_cases_run": len(all_results),
        "total_passed": sum(
            r["status"] == "PASS"
            for r in all_results
        ),
    }

    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )

    print("", flush=True)
    print("===== PROOF GYM SCORECARD =====", flush=True)

    for level, stat in summaries.items():
        print(
            f"{level} "
            f"{stat['passed']}/{stat['total']} "
            f"median_edits={stat['median_accepted_edits_on_pass']} "
            f"generation_failures={stat['generation_failures']}",
            flush=True,
        )

    print(f"FRONTIER={frontier}", flush=True)
    print("REMOTE_CALLS=0", flush=True)
    print(f"SUMMARY={root / 'summary.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
