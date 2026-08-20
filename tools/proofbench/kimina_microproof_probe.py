#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from tools.proofbench.kimina_specialist import clean_block, extract_candidates

ROOT = Path(__file__).resolve().parents[2]

MODEL = Path(
    os.environ.get(
        "KIMINA_MODEL",
        str(
            Path.home()
            / ".cache/researchgpt-models/kimina"
            / "Kimina-Prover-RL-1.7B.Q4_K_M.gguf"
        ),
    )
)

LLAMA = Path(
    os.environ.get(
        "LLAMA_CPP_CLI_BIN",
        "/home/aryad/llama.cpp-direct/build/bin/llama-cli",
    )
)

LEAN = Path(os.environ.get("LEAN_BIN", str(Path.home() / ".elan/bin/lean")))

CASES = [
    {
        "id": "reflexivity",
        "decl": "theorem kimina_probe_reflexivity (x : Nat) : x = x := by",
    },
    {
        "id": "composition",
        "decl": (
            "theorem kimina_probe_composition "
            "(P Q R : Prop) "
            "(hPQ : P → Q) "
            "(hQR : Q → R) : P → R := by"
        ),
    },
    {
        "id": "rewrite",
        "decl": (
            "theorem kimina_probe_rewrite "
            "(a b : Nat) "
            "(h : a = b) : 0 + a = b := by"
        ),
    },
]


def run(cmd, *, cwd=None, timeout=120):
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def kimina_prompt(case):
    return f"""Solve this Lean 4 theorem.

Think briefly, then provide a valid Lean 4 proof.
The final formal proof must complete the theorem below.

import Std
set_option autoImplicit false

{case["decl"]}
"""


def invoke_kimina(case):
    cp = run(
        [
            LLAMA,
            "-m",
            MODEL,
            "-c",
            "2048",
            "-t",
            "2",
            "-n",
            "640",
            "-st",
            "--simple-io",
            "--skip-chat-parsing",
            "-rea",
            "on",
            "--reasoning-budget",
            "96",
            "--temp",
            "0.4",
            "--top-p",
            "0.95",
            "-p",
            kimina_prompt(case),
        ],
        timeout=180,
    )
    return cp.returncode, cp.stdout




def lean_source(case, body):
    indented = "\n".join("  " + x for x in body.splitlines())

    return (
        "import Std\n"
        "set_option autoImplicit false\n\n"
        f'{case["decl"]}\n'
        f"{indented}\n"
    )


def validate(case, body):
    with tempfile.TemporaryDirectory(prefix="kimina-proof-probe-") as td:
        p = Path(td) / "Probe.lean"
        p.write_text(lean_source(case, body), encoding="utf-8")

        cp = run(
            [LEAN, p.name],
            cwd=td,
            timeout=30,
        )

        return cp.returncode == 0, cp.stdout


def main():
    print(f"MODEL={MODEL}")
    print(f"LLAMA={LLAMA}")
    print(f"LEAN={LEAN}")

    if not MODEL.is_file():
        raise SystemExit("MODEL_MISSING")
    if not LLAMA.is_file():
        raise SystemExit("LLAMA_MISSING")
    if not LEAN.is_file():
        raise SystemExit("LEAN_MISSING")

    results = []

    for case in CASES:
        print()
        print("=" * 72)
        print(f"CASE={case['id']}")
        print("=" * 72)

        rc, raw = invoke_kimina(case)

        print(f"KIMINA_RC={rc}")
        print("----- RAW KIMINA OUTPUT -----")
        print(raw)
        print("----- END RAW OUTPUT -----")

        candidates = extract_candidates(raw)

        print(f"EXTRACTED_CANDIDATES={len(candidates)}")

        verified = None

        for i, body in enumerate(candidates, 1):
            ok, diagnostic = validate(case, body)

            print()
            print(f"CANDIDATE_{i}={body!r}")
            print(f"CANDIDATE_{i}_LEAN={'PASS' if ok else 'FAIL'}")

            if not ok:
                compact = diagnostic.strip().replace("\n", " | ")
                print(f"CANDIDATE_{i}_DIAGNOSTIC={compact[:600]}")

            if ok:
                verified = body
                break

        record = {
            "case": case["id"],
            "kimina_rc": rc,
            "candidate_count": len(candidates),
            "verified": verified is not None,
            "verified_body": verified,
        }
        results.append(record)

        print()
        print("CASE_RESULT=" + json.dumps(record, sort_keys=True))

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for r in results:
        print(
            f"{r['case']:20s} "
            f"verified={r['verified']} "
            f"candidates={r['candidate_count']} "
            f"body={r['verified_body']!r}"
        )

    passed = sum(r["verified"] for r in results)

    print()
    print(f"VERIFIED={passed}/{len(results)}")
    print("KIMINA_MICROPROOF_PROBE=COMPLETE")


if __name__ == "__main__":
    main()
