"""Isolated, bounded benchmarks for orchestration semantics.

This module intentionally has no production-runtime imports.  In particular,
the verification-oblivious mode is incapable of issuing a ResearchPackage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class BenchmarkMode(str, Enum):
    FULL = "FULL"
    FIXED_SEQUENTIAL = "FIXED_SEQUENTIAL"
    VERIFICATION_OBLIVIOUS_BASELINE = "VERIFICATION_OBLIVIOUS_BASELINE"


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    family: str
    objective: str
    expected_verifiers: tuple[str, ...]


def benchmark_schema():
    return {"schema_version": 1, "modes": [mode.value for mode in BenchmarkMode],
            "metrics": ["task_completion", "verified_task_completion", "formal_proof_success",
                        "executable_test_success", "verifier_failures", "replans", "dag_nodes_executed",
                        "model_tool_calls", "inference_resource_cost", "wall_clock_seconds",
                        "verified_claim_coverage", "provenance_completeness"]}


def benchmark_record(task, mode, metrics=None):
    mode = BenchmarkMode(mode)
    return {"schema_version": 1, "kind": "orchestrator_benchmark_record", "task": asdict(task),
            "mode": mode.value,
            "trust_status": "UNTRUSTED_BENCHMARK_ONLY" if mode is BenchmarkMode.VERIFICATION_OBLIVIOUS_BASELINE else "VERIFIER_GATED",
            "normal_research_package_allowed": mode is not BenchmarkMode.VERIFICATION_OBLIVIOUS_BASELINE,
            "metrics": metrics or {}}


def public_demo_seed():
    return {"id": "bounded-shortest-path-invariant-v1", "objective": "Compare two bounded shortest-path implementations and formally state a simple path invariant.",
            "expected_workflow": ["question_refinement", "algorithmic_hypothesis", "implementation",
                                  "test_failure", "repair_replan", "formal_decomposition", "lean_verification",
                                  "claim_to_evidence_report"],
            "success_criteria": ["executable test evidence is recorded", "formal claim has Lean verifier evidence",
                                 "failed attempt remains in lineage"],
            "note": "Seed/config only: no model output or successful run is fabricated."}
