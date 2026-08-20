import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.proofbench.kimina_specialist import KiminaMicroProofSolver, extract_candidates, public_kimina_prompt
from tools.proofbench.proof_gym import GymCase, resolve_lean
from tools.proofbench.proof_orchestrator import LocalNextTacticSolver, ProofOrchestrator, ProofPrefix


class FakeRun:
    def __init__(self, output="", rc=0, error=None):
        self.output, self.rc, self.error, self.calls, self.command = output, rc, error, 0, None
    def __call__(self, command, **kwargs):
        self.calls += 1; self.command = command
        if self.error:
            raise self.error
        return type("CP", (), {"returncode": self.rc, "stdout": self.output})()


class FakeProvider:
    def __init__(self): self.calls = 0
    def generate_structured(self, *args, **kwargs):
        self.calls += 1
        return {"structured": {"tactic": "rfl"}, "model": "fake-qwen"}


class KiminaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.lean = resolve_lean(); self.model = self.root / "model.gguf"; self.binary = self.root / "llama-cli"
        self.model.write_bytes(b"fake"); self.binary.write_text("fake")
    def tearDown(self): self.temp.cleanup()
    def case(self, declaration, expected): return GymCase("Lx", "case", "target", declaration, expected, "test")
    def solver(self, output="", rc=0, error=None):
        runner = FakeRun(output, rc, error)
        return KiminaMicroProofSolver(model=self.model, llama=self.binary, runner=runner), runner
    def solve(self, case, kimina=None):
        return ProofOrchestrator(self.lean, self.root / "ws", self.root / "out", kimina=kimina).solve(case)

    def test_disabled_never_invoked_and_deterministic_success_skips_kimina(self):
        self.solve(self.case("(x : Nat) : x = x", "∀ (x : Nat), x = x"))
        solver, runner = self.solver("rfl")
        result = self.solve(self.case("(x : Nat) : x = x", "∀ (x : Nat), x = x"), solver)
        self.assertTrue(result["pass"]); self.assertEqual(runner.calls, 0)

    def test_generic_qwen_remains_independent_of_kimina_and_remote(self):
        provider = FakeProvider()
        case = self.case("(x : Nat) : x = x", "∀ (x : Nat), x = x")
        with patch("tools.proofbench.proof_orchestrator.deterministic_candidates", return_value=[]):
            result = ProofOrchestrator(self.lean, self.root / "qws", self.root / "qout",
                                       solver=LocalNextTacticSolver(provider)).solve(case)
        self.assertTrue(result["pass"]); self.assertEqual(provider.calls, 3)
        self.assertNotIn("REMOTE_PLAN_RECEIVED", (self.root / "qout" / "events.jsonl").read_text())

    def test_dead_end_invokes_and_partial_intro_is_valid_transition(self):
        solver, runner = self.solver("</think>\n```lean\nexact Or.inl h\n```")
        case = self.case("(P Q : Prop) (h : P) : P ∨ Q", "∀ (P Q : Prop), P → P ∨ Q")
        with patch("tools.proofbench.proof_orchestrator.deterministic_candidates", return_value=[]):
            result = self.solve(case, solver)
        self.assertGreaterEqual(runner.calls, 1)
        self.assertTrue(result["pass"])
        attempts = (self.root / "out" / "attempts.jsonl").read_text()
        self.assertIn('"source": "kimina"', attempts)
        self.assertIn("FINAL_PASS", attempts)
        # Direct strict probing confirms a partial Kimina-like prefix is a
        # valid successor, not incorrectly treated as a complete theorem.
        orch = ProofOrchestrator(self.lean, self.root / "partial", self.root / "partial-out")
        outcome, snap, _ = orch.extractor.probe(self.case("(P Q : Prop) : P → Q → P", "∀ (P Q : Prop), P → Q → P"), ProofPrefix(["intro h"]))
        self.assertEqual(outcome.value, "VALID_NEW_GOAL_STATE"); self.assertIsNotNone(snap)

    def test_extraction_prefers_formal_boundary_and_supports_chunks(self):
        out = "intro bad\n</think>\n```lean\nintro h\nexact h\n```\n"
        candidates = extract_candidates(out)
        self.assertIn("intro h\nexact h", candidates); self.assertNotIn("intro bad", candidates)
        self.assertIn("rw [h]", extract_candidates("[End thinking]\nrw [h]"))
        self.assertIn("intro h\nexact h", extract_candidates("intro h\nexact h"))

    def test_echoed_prompt_unclosed_thinking_and_diagnostics_never_become_candidates(self):
        prompt = "Current exact Lean goal/context:\nP Q : Prop\nh : P\n⊢ Q\n"
        solver, _ = self.solver(prompt + "<think> intro h")
        generation = solver.generate(prompt)
        self.assertEqual(generation.formal_region_classification, "REASONING_TRUNCATED")
        self.assertEqual(extract_candidates(generation.output), [])
        self.assertEqual(extract_candidates("Tactic `introN` failed: no binders\nError: bad\nunknown tactic foo"), [])

    def test_completed_reasoning_extracts_formal_text_and_prompt_marks_prefix_executed(self):
        self.assertIn("exact h", extract_candidates("<think>reason</think>\nexact h"))
        prompt = public_kimina_prompt(declaration="theorem t : P", goal="h : P\n⊢ P", prefix="intro h", rejected=[])
        self.assertIn("ALREADY EXECUTED", prompt)
        self.assertLess(prompt.index("Current exact Lean goal/context"), prompt.index("Original theorem declaration"))

    def test_prose_and_invalid_candidate_need_lean_verification(self):
        self.assertEqual(extract_candidates("This is a wonderful proof."), [])
        solver, _ = self.solver("</think>\nexact nope")
        result = self.solve(self.case("(P : Prop) : P", "∀ (P : Prop), P"), solver)
        self.assertFalse(result["pass"])
        self.assertIn("LEAN_SYNTAX_OR_TACTIC_FAILURE", (self.root / "out" / "attempts.jsonl").read_text())

    def test_generation_failures_are_distinct_and_bounded(self):
        for output, rc, error, expected in [
            ("", 0, None, "OUTPUT_EMPTY"), ("bad", 7, None, "PROCESS_FAILURE"),
            ("", 0, subprocess.TimeoutExpired(["x"], 1), "TIMEOUT"),
        ]:
            solver, _ = self.solver(output, rc, error); generation = solver.generate("public")
            self.assertEqual(generation.status, expected)
        missing = KiminaMicroProofSolver(model=self.root / "none", llama=self.binary)
        self.assertEqual(missing.generate("public").status, "MISSING_MODEL")

    def test_prompt_blocks_private_markers_and_invocation_has_required_flags(self):
        with self.assertRaises(ValueError):
            public_kimina_prompt(declaration="theorem x : True", goal="⊢ True", prefix="", rejected=[{"diagnostic": "hidden secret"}])
        solver, runner = self.solver("rfl"); solver.generate("public")
        self.assertEqual(runner.calls, 1)
        self.assertIn("-st", runner.command); self.assertIn("--simple-io", runner.command)
        self.assertIn("--skip-chat-parsing", runner.command); self.assertIn("-rea", runner.command)
        self.assertIn("--no-display-prompt", runner.command)
        self.assertNotIn("--grammar", runner.command)


if __name__ == "__main__": unittest.main()
