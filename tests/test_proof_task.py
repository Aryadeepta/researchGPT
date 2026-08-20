import json
import tempfile
import unittest
from pathlib import Path

from tools.proofbench.proof_orchestrator import ProofOrchestrator, StrategyPortfolio, prove_task
from tools.proofbench.proof_task import ProofTask
from tools.proofbench.proof_gym import resolve_lean


class ProofTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.lean = resolve_lean()
    def tearDown(self): self.temp.cleanup()
    def task(self, declaration, expected):
        return ProofTask("generic", "generic_theorem", declaration, expected, provenance="unit")
    def test_generic_context_application_is_lean_verified(self):
        task = self.task("(P Q R : Prop) (hPQ : P → Q) (hQR : Q → R) : P → R", "∀ (P Q R : Prop), (P → Q) → (Q → R) → P → R")
        result = ProofOrchestrator(self.lean, self.root / "ws", self.root / "out").solve(task)
        self.assertTrue(result["pass"]); self.assertTrue(result["artifact_sha256"])
        self.assertIn("DECOMPOSITION", (self.root / "out" / "events.jsonl").read_text())
    def test_cli_json_task_and_progress_are_observational(self):
        payload = {"task_id":"cli", "theorem":"cli_t", "declaration":"(P : Prop) : P → P", "expected_type":"∀ (P : Prop), P → P"}
        path = self.root / "task.json"; path.write_text(json.dumps(payload))
        self.assertEqual(prove_task(path, result_root=self.root / "a"), 0)
        self.assertEqual(prove_task(path, result_root=self.root / "b", progress=True), 0)
        self.assertEqual(json.loads((self.root / "a" / "result.json").read_text())["prefix"], json.loads((self.root / "b" / "result.json").read_text())["prefix"])
    def test_registry_has_stable_generic_ids_without_benchmark_ids(self):
        ids = [item.strategy_id for item in StrategyPortfolio.registry]
        self.assertIn("context.assumption", ids); self.assertIn("decompose.constructor", ids)
        joined = " ".join(ids)
        self.assertNotRegex(joined, r"L[123]-\d{3}|target_")


if __name__ == "__main__": unittest.main()
