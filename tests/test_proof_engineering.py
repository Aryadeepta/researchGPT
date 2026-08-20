import tempfile
import unittest
from pathlib import Path

from tools.proofbench.proof_engineering import (
    BoundedToolSupervisor, Capability, ObligationClass, ProofCertificate,
    ProofDag, ProofDagNode, available_capabilities, capability_context,
    classify_obligation, deterministic_partition, recovery_stages, split_partition,
)


class ProofEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_manifest_is_explicit_and_only_advertises_available_tools(self):
        manifest = available_capabilities(self.root, "/not/a/lean")
        ids = {x.stable_id for x in manifest}
        self.assertIn("python_exec", ids); self.assertNotIn("lean_check", ids)
        self.assertIn("available_capabilities", capability_context(manifest))

    def test_classifier_and_recovery_distinguish_resource_failure(self):
        a = classify_obligation("Finset.all", metadata={"candidate_count": 12000, "enumerative_lower_bound": True}, diagnostics="LEAN_TIMEOUT")
        self.assertEqual(a.obligation_class, ObligationClass.ENUMERATIVE_LOWER_BOUND.value)
        stages = recovery_stages(a, available_capabilities(self.root, "/not/a/lean"), ["direct-lean"])
        self.assertEqual(stages[0]["status"], "SKIPPED")
        self.assertTrue(any(x["stage"] == "computational-scout" and x["status"] == "APPLICABLE" for x in stages))

    def test_bounded_python_is_persisted_and_untrusted(self):
        caps = [Capability("python_exec", "argv", [], [], {"filesystem":"workspace"}, {"timeout_seconds":5,"max_output_bytes":1024})]
        s = BoundedToolSupervisor(self.root, self.root / "trace", caps)
        record = s.execute("python_exec", ["python3", "-c", "print('candidate true')"])
        self.assertEqual(record.exit_status, 0); self.assertTrue(Path(record.stdout_path).is_file())
        self.assertFalse(hasattr(record, "proof"))
        with self.assertRaises(ValueError): s.execute("lean_check", ["lean", "X.lean"])
        with self.assertRaises(ValueError): s.execute("python_exec", ["python", "-c", "pass"])

    def test_partition_is_complete_and_child_failure_blocks_parent(self):
        parts = deterministic_partition(11, 3)
        self.assertEqual(sum(x["end"]-x["start"] for x in parts), 11)
        self.assertEqual(len(split_partition(parts[0])), 2)
        dag = ProofDag(); dag.add(ProofDagNode("c0", "P0", "0")); dag.add(ProofDagNode("c1", "P1", "1")); dag.add(ProofDagNode("parent", "P", "p", ["c0","c1"]))
        self.assertTrue(dag.verify("c0", verifier="Lean", artifact_hash="a"))
        self.assertFalse(dag.verify("parent", verifier="Lean", artifact_hash="p"))
        self.assertTrue(dag.verify("c1", verifier="Lean", artifact_hash="b")); self.assertTrue(dag.verify("parent", verifier="Lean", artifact_hash="p"))

    def test_certificate_is_not_verified_by_construction(self):
        path = self.root / "certificate.json"; path.write_text("{}")
        c = ProofCertificate("finite-case-table", "python", "claim", str(path), "hash", "checker", "1")
        self.assertEqual(c.verification_result, "UNVERIFIED")

if __name__ == "__main__": unittest.main()
