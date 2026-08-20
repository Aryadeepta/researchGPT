import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.research_package import stable_hash, verify_research_package
from src.storage import LocalArtifactStore


class ResearchPackageProofEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _durable_formal_package(self, run_id="formal-evidence"):
        store = LocalArtifactStore(self.root / "durable")
        scratch = Path(tempfile.mkdtemp(prefix="rgpt-proof-scratch-"))
        source = scratch / "ExactBasis.lean"
        source.write_text("theorem x : True := by native_decide\n")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        store.put_artifact(run_id, source, "formal/ExactBasis.lean", "test")
        for name, contents in (
            ("lean_stdout.txt", "axioms x : [native_decide]\n"),
            ("lean_stderr.txt", ""),
            ("lean_axioms.txt", "axioms x : [native_decide]\n"),
        ):
            local = scratch / name
            local.write_text(contents)
            store.put_artifact(run_id, local, f"formal/{name}", "Lean")
        metadata = {
            "verifier": "Lean", "verifier_trust": "NATIVE_DECIDE",
            "command": ["lean", "formal/ExactBasis.lean"],
            "executable": "lean", "executable_version": "Lean 4", "exit_code": 0,
            "input_sha256": source_hash, "stdout_artifact": "formal/lean_stdout.txt",
            "stderr_artifact": "formal/lean_stderr.txt", "axioms_artifact": "formal/lean_axioms.txt",
            "claim_obligation_sha256": "obligation",
        }
        local = scratch / "metadata.json"
        local.write_text(json.dumps(metadata))
        store.put_artifact(run_id, local, "formal/lean_verification.json", "Lean")
        claim = {
            "claim_id": "exact", "claim_class": "bounded_correctness",
            "evidence_modalities": ["formal_proof"],
            "formal_evidence": {
                "artifact_path": "formal/ExactBasis.lean", "artifact_sha256": source_hash,
                "verifier_metadata_artifact": "formal/lean_verification.json",
            },
        }
        package = {
            "package_id": "RP-formal-evidence", "version": 1,
            "claim_evidence_ledger": {"claims": [claim]},
            "artifact_manifest": store.load_manifest(run_id),
            "research_readiness_report": {"ready": True},
        }
        package["package_hash"] = stable_hash(package)
        package_path = store.run_root(run_id) / "packages" / "v1" / "research_package.json"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        package_path.write_text(json.dumps(package))
        return store, run_id, scratch, source_hash

    def test_durable_formal_source_survives_scratch_deletion(self):
        store, run_id, scratch, source_hash = self._durable_formal_package()
        shutil.rmtree(scratch)
        durable = Path(store.get_artifact_path(run_id, "formal/ExactBasis.lean"))
        self.assertTrue(durable.is_file())
        self.assertEqual(hashlib.sha256(durable.read_bytes()).hexdigest(), source_hash)
        self.assertEqual(verify_research_package(store, run_id)["status"], "PASS")

    def test_missing_tampered_and_wrong_sha_formal_artifacts_fail(self):
        for mode in ("missing", "tampered", "wrong-sha"):
            with self.subTest(mode=mode):
                store, run_id, scratch, _ = self._durable_formal_package(f"formal-{mode}")
                durable = Path(store.get_artifact_path(run_id, "formal/ExactBasis.lean"))
                if mode == "missing":
                    durable.unlink()
                elif mode == "tampered":
                    durable.write_bytes(durable.read_bytes() + b" ")
                else:
                    package_path = store.run_root(run_id) / "packages" / "v1" / "research_package.json"
                    package = json.loads(package_path.read_text())
                    package["claim_evidence_ledger"]["claims"][0]["formal_evidence"]["artifact_sha256"] = "0" * 64
                    package["package_hash"] = stable_hash({k: v for k, v in package.items() if k != "package_hash"})
                    package_path.write_text(json.dumps(package))
                self.assertEqual(verify_research_package(store, run_id)["status"], "FAIL")
                shutil.rmtree(scratch)

    def test_trust_metadata_and_tmp_claim_reference_are_rejected(self):
        for mode in ("missing-trust", "tmp-reference"):
            with self.subTest(mode=mode):
                store, run_id, scratch, _ = self._durable_formal_package(f"formal-{mode}")
                package_path = store.run_root(run_id) / "packages" / "v1" / "research_package.json"
                package = json.loads(package_path.read_text())
                if mode == "missing-trust":
                    metadata_path = Path(store.get_artifact_path(run_id, "formal/lean_verification.json"))
                    metadata = json.loads(metadata_path.read_text())
                    metadata.pop("verifier_trust")
                    metadata_path.write_text(json.dumps(metadata))
                else:
                    package["claim_evidence_ledger"]["claims"][0]["formal_evidence"]["artifact_path"] = "/tmp/ExactBasis.lean"
                    package["package_hash"] = stable_hash({k: v for k, v in package.items() if k != "package_hash"})
                    package_path.write_text(json.dumps(package))
                self.assertEqual(verify_research_package(store, run_id)["status"], "FAIL")
                shutil.rmtree(scratch)


if __name__ == "__main__":
    unittest.main()
