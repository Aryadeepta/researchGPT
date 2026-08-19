import tempfile
import unittest
from pathlib import Path

from src.research_state import create_run_state, record_verification
from src.run_inspection import export_graph, graph_mermaid, provenance_manifest, replay_dry_run
from src.storage import LocalArtifactStore


class RunInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = LocalArtifactStore(self.temp.name)
        self.state = create_run_state("inspect", "bounded objective")
        self.state["dag"]["nodes"]["question_discovery"].update({"status": "COMPLETED", "artifacts": ["research_spec.json"]})
        self.state["dag"]["nodes"]["evidence_discovery"].update({"status": "COMPLETED", "verification_state": "VERIFICATION_FAILED", "verification_evidence": ["validation/report.json"], "replans": ["evidence_discovery_repair_1"]})
        sample = Path(self.temp.name) / "sample.json"; sample.write_text("{}")
        self.store.put_artifact("inspect", sample, "research_spec.json", "test")
        self.store.atomic_update_state("inspect", self.state)

    def tearDown(self): self.temp.cleanup()

    def test_typed_graph_is_read_only_and_contains_edges_and_hashes(self):
        before = self.store.load_state("inspect")
        graph = export_graph(before, self.store.load_manifest("inspect"))
        question = next(node for node in graph["nodes"] if node["node_id"] == "question_discovery")
        evidence = next(node for node in graph["nodes"] if node["node_id"] == "evidence_discovery")
        self.assertTrue(question["generated_artifacts"][0]["sha256"])
        self.assertIn("evidence_discovery", question["downstream"])
        self.assertEqual(evidence["verification_state"], "VERIFICATION_FAILED")
        self.assertEqual(before, self.store.load_state("inspect"))
        self.assertIn("flowchart TD", graph_mermaid(graph))

    def test_provenance_and_dry_replay_detect_mutated_artifacts_without_execution(self):
        manifest = self.store.load_manifest("inspect")
        provenance = provenance_manifest(self.state, manifest)
        self.assertEqual(provenance["kind"], "replayable_execution_manifest")
        good = replay_dry_run(self.store, "inspect")
        self.assertEqual(good["status"], "OK"); self.assertFalse(good["model_execution"])
        Path(self.store.get_artifact_path("inspect", "research_spec.json")).write_text("mutated")
        bad = replay_dry_run(self.store, "inspect")
        self.assertEqual(bad["status"], "INVALID_PROVENANCE")

    def test_generation_is_untrusted_and_verification_history_is_append_only(self):
        node = self.state["dag"]["nodes"]["question_discovery"]
        self.assertEqual(node["verification_state"], "GENERATED_UNVERIFIED")
        record_verification(self.state, "question_discovery", "VERIFICATION_FAILED", ["failure.json"], "test", "failed")
        record_verification(self.state, "question_discovery", "REPLAN_REQUESTED", reason="repair")
        self.assertEqual(len(node["verification_history"]), 2)
        self.assertEqual(node["verification_history"][0]["status"], "VERIFICATION_FAILED")
