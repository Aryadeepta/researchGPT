import json
import tempfile
import unittest
from pathlib import Path

from tools.proofbench.adapters.additive_basis import AdditiveBasisAdapter
from tools.proofbench.proof_session import ProofSession, SessionStatus


class ProofSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.task = self.root / "task.json"
        self.task.write_text(json.dumps({"task_id":"sample-additive","theorem":"lower bound", "finite_additive_basis":{"n":12,"forbidden":[2],"max_selected":4}}))

    def tearDown(self): self.temp.cleanup()

    def test_session_has_stable_id_notebook_and_verified_lifecycle(self):
        one = ProofSession.from_task(self.root / "out", self.task); two = ProofSession.from_task(self.root / "out", self.task)
        self.assertEqual(one.session_id, two.session_id)
        one.start(allow_closer=False)
        self.assertEqual(one.status, SessionStatus.ACTIONABLE_HANDOFF)
        self.assertGreaterEqual(one.metrics["verified_lemmas"], 5)
        self.assertTrue((one.path / "proof_notebook.md").is_file())
        self.assertTrue((one.path / "handoff" / "residual.cnf").is_file())
        self.assertTrue(all(x.verification_status == "VERIFIED" for x in one.lemmas))
        self.assertEqual(len(one.dag.nodes), len(one.lemmas))

    def test_invalid_human_hint_is_rejected_and_resume_reuses_checkpoint(self):
        session = ProofSession.from_task(self.root / "out", self.task).start(allow_closer=False)
        before = session.session_id
        bad = self.root / "bad.json"; bad.write_text(json.dumps({"type":"mathematical_hint", "statement":"invented claim"}))
        reopened = ProofSession.open(session.path); self.assertIsNone(reopened.import_human(bad)); self.assertEqual(reopened.session_id, before)
        good = self.root / "good.json"; good.write_text(json.dumps({"type":"mathematical_hint", "statement":session.lemmas[-1].statement}))
        resumed = ProofSession.open(session.path).resume(good)
        self.assertEqual(resumed.session_id, before); self.assertEqual(resumed.status, SessionStatus.VERIFIED)
        self.assertEqual(resumed.human_interventions[-1]["verification_status"], "VERIFIED")
        self.assertGreaterEqual(resumed.metrics["human_interventions"], 2)

    def test_unrelated_boolean_problem_generates_two_verified_lemmas(self):
        task = self.root / "boolean.json"
        # a and (a -> b), expressed as clauses, force a and b.
        task.write_text(json.dumps({"task_id":"implications", "theorem":"b", "boolean_implication_system":{"variables":["a","b"],"clauses":[[1],[-1,2]]}}))
        session = ProofSession.from_task(self.root / "boolean-out", task).start()
        self.assertEqual(session.status, SessionStatus.VERIFIED)
        self.assertGreaterEqual(session.metrics["verified_lemmas"], 2)

    def test_generic_sources_contain_no_demo_constants(self):
        source = (Path(__file__).parents[1] / "tools/proofbench/proof_session.py").read_text().lower()
        for forbidden in ("blind-a", "n = 20", "expected optimum 7", "{2,9,19}"):
            self.assertNotIn(forbidden, source)

    def test_generic_proof_modules_never_import_the_trial_module(self):
        proofbench = Path(__file__).parents[1] / "tools/proofbench"
        forbidden_import = "src." + "mvp_erdos791_trial"
        for source in (proofbench / "proof_session.py", proofbench / "proof_semantic_adapter.py", proofbench / "proof_engineering.py"):
            self.assertNotIn(forbidden_import, source.read_text())

    def test_two_domain_adapters_are_consumed_without_trial_imports(self):
        additive = AdditiveBasisAdapter({"n": 8, "forbidden": [2], "max_selected": 3})
        self.assertIn("def admissible", additive.render_semantics())
        self.assertIn("0", additive.build_constraint_model("toy").variables["x_0"])
        # The generated toy adapter exercises the other, manifest-based
        # semantic-adapter implementation below through ProofSession.

    def test_fresh_semantic_adapter_checkpoint_and_same_session_resume(self):
        skill=self.root/'skill'; skill.mkdir()
        (skill/'semantics.txt').write_text('finite toy semantics\n')
        (skill/'checker.py').write_text('print("checker metadata only")\n')
        (skill/'formal.py').write_text("""import argparse, json
p=argparse.ArgumentParser(); p.add_argument('--output'); a=p.parse_args()
open(a.output,'w').write('import Init.Tactics\\ntheorem toy : True := by trivial\\n')
json.dump({'theorems':['toy'],'proof_nodes':[{'node_id':'HUMAN','dependencies':[]},{'node_id':'FINAL','dependencies':['HUMAN']}]},open(a.output.rsplit('.',1)[0]+'.json','w'))
""")
        manifest={'adapter_id':'toy.adapter','adapter_version':'1','specification':{'universe':[0,1]},'obligation':{'claim':'toy'},'semantics_artifact':{'path':'semantics.txt'},'executable_checker':{'path':'checker.py'},'formal_artifact_generator':{'path':'formal.py'}}
        mp=skill/'adapter_manifest.json'; mp.write_text(json.dumps(manifest))
        session=ProofSession.from_adapter(self.root/'adapter-out',mp).start()
        self.assertEqual(session.status,SessionStatus.ACTIONABLE_HANDOFF)
        data=json.loads((session.path/'session.json').read_text())
        for field in ('session_schema_version','adapter_hash','specification_hash','obligation_hash','semantics_hash','verified_artifact_hashes','handoff_hash'):
            self.assertIn(field,data)
        before=session.session_id; hint=self.root/'human.md'; hint.write_text('advisory')
        resumed=ProofSession.open(session.path).resume(hint)
        self.assertEqual(resumed.session_id,before); self.assertEqual(resumed.status,SessionStatus.VERIFIED)
        # Adapter and semantics identity are trusted checkpoint inputs, not inferred.
        data['adapter_hash']='0'*64; (session.path/'session.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'RESUME_INTEGRITY_ERROR'): ProofSession.open(session.path)


if __name__ == "__main__": unittest.main()
