import tempfile, unittest
from pathlib import Path
from tools.proofbench.proof_engine import *

class EngineTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(); self.w=Path(self.t.name); (self.w/'solution.py').write_text('x = 0\n'); (self.w/'Solution.lean').write_text('theorem x : True := by trivial\n'); self.e=ProofEngine(self.w,public_python=lambda p:(p.read_text()=='x = 1\n','bad'),lean='/bin/true',theorem_shape=lambda p:(True,''),axiom_integrity=lambda p:(True,''))
 def tearDown(self):self.t.cleanup()
 def patch(self,n,old,new): return self.e.execute({'tool':'patch','file':n,'sha256':sha(old.encode()),'diff':f'--- {n}\n+++ {n}\n@@ -1 +1 @@\n-{old}+{new}'})
 def test_one_turn_and_shell_rejected(self): self.assertEqual(self.e.execute({'tool':'shell'}).code,'TOOL_REJECTED'); self.assertEqual(self.e.actions,1)
 def test_patch_guards(self):
  self.assertEqual(self.e.execute({'tool':'patch','file':'TASK.md','sha256':'x','diff':''}).code,'PATH_REJECTED'); self.assertEqual(self.e.execute({'tool':'patch','file':'solution.py','sha256':'x','diff':''}).code,'STALE_PATCH'); self.assertEqual(self.e.execute({'tool':'patch','file':'solution.py','sha256':sha(b'x = 0\n'),'diff':'+'*6001}).code,'PATCH_TOO_LARGE')
 def test_malformed_unchanged(self):
  before=(self.w/'solution.py').read_text(); self.assertEqual(self.e.execute({'tool':'patch','file':'solution.py','sha256':sha(before.encode()),'diff':'bad'}).code,'MALFORMED_PATCH'); self.assertEqual((self.w/'solution.py').read_text(),before)
 def test_gates_freeze_and_best(self):
  self.assertEqual(self.e.check_lean().code,'PYTHON_GATE'); old='x = 0\n'; self.assertTrue(self.patch('solution.py',old,'x = 1\n').ok); self.assertTrue(self.e.check_python().ok); self.assertIn('solution.py',self.e.frozen); self.assertEqual(self.e.execute({'tool':'patch','file':'solution.py','sha256':sha(b'x = 1\n'),'diff':'x'}).code,'FILE_FROZEN'); self.assertTrue(self.e.check_lean().ok); self.assertEqual(self.e.phase,Phase.PUBLIC_COMPLETE)
 def test_integrity_and_plateau(self):
  (self.w/'Solution.lean').write_text('by sorry\n'); self.e.py_state=PyState.PUBLIC_PASS; self.assertEqual(self.e.check_lean().code,'PROOF_INTEGRITY_FAILURE'); [self.e.execute({'tool':'status'}) for _ in range(8)]; self.assertTrue(self.e.plateau())
 def test_rank_not_sha(self): self.assertEqual(self.e.rank(),0); self.e.py_state=PyState.COMPILE_FAILURE; a=self.e.rank(); (self.w/'solution.py').write_text('other\n'); self.assertEqual(a,self.e.rank())
 def test_best_survives_regression_and_policy(self):
  self.e.py_state=PyState.PUBLIC_PASS; self.e.checkpoint('best'); best=self.e.best; self.e.py_state=PyState.COMPILE_FAILURE; self.e.checkpoint('regression'); self.assertEqual(self.e.best,best); p=EscalationPolicy(); self.assertFalse(p.allow('luna',False)); self.assertTrue(p.allow('luna',True)); p.record('luna'); self.assertFalse(p.allow('luna',True)); self.assertTrue(p.allow('terra',True)); p.record('terra'); self.assertFalse(p.allow('terra',True)); self.assertFalse(p.allow('sol',True))
 def test_public_runtime_failure_cannot_enter_lean(self): self.assertFalse(self.e.check_python().ok); self.assertEqual(self.e.phase,Phase.PYTHON); self.assertEqual(self.e.check_lean().code,'PYTHON_GATE')
 def test_compile_is_not_proof_verification(self):
  self.e.py_state=PyState.PUBLIC_PASS; self.e.theorem_shape=lambda p:(False,'wrong theorem'); self.assertEqual(self.e.check_lean().code,'THEOREM_SHAPE_FAILURE'); self.e.theorem_shape=lambda p:(True,''); self.e.axiom_integrity=lambda p:(False,'sorryAx'); self.assertEqual(self.e.check_lean().code,'AXIOM_INTEGRITY_FAILURE')
 def test_phase_restricts_candidate_file(self):
  self.assertEqual(self.e.execute({'tool':'patch','file':'Solution.lean','sha256':sha((self.w/'Solution.lean').read_bytes()),'diff':'--- Solution.lean\n+++ Solution.lean\n@@ -1 +1 @@\n-x\n+y\n'}).code,'PHASE_FILE_REJECTED')
