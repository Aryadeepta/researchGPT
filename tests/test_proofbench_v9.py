import json, os, subprocess, tempfile, unittest
from pathlib import Path
from tools.proofbench import v9_controller as v

class V9Tests(unittest.TestCase):
 def engine(self,p):
  (p/'solution.py').write_text('x=1\n'); (p/'Solution.lean').write_text('theorem x : True := by trivial\n')
  return v.ProofEngine(p,public_python=lambda q:(q.read_text()=='x=2\n','bad'),lean='/bin/true',theorem_shape=lambda q:(True,''),axiom_integrity=lambda q:(True,''))
 def test_demo_public_complete_no_remote(self):
  with tempfile.TemporaryDirectory() as d:
   e,a=v.fake_demo(Path(d)); self.assertEqual(e.phase.value,'PUBLIC_COMPLETE'); self.assertGreaterEqual(len(a),6); self.assertFalse((Path(d)/'escalation.jsonl').exists())
 def test_hidden_commitments_fresh_and_safe(self):
   a,ha=v.hidden_suite('H1'); b,hb=v.hidden_suite('H1'); self.assertNotEqual(ha,hb); record={'commitment_sha256':ha,'case_count':len(a)}; self.assertNotIn('nonce',json.dumps(record))
 def test_hidden_runs_only_after_public(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'solution.py').write_text('x=1\n'); (p/'Solution.lean').write_text('x\n'); e=v.make_engine(p,p,lean='/bin/true'); self.assertEqual(v.qualify_hidden('H1',e,lambda c:True,p),'PUBLIC_GATE'); e.phase=v.Phase.PUBLIC_COMPLETE; self.assertEqual(v.qualify_hidden('H1',e,lambda c:False,p),'HIDDEN_GENERALIZATION_FAILURE'); rec=json.loads((p/'hidden-commitments.jsonl').read_text()); self.assertEqual(set(rec),{'task_id','case_count','commitment_sha256'})
 def test_residual_scrubs_hidden(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'solution.py').write_text('x=1\n'); (p/'Solution.lean').write_text('x\n'); e=v.make_engine(p,p,lean='/bin/true'); self.assertNotIn('hidden',v.residual('H1',e,'hidden marker'))
 def test_status_read_only(self):
  with tempfile.TemporaryDirectory() as d:
   before=set(Path(d).iterdir()); v.status(); self.assertEqual(before,set(Path(d).iterdir()))
 def test_background_alive(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)/'run'; subprocess.run(['python3','tools/proofbench/v9_controller.py','--controller','--result-root',str(r),'--blocking-seconds','.3'],check=True); self.assertTrue((r/'controller.ready').exists())
 def test_self_tests(self): self.assertEqual(v.self_test(),0); self.assertEqual(v.integration_self_test(),0)
 def test_action_protocol_and_public_prompt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'solution.py').write_text('x=1\n'); (p/'Solution.lean').write_text('theorem x : True := by trivial\n'); e=v.make_engine(p,p,lean='/bin/true')
   self.assertIsNone(v.parse_action('{"tool":"shell"}')); self.assertEqual(v.parse_action('{"tool":"status"}')['tool'],'status'); self.assertNotIn('hidden',v.prompt('H1',e).lower())
 def test_escalation_uses_only_same_candidate_patch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'solution.py').write_text('x=1\n'); (p/'Solution.lean').write_text('theorem x : True := by trivial\n'); e=v.make_engine(p,p,lean='/bin/true'); policy=v.EscalationPolicy()
   result=v.escalation_step('H1',e,policy,'bad',p,lambda *args:{'code':'REMOTE_PATCH_REJECTED'})
   self.assertEqual(result.code,'REMOTE_PATCH_REJECTED'); self.assertEqual(policy.calls['luna'],1)
 def test_v9_reducer_and_structured_control(self):
  r=v.V9BoundedIdentityReducer(); text='{"task":"H1","public":"'+'x '*800+'"}'
  self.assertEqual(r.reduce(text),text); self.assertNotEqual(r.reduce(text),'{"task": "H1"}')
  class P:
   def generate(self,*a): raise AssertionError('plain generate forbidden')
  p=v.v9_local_provider(P()); self.assertIsInstance(p.context_reducer,v.V9BoundedIdentityReducer)
 def test_replacement_uses_supervisor_sha_and_engine_patch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); e=self.engine(p)
   class P:
    def generate_structured(self,*a,**k): return {'structured':{'replacement':'x=2\n'}}
   r,code=v.execute_plan('H1',e,{'action':'request_edit'},'',P())
   self.assertEqual(r.code,'PYTHON_PASS'); self.assertEqual((p/'solution.py').read_text(),'x=2\n'); self.assertEqual(e.patch_count,1); self.assertNotIn('sha256',v.edit_prompt('H1',e))
 def test_identical_invalid_and_plateau_are_no_progress(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); e=self.engine(p)
   class Same:
    def generate_structured(self,*a,**k): return {'structured':{'replacement':'x=1\n'}}
   self.assertEqual(v.execute_plan('H1',e,{'action':'request_edit'},'',Same())[1],'LOCAL_EDIT_IDENTICAL'); self.assertEqual(e.patch_count,0)
  h=[{'rank_after':0,'progress':False} for _ in range(8)]; self.assertTrue(v.control_plateau(h)); h[-1]['progress']=True; self.assertFalse(v.control_plateau(h))
 def test_policy_bounds_and_residual_public_material(self):
  p=v.EscalationPolicy(luna=1,terra=1,sol=0); self.assertFalse(p.allow('luna',False)); p.record('luna'); self.assertTrue(p.allow('terra',True)); p.record('terra'); self.assertFalse(p.allow('sol',True))
  with tempfile.TemporaryDirectory() as d:
   q=Path(d); e=self.engine(q); (q/'TASK.md').write_text('public requirement'); (q/'Spec.lean').write_text('public spec')
   out=v.residual('H1',e,'public diagnostic'); self.assertIn('public requirement',out); self.assertIn('public spec',out); self.assertNotIn('hidden',out)
 def test_controller_counts_malformed_turns_and_only_escalates_after_plateau(self):
  class Bad:
   def generate_structured(self,*a,**k): return {'structured':{'not_action':'shell'}}
  calls=[]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   fake_lean=root/'fake-lean'
   fake_lean.write_text(
    '#!/bin/sh\n'
    'if [ "$1" = "-o" ]; then\n'
    '  printf "fake-olean\\n" > "$2"\n'
    'fi\n'
    'exit 0\n'
   )
   fake_lean.chmod(0o755)
   self.assertEqual(v.controller_main(root,provider=Bad(),lean=str(fake_lean),max_actions=8,remote_transport=lambda model,task,engine,diagnostic,root:(calls.append((model,diagnostic)) or {'code':'REMOTE_CODEX_PROCESS_FAILURE'})),0)
   log=(root/'controller.log').read_text(); summary=json.loads((root/'summary.json').read_text())
   self.assertIn('AGENT_CONTROL_PLATEAU H1',log); self.assertEqual(summary['remote_calls'],{'luna':0,'terra':0,'sol':0}); self.assertEqual(calls[0][0],'luna'); self.assertNotIn('hidden',calls[0][1])
 def test_remote_progress_is_rank_based_and_uses_engine_patch(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); e=self.engine(p); (p/'TASK.md').write_text('PUBLIC TASK'); (p/'Spec.lean').write_text('PUBLIC SPEC')
   policy=v.EscalationPolicy(); seen={}
   def transport(model,task,engine,diagnostic,root):
    seen.update({'model':model,'task':task}); return {'code':'REMOTE_CANDIDATE','candidate':'x=2\n','diagnostic':'public'}
   r=v.escalation_step('H1',e,policy,'public',p,transport)
   self.assertEqual(r.code,'REMOTE_PROGRESS'); self.assertEqual(e.patch_count,1); self.assertEqual(seen['model'],'luna')
   self.assertEqual((p/'solution.py').read_text(),'x=2\n')
   rec=json.loads((p/'escalation.jsonl').read_text()); self.assertTrue(rec['patch_applied']); self.assertTrue(rec['progress'])
 def test_changed_candidate_without_public_progress_is_not_progress(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); e=self.engine(p); policy=v.EscalationPolicy()
   r=v.escalation_step('H1',e,policy,'public',p,lambda *a:{'code':'REMOTE_CANDIDATE','candidate':'x=3\n'})
   self.assertEqual(r.code,'REMOTE_VALIDATOR_NO_PROGRESS'); self.assertEqual(policy.calls['luna'],1)
 def test_no_change_and_infrastructure_are_bounded_and_distinct(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); e=self.engine(p); policy=v.EscalationPolicy()
   same=lambda *a:{'code':'REMOTE_CANDIDATE','candidate':'x=1\n'}
   self.assertEqual(v.escalation_step('H1',e,policy,'public',p,same).code,'REMOTE_NO_CHANGE')
   self.assertEqual(policy.calls['luna'],1); self.assertFalse(policy.allow('luna',True))
   policy=v.EscalationPolicy(); fail=lambda *a:{'code':'REMOTE_TIMEOUT','returncode':124,'diagnostic':'bounded tail'}
   self.assertEqual(v.escalation_step('H1',e,policy,'public',p,fail).code,'REMOTE_TIMEOUT')
   self.assertEqual(policy.calls['luna'],0); self.assertEqual(policy.infrastructure_retries['luna'],1)
   self.assertEqual(v.escalation_step('H1',e,policy,'public',p,fail).code,'REMOTE_INFRA_UNAVAILABLE')
   self.assertEqual(policy.calls['luna'],0); self.assertFalse(policy.allow('terra',True))
 def test_remote_workspace_is_public_and_phase_selects_candidate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); e=self.engine(p); (p/'TASK.md').write_text('PUBLIC TASK'); (p/'Spec.lean').write_text('PUBLIC SPEC')
   fake=p/'codex'; capture=p/'capture.json'; fake.write_text("#!/usr/bin/env python3\nimport json,os\nfrom pathlib import Path\nPath(os.environ['CAPTURE']).write_text(json.dumps(sorted(x.name for x in Path('.').iterdir())))\nPath('solution.py').write_text('x=2\\n')\nprint('no sha or diff')\n"); fake.chmod(0o755)
   oldpath=os.environ.get('PATH',''); oldenable=os.environ.get('PROOFBENCH_V9_ENABLE_REMOTE')
   os.environ['PATH']=str(p)+os.pathsep+oldpath; os.environ['CAPTURE']=str(capture); os.environ['PROOFBENCH_V9_ENABLE_REMOTE']='1'
   try:
    out=v.default_remote_transport('luna','H1',e,'PUBLIC DIAGNOSTIC',p)
   finally:
    os.environ['PATH']=oldpath; os.environ.pop('CAPTURE',None)
    if oldenable is None: os.environ.pop('PROOFBENCH_V9_ENABLE_REMOTE',None)
    else: os.environ['PROOFBENCH_V9_ENABLE_REMOTE']=oldenable
   self.assertEqual(set(json.loads(capture.read_text())),{'TASK.md','Spec.lean','DIAGNOSTIC.txt','solution.py'})
   self.assertNotIn('sha256',out); self.assertNotIn('diff',out); self.assertEqual(out['candidate'],'x=2\n')
