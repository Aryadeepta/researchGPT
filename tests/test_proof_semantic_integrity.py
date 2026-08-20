"""Regression guards for the proof-session semantic boundary."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from tools.proofbench.proof_engineering import BooleanConstraint, FiniteBooleanModel, additive_basis_model, dimacs_export, propagate_constraints, readable_deduction_trace, residual_model
from tools.proofbench.proof_session import ProofSession, SessionStatus


class SemanticIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.task=self.root/'blind.json'; self.task.write_text(json.dumps({'task_id':'semantic','theorem':'no basis <= 6','finite_additive_basis':{'n':20,'forbidden':[2,9,19],'max_selected':6}}))
    def tearDown(self): self.tmp.cleanup()
    def session(self, close=False): return ProofSession.from_task(self.root/'out',self.task).start(allow_closer=close)

    def test_semantic_lemmas_are_not_proxies_and_are_inclusive(self):
        s=self.session(); zero=next(x for x in s.lemmas if x.statement=='0 must be selected')
        text=Path(zero.artifact_path).read_text()
        self.assertIn('def admissible',text); self.assertIn('List.range (N + 1)',text); self.assertNotIn('x = true → x = true',text)
        self.assertIn('x_20',s.model.variables); self.assertTrue(any(c.constraint_id=='coverage-20' for c in s.model.constraints))
    def test_coverage_is_structured_and_never_emits_satisfied_target(self):
        s=self.session(); trace={x.get('constraint_id'):x for x in s.trace}
        self.assertNotIn('coverage-4',trace)
        self.assertIn(['x_4','x_5'],trace['coverage-9']['alternatives'])
        self.assertIn(['x_6'],trace['coverage-9']['alternatives']); self.assertIn(['x_8'],trace['coverage-9']['alternatives'])
    def test_residual_is_smaller_and_cnf_is_non_evidence(self):
        model=additive_basis_model(20,{2,9,19},6,'x'); facts,_,_=propagate_constraints(model); reduced=residual_model(model,facts)
        info=dimacs_export(reduced,self.root/'cnf'); self.assertLess(len(reduced.variables),len(model.variables)); self.assertEqual(info['cardinality_encoding'],'sinz-sequential-counter'); self.assertEqual(info['solver_status'],'UNRUN')
    def _sat(self, clauses, assignment):
        """Small deterministic DPLL evaluator, independent of encoder text."""
        def go(values):
            changed=True
            while changed:
                changed=False
                for clause in clauses:
                    live=[x for x in clause if abs(x) not in values]
                    if any(values.get(abs(x)) == (x > 0) for x in clause): continue
                    if not live: return False
                    if len(live)==1: values[abs(live[0])]=live[0]>0; changed=True
            unknown=next((abs(x) for c in clauses for x in c if abs(x) not in values),None)
            return True if unknown is None else go({**values,unknown:False}) or go({**values,unknown:True})
        return go(dict(assignment))
    def _cardinality_cnf(self,n,k,kind='at_most'):
        model=FiniteBooleanModel('cardinality',{f'x{i}':str(i) for i in range(n)},[BooleanConstraint(kind,list(range(1,n+1)),k,'card')])
        info=dimacs_export(model,self.root/f'{kind}-{n}-{k}')
        lines=Path(info['cnf_path']).read_text().splitlines(); return [[int(x) for x in line.split()[:-1]] for line in lines if line and line[0] not in 'cp']
    def test_cardinality_encoding_exhaustive_equivalence(self):
        for n in range(1,8):
            for k in range(n+1):
                for kind in ('at_most','at_least'):
                    clauses=self._cardinality_cnf(n,k,kind)
                    for bits in itertools.product((False,True),repeat=n):
                        sat=self._sat(clauses,{i+1:b for i,b in enumerate(bits)})
                        expected=sum(bits)<=k if kind=='at_most' else sum(bits)>=k
                        self.assertEqual(sat,expected,(n,k,kind,bits))
        # Regression for the independently discovered bad assignment.
        self.assertFalse(self._sat(self._cardinality_cnf(3,1),{1:True,2:True,3:False}))
    def test_structured_human_hint_is_checked_and_used(self):
        s=self.session(); hint=self.root/'hint.json'; hint.write_text(json.dumps({'type':'structured_lemma','statement':'4 or 5','or':[[4],[5]]}))
        s=ProofSession.open(s.path).resume(hint); human=s.human_interventions[-1]; self.assertEqual(human['verification_status'],'VERIFIED'); self.assertEqual(s.status,SessionStatus.VERIFIED); self.assertIn(human['lemma_id'],s.dag.nodes['FINAL'].dependencies)
        source=(s.path/'final_certificate.lean').read_text(); self.assertIn('residual_compose_list',source); self.assertIn('human_residual',source); self.assertIn('noAdmissibleWithHuman',source); self.assertNotIn('final_from_human',source)
    def test_autonomous_dag_is_honest_and_resume_detects_tampering(self):
        autonomous=self.session(close=True); self.assertEqual(autonomous.dag.nodes['FINAL'].dependencies,[])
        self.assertEqual(autonomous.metrics['certificate_type'],'independent-finite-certificate-closer')
        handoff=self.session(); semantic=handoff.path/'Semantics.lean'; semantic.write_text(semantic.read_text()+'\n-- tampered\n')
        with self.assertRaisesRegex(ValueError,'RESUME_INTEGRITY_ERROR'): ProofSession.open(handoff.path).resume(self.root/'none.json')
        clean=self.session(); lemma=Path(clean.lemmas[0].artifact_path); lemma.write_text(lemma.read_text()+'\n-- tampered\n')
        with self.assertRaisesRegex(ValueError,'RESUME_INTEGRITY_ERROR'): ProofSession.open(clean.path).resume(self.root/'none.json')
    def test_human_property_is_new_before_handoff(self):
        s=self.session(); formal='(candidates.all (fun A => !(admissible A) || ((A.contains 4) || (A.contains 5)))) = true'
        self.assertNotIn(formal,[x.formal_statement for x in s.lemmas if x.verification_status=='VERIFIED'])
    def test_invalid_human_hint_and_boolean_true_closer_are_rejected(self):
        s=self.session(); bad=self.root/'bad.json'; bad.write_text(json.dumps({'type':'structured_lemma','or':[[99]]})); self.assertIsNone(s.import_human(bad))
        task=self.root/'bool.json'; task.write_text(json.dumps({'theorem':'b','boolean_implication_system':{'variables':['a','b'],'clauses':[[1]]}})); b=ProofSession.from_task(self.root/'bool-out',task).start(); self.assertEqual(b.status,SessionStatus.PROOF_RECOVERY_EXHAUSTED)


if __name__=='__main__': unittest.main()
