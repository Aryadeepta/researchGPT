import tempfile
import unittest
from pathlib import Path

from tools.proofbench.proof_engineering import (
    Capability, additive_basis_model, create_handoff_bundle, dimacs_export,
    propagate_constraints, readable_deduction_trace,
)


class ProofHandoffTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_scout_dimacs_and_human_trace_are_deterministic(self):
        model=additive_basis_model(20,{2,9,19},6,"blind-a")
        facts,residual,trace=propagate_constraints(model)
        self.assertEqual({k:facts[k] for k in ("x_0","x_1","x_2","x_3","x_9","x_19")},
            {"x_0":True,"x_1":True,"x_2":False,"x_3":True,"x_9":False,"x_19":False})
        self.assertTrue(any(c.constraint_id=="coverage-5" for c in residual))
        a=dimacs_export(model,self.root/"a"); b=dimacs_export(model,self.root/"b")
        self.assertEqual(a["cnf_sha256"],b["cnf_sha256"]); self.assertEqual(a["variable_map_sha256"],b["variable_map_sha256"])
        human=readable_deduction_trace(model,facts,trace)
        self.assertTrue(any(x.get("constraint_id")=="coverage-5" and set(x.get("variables",[]))=={"x_4","x_5"} for x in human))

    def test_no_solver_still_creates_actionable_handoff(self):
        bundle=create_handoff_bundle(self.root/"handoff",obligation_id="lb",obligation_hash="obligation",goal="⊢ False",classification="FINITE_COMBINATORIAL",model=additive_basis_model(8,{2},3,"obligation"),verified_prefix=["have h := x"],dag_state={"bp":{"status":"VERIFIED"}},attempts=[],diagnostics="LEAN_RESOURCE_EXHAUSTED",capabilities=[Capability("python_exec","python",[],[],{},{})])
        self.assertEqual(bundle["terminal_state"],"ACTIONABLE_HANDOFF")
        self.assertTrue((self.root/"handoff"/"residual.cnf").is_file())
        self.assertTrue((self.root/"handoff"/"codex_advisory.md").is_file())
        self.assertTrue(bundle["resume"]["validation_required"])
