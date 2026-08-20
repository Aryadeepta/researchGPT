import json, tempfile, unittest
from pathlib import Path
from tools.proofbench.proof_orchestrator import (CodexProofPlanner, GoalSnapshot, LeanGoalExtractor, LocalNextTacticSolver, ProbeOutcome, ProofOrchestrator, ProofPrefix, ProofState, TacticAttempt, TypedContextClosure, goal_fingerprint, normalize_goal, public_residual)
from tools.proofbench.proof_gym import GymCase, resolve_lean

class FakeProvider:
    def __init__(self, values): self.values=list(values); self.calls=0
    def generate_structured(self, *a, **kw):
        self.calls+=1
        return {"structured":{"tactic":self.values.pop(0) if self.values else "nonsense"},"model":"fake"}

class OrchestratorTests(unittest.TestCase):
    def setUp(self): self.t=tempfile.TemporaryDirectory(); self.root=Path(self.t.name); self.lean=resolve_lean()
    def tearDown(self): self.t.cleanup()
    def case(self, decl, expected=""):
        return GymCase("Lx","case","target",decl,expected or "", "test")
    def solve_case(self, case, **kw):
        return ProofOrchestrator(self.lean,self.root/"ws",self.root/"out",**kw).solve(case)
    def test_normalization_fingerprint(self):
        self.assertEqual(goal_fingerprint("x.lean:1:2: error: ⊢ P"),goal_fingerprint("x.lean:9:8: error: ⊢ P"))
        self.assertEqual(normalize_goal(" a\n b "),"a b")
    def test_reflexivity_final_and_authoritative_validation(self):
        c=self.case("(x : Nat) : x = x", "∀ (x : Nat), x = x")
        r=self.solve_case(c); self.assertTrue(r["pass"]); self.assertEqual(r["prefix"],["rfl"])
        self.assertTrue((self.root/"ws"/"Solution.lean").is_file())
    def test_constructor_can_increase_goals(self):
        c=self.case("(P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q", "∀ (P Q : Prop), P → Q → P ∧ Q")
        r=self.solve_case(c); self.assertTrue(r["pass"]); self.assertIn("constructor",r["prefix"])
    def test_conjunction_projection(self):
        c=self.case("(P Q : Prop) (h : P ∧ Q) : Q ∧ P", "∀ (P Q : Prop), P ∧ Q → Q ∧ P")
        r=self.solve_case(c); self.assertTrue(r["pass"])
    def test_typed_context_depth_two_and_multi_argument_are_lean_verified(self):
        c=self.case("(A B C : Prop) (h1 : A → B) (h2 : B → C) (hA : A) : C", "∀ (A B C : Prop), (A → B) → (B → C) → A → C")
        r=self.solve_case(c); self.assertTrue(r["pass"]); self.assertIn("context.closure.depth-2",r["accepted_strategies"])
        c=self.case("(A B C : Prop) (h : A → B → C) (ha : A) (hb : B) : C", "∀ (A B C : Prop), (A → B → C) → A → B → C")
        self.assertTrue(self.solve_case(c)["pass"])
    def test_context_closure_is_bounded_and_stable(self):
        s=ProofState(ProofPrefix(),GoalSnapshot("h1 : A → B\nh2 : B → C\nhA : A\n⊢ C","h1 : A → B\nh2 : B → C\nhA : A\n⊢ C",["⊢ C"],"g"))
        one=TypedContextClosure().candidates(s); two=TypedContextClosure().candidates(s)
        self.assertEqual([x.text for x in one],[x.text for x in two]); self.assertLessEqual(len(one),TypedContextClosure.max_candidates)
    def test_blueprint_is_acyclic_and_unverified_until_closed(self):
        c=self.case("(P Q : Prop) (hP : P) (hQ : Q) : P ∧ Q", "∀ (P Q : Prop), P → Q → P ∧ Q")
        r=self.solve_case(c); self.assertTrue(r["pass"])
        nodes=[json.loads(x) for x in (self.root/"out"/"blueprint.jsonl").read_text().splitlines()]
        self.assertTrue(any(x["status"]=="VERIFIED" for x in nodes))
        self.assertFalse(any(x["node_id"] in x["dependencies"] for x in nodes))
    def test_no_progress_and_failed_tactic_rejected(self):
        c=self.case("(P : Prop) : P")
        ex=LeanGoalExtractor(self.lean,self.root/"probe")
        ex.workspace.mkdir(); outcome, snap, _=ex.probe(c,ProofPrefix(["rfl"]))
        self.assertEqual(outcome,ProbeOutcome.LEAN_SYNTAX_OR_TACTIC_FAILURE); self.assertIsNone(snap)
    def test_qwen_fake_valid_and_malformed_bounded(self):
        c=self.case("(x : Nat) : x = x", "∀ (x : Nat), x = x")
        provider=FakeProvider(["rfl"]); solver=LocalNextTacticSolver(provider)
        # deterministic portfolio finishes first, but the fake solver is still bounded and safe.
        r=ProofOrchestrator(self.lean,self.root/"qws",self.root/"qout",solver=solver,qwen_attempts=2).solve(c)
        self.assertTrue(r["pass"]); self.assertLessEqual(provider.calls,2)
        self.assertIsNone(LocalNextTacticSolver(FakeProvider(["x"*257])).propose(c,ProofState(ProofPrefix(),GoalSnapshot("","⊢ P",["⊢ P"],"x")),[]))
    def test_remote_disabled_and_private_residual_rejected(self):
        self.assertIsNone(CodexProofPlanner(enabled=False).plan({"x":"public"}))
        c=self.case("(P : Prop) : P")
        state=ProofState(ProofPrefix(),GoalSnapshot("","⊢ P",["⊢ P"],"f"))
        with self.assertRaises(ValueError): public_residual(c,state,[TacticAttempt("a","x","x","hidden sentinel")])
    def test_forbidden_prefix_never_passes(self):
        c=self.case("(x : Nat) : x = x")
        ex=LeanGoalExtractor(self.lean,self.root/"forbidden"); ex.workspace.mkdir()
        self.assertEqual(ex.probe(c,ProofPrefix(["sorry"]))[0],ProbeOutcome.INTEGRITY_FAILURE)

    def test_finite_failure_is_actionable_handoff(self):
        c=self.case("(P : Prop) : P")
        # This intentionally exhausts direct Lean/search while retaining a
        # generic finite recovery adapter.  The bundle is recovery data only.
        object.__setattr__(c, "recovery_metadata", {"candidate_count": 1000,
            "finite_additive_basis": {"n": 8, "forbidden": [2], "max_selected": 3}})
        r=self.solve_case(c,max_nodes=1,max_depth=0)
        self.assertFalse(r["pass"]); self.assertEqual(r["terminal_state"],"ACTIONABLE_HANDOFF")
        handoff=Path(r["handoff_path"])
        self.assertTrue(handoff.is_file()); self.assertTrue((handoff.parent/"residual.cnf").is_file())
        self.assertIn("bottleneck",json.loads(handoff.read_text()))

if __name__ == "__main__": unittest.main()
