import unittest

from tools.proofbench.proof_orchestrator import (
    probe_failure_has_only_terminal_sentinel,
)


class ProbeClassifierTests(unittest.TestCase):

    def test_deliberate_terminal_fail_is_admissible(self):
        out = """.orchestrator-probe.lean:9:4: error: Failed: `fail` tactic was invoked
a b : Nat
h : a = b
⊢ 0 + b = b
"""
        self.assertTrue(
            probe_failure_has_only_terminal_sentinel(out)
        )

    def test_unknown_identifier_is_not_admissible(self):
        out = """.orchestrator-probe.lean:6:6: error(lean.unknownIdentifier): Unknown identifier `h`
.orchestrator-probe.lean:5:113: error: unsolved goals
P Q : Prop
⊢ P ∧ Q
"""
        self.assertFalse(
            probe_failure_has_only_terminal_sentinel(out)
        )

    def test_unknown_tactic_is_not_admissible(self):
        out = """.orchestrator-probe.lean:7:3: error: unknown tactic
.orchestrator-probe.lean:5:113: error: unsolved goals
P Q : Prop
⊢ P
"""
        self.assertFalse(
            probe_failure_has_only_terminal_sentinel(out)
        )

    def test_candidate_error_plus_terminal_fail_is_not_admissible(self):
        out = """.orchestrator-probe.lean:6:2: error: unknown tactic
.orchestrator-probe.lean:9:4: error: Failed: `fail` tactic was invoked
P : Prop
⊢ P
"""
        self.assertFalse(
            probe_failure_has_only_terminal_sentinel(out)
        )


if __name__ == "__main__":
    unittest.main()
