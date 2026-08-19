import copy
import unittest

from src.budget_control import (
    agent_iteration_usage,
    extend_budget,
    reconcile_legacy_agent_iteration_blocks,
    record_budget_block,
)
from src.research_state import block_node, create_run_state, ready_nodes


class BudgetControlTests(unittest.TestCase):
    def blocked_state(self, usage=3, limit=2, second=False):
        state = create_run_state("run", "generic topic")
        state["budget"]["calls"] = [{"status": "SUCCESS"} for _ in range(usage)]
        block_node(state, "question_discovery", "BLOCKED_BUDGET", "MAX_AGENT_ITERATIONS")
        record_budget_block(state, "question_discovery", "agent_iterations", limit, usage,
                            "candidate_question_generation", "MAX_AGENT_ITERATIONS")
        if second:
            node = state["dag"]["nodes"]["evidence_discovery"]
            node.update({"status": "BLOCKED_BUDGET", "failure_reason": "MAX_RUN_LLM_USD"})
            record_budget_block(state, "evidence_discovery", "run_llm_usd", 1, 1,
                                "evidence_discovery", "MAX_RUN_LLM_USD")
        return state

    def test_initial_block_record_has_recovery_provenance(self):
        state = self.blocked_state()
        block = state["budget_blocks"][0]
        self.assertEqual(block["limit_at_block"], 2)
        self.assertEqual(block["usage_at_block"], 3)
        self.assertEqual(block["recoverability"], "REQUIRES_EXPLICIT_BUDGET_EXTENSION")

    def test_extension_persists_append_only_authorization_and_history(self):
        state = self.blocked_state()
        original_block = copy.deepcopy(state["budget_blocks"][0])
        scientific_before = copy.deepcopy(state["research_spec"])
        result = extend_budget(state, "agent_iterations", 5, "delegated free local budget")
        self.assertEqual(state["budget_blocks"][0], original_block)
        self.assertEqual(result["reopened_nodes"], ["question_discovery"])
        self.assertEqual(state["dag"]["nodes"]["question_discovery"]["status"], "PENDING")
        auth = state["budget_authorizations"][0]
        self.assertEqual((auth["previous_limit"], auth["new_limit"], auth["usage_at_authorization"]), (2, 5, 3))
        self.assertEqual(auth["additional_headroom"], 2)
        self.assertEqual(auth["source"], "USER_DELEGATED_CODEX")
        self.assertEqual(state["research_spec"], scientific_before)
        self.assertEqual(state["status"], "PLANNED_RESEARCH")
        self.assertEqual([node["node_id"] for node in ready_nodes(state)], ["question_discovery"])

    def test_invalid_or_decreasing_extension_rejected(self):
        state = self.blocked_state()
        with self.assertRaises(ValueError):
            extend_budget(state, "agent_iterations", 2, "no increase")
        with self.assertRaises(ValueError):
            extend_budget(state, "unknown", 10, "unsupported")

    def test_extension_at_current_usage_remains_blocked(self):
        state = self.blocked_state(usage=3, limit=2)
        result = extend_budget(state, "agent_iterations", 3, "authorized but exhausted")
        self.assertEqual(result["reopened_nodes"], [])
        self.assertEqual(state["status"], "BLOCKED_BUDGET")

    def test_only_matching_budget_blocks_reopen(self):
        state = self.blocked_state(second=True)
        result = extend_budget(state, "agent_iterations", 5, "matching budget only")
        self.assertEqual(result["reopened_nodes"], ["question_discovery"])
        self.assertEqual(state["dag"]["nodes"]["evidence_discovery"]["status"], "BLOCKED_BUDGET")
        self.assertEqual(state["status"], "BLOCKED_BUDGET")

    def test_infrastructure_failures_do_not_count(self):
        state = create_run_state("run", "topic")
        state["budget"]["calls"] = [
            {"failure_type": "TRANSIENT_LOCAL_RUNTIME_FAILURE"},
            {"failure_type": "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"},
            {"failure_type": "SCHEMA_VALIDATION_FAILURE"},
            {"status": "SUCCESS"},
        ]
        self.assertEqual(agent_iteration_usage(state), 2)

    def test_legacy_block_reconciliation_is_idempotent(self):
        state = create_run_state("run", "topic")
        block_node(state, "question_discovery", "BLOCKED_BUDGET", "MAX_AGENT_ITERATIONS")
        self.assertEqual(len(reconcile_legacy_agent_iteration_blocks(state, 20)), 1)
        self.assertEqual(len(reconcile_legacy_agent_iteration_blocks(state, 99)), 0)
        self.assertEqual(state["budget_blocks"][0]["limit_at_block"], 20)


if __name__ == "__main__":
    unittest.main()
