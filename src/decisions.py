from copy import deepcopy
from uuid import uuid4

from src.research_state import now_iso


AUTO = "AUTO"
AGENTIC_RESOLUTION = "AGENTIC_RESOLUTION"
HUMAN_REQUIRED = "HUMAN_REQUIRED"


def make_decision_request(run_id, stage, question, why_human_is_needed, options, severity="MEDIUM", evidence=None, recommended_option=None, confidence=0.0, blocked_nodes=None, extra=None):
    request = {
        "decision_id": f"D{uuid4().hex[:8]}",
        "run_id": run_id,
        "stage": stage,
        "severity": severity,
        "question": question,
        "why_human_is_needed": why_human_is_needed,
        "evidence": evidence or [],
        "options": options,
        "recommended_option": recommended_option,
        "recommendation_confidence": confidence,
        "blocked_nodes": blocked_nodes or [],
        "status": "WAITING_FOR_HUMAN",
        "created_at": now_iso(),
        "resolved_at": None,
        "selected_option": None,
        "free_text": None,
    }
    if extra:
        request.update(extra)
    return request


class DecisionEngine:
    def classify(self, decision):
        risk = decision.get("risk", "low")
        reversible = decision.get("reversible", True)
        material = decision.get("material_scientific_impact", False)
        cost = float(decision.get("estimated_cost_usd", 0.0))
        if not material and reversible and risk == "low" and cost == 0:
            return AUTO
        if reversible and risk in ("low", "medium") and decision.get("candidate_analyses"):
            analyses = decision["candidate_analyses"]
            choices = [a.get("choice") for a in analyses if a.get("confidence", 0) >= 0.7]
            if choices and choices.count(choices[0]) == len(choices):
                return AGENTIC_RESOLUTION
        return HUMAN_REQUIRED

    def resolve_or_request(self, state, decision, notifier=None):
        level = self.classify(decision)
        if level in (AUTO, AGENTIC_RESOLUTION):
            record = deepcopy(decision)
            record["decision_level"] = level
            record["status"] = "RESOLVED"
            record["selected_option"] = decision.get("recommended_option") or decision.get("default_option")
            record["resolved_at"] = now_iso()
            state.setdefault("decision_history", []).append(record)
            return record
        request = make_decision_request(
            state["run_id"],
            decision.get("stage", "unknown"),
            decision["question"],
            decision.get("why_human_is_needed", "material decision requires human judgment"),
            decision.get("options", []),
            severity=decision.get("severity", "MEDIUM"),
            evidence=decision.get("evidence", []),
            recommended_option=decision.get("recommended_option"),
            confidence=decision.get("recommendation_confidence", 0.0),
            blocked_nodes=decision.get("blocked_nodes", []),
            extra={key: decision[key] for key in (
                "external_reasoning_bundle", "response_kind", "response_contract", "continuation") if key in decision},
        )
        state.setdefault("decisions", []).append(request)
        state["status"] = "WAITING_FOR_HUMAN"
        if notifier:
            notifier.notify_transition(state, "HUMAN_REQUIRED", request)
        return request


def list_open_decisions(state):
    return [d for d in state.get("decisions", []) if d.get("status") == "WAITING_FOR_HUMAN"]


def submit_decision(state, decision_id, option=None, free_text=None):
    for decision in state.get("decisions", []):
        if decision.get("decision_id") == decision_id:
            if decision.get("status") != "WAITING_FOR_HUMAN":
                raise ValueError(f"decision already resolved: {decision_id}")
            decision["status"] = "RESOLVED"
            decision["selected_option"] = option
            decision["free_text"] = free_text
            decision["resolved_at"] = now_iso()
            state.setdefault("decision_history", []).append(deepcopy(decision))
            blocked = set(decision.get("blocked_nodes", []))
            for node in state.get("dag", {}).get("nodes", {}).values():
                if node.get("node_id") in blocked and node.get("status") == "WAITING_FOR_HUMAN":
                    node["status"] = "PENDING"
                    node["failure_reason"] = None
                    node["lease"] = None
                    node["updated_at"] = now_iso()
            if not list_open_decisions(state):
                state["status"] = "PLANNED_RESEARCH"
            state["updated_at"] = now_iso()
            return decision
    raise KeyError(decision_id)
