import json
from uuid import uuid4

from src.research_state import now_iso


def create_engineering_request(state, problem, why_blocked, relevant_files=None, required_behavior="", acceptance_tests=None, suggested_implementation=""):
    request = {
        "engineering_request_id": f"E{uuid4().hex[:8]}",
        "run_id": state["run_id"],
        "problem": problem,
        "why_research_is_blocked": why_blocked,
        "relevant_files": relevant_files or [],
        "required_behavior": required_behavior,
        "acceptance_tests": acceptance_tests or [],
        "suggested_implementation": suggested_implementation,
        "generated_codex_prompt": (
            f"Research run {state['run_id']} is blocked.\n"
            f"Problem: {problem}\n"
            f"Required behavior: {required_behavior}\n"
            f"Acceptance tests: {json.dumps(acceptance_tests or [])}\n"
        ),
        "created_at": now_iso(),
        "status": "OPEN",
    }
    state.setdefault("engineering_requests", []).append(request)
    state["status"] = "BLOCKED_ENGINEERING_REQUIRED"
    return request
