import re
from copy import deepcopy

from src.research_modalities import assess_automation_closure
from src.research_state import now_iso


PRIOR_WORK_COVERAGE = {"SUFFICIENT", "PARTIAL", "UNKNOWN", "UNAVAILABLE"}
LITERATURE_MODALITIES = {"literature_metadata"}


def normalize_research_query(raw_query):
    return re.sub(r"[_\s]+", " ", str(raw_query or "")).strip()


def classify_prior_work_coverage(attempts, relevant_count=0, sufficient=False):
    statuses = [item.get("status") for item in attempts or []]
    if sufficient:
        status = "SUFFICIENT"
    elif relevant_count:
        status = "PARTIAL"
    elif statuses and all(item == "PROVIDER_UNAVAILABLE" for item in statuses):
        status = "UNAVAILABLE"
    else:
        status = "UNKNOWN"
    return {
        "status": status,
        "retrieval_outcomes": deepcopy(attempts or []),
        "retrieved_record_count": int(relevant_count),
        "novelty_status": "ASSESSABLE" if status == "SUFFICIENT" else "NOT_ESTABLISHED",
        "novelty_score": None,
        "interpretation": (
            "Bounded retrieval coverage; an unsuccessful retrieval does not establish that prior work does not exist."
        ),
        "assessed_at": now_iso(),
    }


def question_scope_modalities(question):
    text = " ".join(str(question or "").lower().split())
    universal = any(marker in text for marker in (
        " always ", " for every ", " all possible ", " universally ", " in all cases ", "prove that", "proof of",
    ))
    bounded = any(marker in text for marker in (
        " bounded ", " specified ", " under ", " within ", " tested ", " benchmark ", "finite ", "input regime",
    ))
    if universal and not bounded:
        return {"scope_type": "UNIVERSAL_DEDUCTIVE", "bounded": False,
                "required_evidence_modalities": ["formal_proof"], "classification_origin": "deterministic_scope_semantics"}
    return {"scope_type": "BOUNDED_COMPUTATIONAL", "bounded": True,
            "required_evidence_modalities": ["executable_computation"], "classification_origin": "deterministic_scope_semantics"}


def validate_bounded_computational_question(question):
    text = " ".join(str(question or "").lower().split())
    errors = []
    if text.count("?") != 1:
        errors.append("produce one research question only")
    words = re.findall(r"[a-z0-9]+", text)
    fivegrams = [tuple(words[index:index + 5]) for index in range(max(0, len(words) - 4))]
    if fivegrams and max(fivegrams.count(item) for item in set(fivegrams)) > 2:
        errors.append("question contains excessive repeated phrasing")
    observable_markers = {
        "compare", "comparison", "versus", "difference", "effect", "impact", "varying", "measured", "measurement",
        "performance", "runtime", "memory", "correctness", "configuration", "algorithm", "strategy",
    }
    if not observable_markers.intersection(words):
        errors.append("question must identify an observable computational relationship or comparison")
    if not question_scope_modalities(question)["bounded"]:
        errors.append("question requires universal deductive support; bounded computational scope is required")
    return errors


def may_continue_without_literature(required_modalities, capabilities, policy=None, artifacts=None):
    policy = policy or {"objective": "autonomous_research"}
    assessment = assess_automation_closure(required_modalities, capabilities, artifacts)
    allowed = (
        policy.get("objective", "autonomous_research") == "autonomous_research"
        and assessment["automation_closure"] == "HIGH"
        and not set(required_modalities).intersection(LITERATURE_MODALITIES)
    )
    return allowed, assessment


def record_scope_refinement(original_question, refined_question, reason):
    return {"original_question": original_question, "refined_question": refined_question,
            "reason": reason, "transition": "EXPLICIT_SCOPE_REFINEMENT", "created_at": now_iso()}


def novelty_claim_allowed(prior_work_coverage):
    return (prior_work_coverage or {}).get("status") == "SUFFICIENT"
