import json
import os
import re
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from src.adversarial import deterministic_adversarial_review
from src.capabilities import SkillManager, SkillRegistry, capability_requirement
from src.computational_skeleton import (canonical_semantic_id, clarification_effective,
                                        classify_measurement_source, control_availability_for_refinement,
                                        deterministic_bounded_question, experiment_relation_coherence,
                                        experimental_skeleton_validation, finalize_skeleton_status,
                                        measurement_direction_errors, measurement_from_kind,
                                        measurement_source_compatible, semantic_display_errors,
                                        normalize_semantic_value, observable_specificity_errors, semantic_value_record,
                                        variable_measurement_coherence,
                                        validate_measurement_contract, validate_variable_contract)
from src.computational_experiments import (adjudicate_executable_claim_candidates, executable_evidence_skill,
                                           validate_executable_evidence_protocol)
from src.decisions import DecisionEngine, list_open_decisions, submit_decision
from src.engineering import create_engineering_request
from src.literature import dedupe_records, literature_provider_from_env
from src.llm_gateway import BudgetExceeded, LLMRequest, MalformedStructuredOutput, MissingLLMProvider, ModelGateway
from src.local_inference import (
    CANONICAL_LLM_TASK_CLASSES,
    NoEligibleLocalModel,
    LocalRuntimeInfrastructureFailure,
    StructuredDecodingConfigurationFailure,
    StructuredGenerationExhausted,
    UnknownLLMTaskClass,
    validate_json_schema_subset,
    RuntimeRegistry,
)
from src.objective_coverage import selected_objective_coverage
from src.prior_work import (classify_prior_work_coverage, may_continue_without_literature,
                            normalize_research_query, question_scope_modalities,
                            validate_bounded_computational_question)
from src.research_modalities import (REQUIREMENT_TYPE_MODALITIES, ROUTE_MODALITIES,
                                     assess_automation_closure, objective_required_modalities,
                                     rank_candidate_questions)
from src.research_tools import classify_provider_search_outcome, provider_descriptor
from src.research_package import write_research_package
from src.research_state import block_node, complete_node, now_iso, record_verification


NODE_LLM_TASK_CLASS = {
    "question_discovery": "candidate_question_generation",
    "question_refinement": "candidate_question_generation",
    "feasibility_analysis": "research_feasibility_analysis",
    "skill_discovery_creation": "skill_code_generation",
    "adversarial_falsification": "adversarial_criticism",
    "claim_adjudication": "claim_adjudication",
}


SCORE_SCHEMA = {"type": "number", "minimum": 0.0, "maximum": 1.0}


QUESTION_DISCOVERY_SCHEMA = {
    "type": "object",
    "required": ["candidate_questions", "search_queries"],
    "additionalProperties": False,
    "properties": {
        "candidate_questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["question", "why_interesting", "falsifiability", "local_executability"],
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string", "minLength": 12},
                    "why_interesting": {"type": "string", "minLength": 8},
                    "falsifiability": {"type": "string", "minLength": 8},
                    "local_executability": {"type": "string", "minLength": 2},
                },
            },
        },
        "search_queries": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 3}},
    },
}


GUIDED_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action"],
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["tool", "final"]},
        "tool": {"type": "string", "enum": ["literature_search", "web_search", "fetch_web_source", "run_python", "run_bash", "read_artifact"]},
        "arguments": {"type": "object"},
        "result": {"type": "object"},
    },
}


SEARCH_QUERY_SCHEMA = {
    "type": "object",
    "required": ["query"],
    "additionalProperties": False,
    "properties": {"query": {"type": "string", "minLength": 3}},
}


ONE_QUESTION_SCHEMA = {
    "type": "object",
    "required": ["question"],
    "additionalProperties": False,
    "properties": {"question": {"type": "string", "minLength": 12}},
}

VARIABLE_SCHEMA = {"type": "object", "required": ["variable", "control_type"], "additionalProperties": False,
                   "properties": {"variable": {"type": "string", "minLength": 2},
                                  "control_type": {"type": "string", "enum": ["DIRECT_INPUT", "ALGORITHM_PARAMETER", "DERIVED_FROM_INPUT", "UNCONTROLLED", "UNCERTAIN"]}}}
VARIABLE_VALUE_SCHEMA = {"type": "object", "required": ["variable"], "additionalProperties": False,
                         "properties": {"variable": {"type": "string", "minLength": 2}}}
CONTROL_TYPE_SCHEMA = {"type": "object", "required": ["control_type"], "additionalProperties": False,
                       "properties": {"control_type": {"type": "string", "enum": ["DIRECT_INPUT", "ALGORITHM_PARAMETER", "DERIVED_FROM_INPUT", "UNCONTROLLED", "UNCERTAIN"]}}}
SOURCE_VARIABLE_SCHEMA = {"type": "object", "required": ["source_variable", "control_type"], "additionalProperties": False,
                          "properties": {"source_variable": {"type": "string", "minLength": 2},
                                         "control_type": {"type": "string", "enum": ["DIRECT_INPUT", "ALGORITHM_PARAMETER", "UNCONTROLLED", "UNCERTAIN"]}}}
MEASUREMENT_SCHEMA = {"type": "object", "required": ["measurement", "measurement_kind"], "additionalProperties": False,
                      "properties": {"measurement": {"type": "string", "minLength": 2},
                                     "measurement_kind": {"type": "string", "enum": ["runtime", "operation_count", "memory", "correctness", "output_value", "other"]}}}
MEASUREMENT_VALUE_SCHEMA = {"type": "object", "required": ["measurement"], "additionalProperties": False,
                            "properties": {"measurement": {"type": "string", "minLength": 2}}}
MEASUREMENT_KIND_SCHEMA = {"type": "object", "required": ["measurement_kind"], "additionalProperties": False,
                           "properties": {"measurement_kind": {"type": "string", "enum": ["runtime", "operation_count", "memory", "correctness", "output_value", "other", "uncertain"]}}}
OBSERVATION_SCHEMA = {"type": "object", "required": ["observation"], "additionalProperties": False,
                      "properties": {"observation": {"type": "string", "minLength": 2}}}
TESTABILITY_SCHEMA = {"type": "object", "required": ["assessment"], "additionalProperties": False,
                      "properties": {"assessment": {"type": "string", "enum": ["TESTABLE", "NOT_TESTABLE", "UNCERTAIN"]}}}
CONTROLLABILITY_SCHEMA = {"type": "object", "required": ["assessment"], "additionalProperties": False,
                          "properties": {"assessment": {"type": "string", "enum": ["DIRECTLY_CONTROLLABLE", "NEEDS_CLARIFICATION", "UNCERTAIN"]}}}
MEASUREMENT_NEUTRALITY_SCHEMA = {"type": "object", "required": ["assessment"], "additionalProperties": False,
                                 "properties": {"assessment": {"type": "string", "enum": ["NEUTRAL_MEASUREMENT", "RESULT_EMBEDDED", "NOT_MEASURABLE", "UNCERTAIN"]}}}
OBSERVABLE_SCHEMA = {"type": "object", "required": ["observable"], "additionalProperties": False,
                     "properties": {"observable": {"type": "string", "minLength": 2}}}
INFORMATIVENESS_SCHEMA = {"type": "object", "required": ["assessment"], "additionalProperties": False,
                          "properties": {"assessment": {"type": "string", "enum": ["INFORMATIVE", "WEAKLY_INFORMATIVE", "UNINFORMATIVE", "UNCERTAIN"]}}}

SCORE_REASON_SCHEMA = {
    "type": "object",
    "required": ["score", "reason"],
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "minLength": 12},
    },
}

NOVELTY_ASSESSMENT_SCHEMA = {
    "type": "object",
    "required": ["assessment", "reason", "confidence"],
    "additionalProperties": False,
    "properties": {
        "assessment": {"type": "string", "enum": ["plausible_gap", "well_covered", "uncertain"]},
        "reason": {"type": "string", "minLength": 12},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


QUESTION_REFINEMENT_SCHEMA = {
    "type": "object",
    "required": ["selected_question", "candidate_evaluations", "rationale"],
    "additionalProperties": False,
    "properties": {
        "selected_question": {"type": "string", "minLength": 12},
        "candidate_evaluations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["question", "feasibility", "novelty_potential", "falsifiability", "evidence_accessibility", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string", "minLength": 12},
                    "feasibility": SCORE_SCHEMA,
                    "novelty_potential": SCORE_SCHEMA,
                    "novelty_status": {"type": "string"},
                    "testability_status": {"type": "string", "enum": ["TESTABLE", "NOT_TESTABLE", "UNCERTAIN"]},
                    "falsifiability_provenance": {"type": "object"},
                    "falsifiability": SCORE_SCHEMA,
                    "evidence_accessibility": SCORE_SCHEMA,
                    "required_evidence_modalities": {"type": "array", "items": {"type": "string", "minLength": 2}},
                    "currently_available_modalities": {"type": "array", "items": {"type": "string", "minLength": 2}},
                    "missing_modalities": {"type": "array", "items": {"type": "string", "minLength": 2}},
                    "automation_closure": {"type": "string", "enum": ["HIGH", "CONDITIONAL", "LOW", "UNKNOWN"]},
                    "assessment_origin": {"type": "string", "minLength": 4},
                    "rationale": {"type": "string", "minLength": 12},
                },
            },
        },
        "rationale": {"type": "string", "minLength": 12},
        "limitations": {"type": "array", "items": {"type": "string", "minLength": 3}},
    },
}


FEASIBILITY_SCHEMA = {
    "type": "object",
    "required": [
        "research_question",
        "methodology",
        "feasibility_verdict",
        "evidence_requirements",
        "resource_constraints",
        "validation_plan",
        "hypotheses",
        "falsification_criteria",
        "required_claims",
        "completion_contracts",
    ],
    "additionalProperties": False,
    "properties": {
        "research_question": {"type": "string", "minLength": 12},
        "methodology": {"type": "string", "minLength": 6},
        "feasibility_verdict": {"type": "string", "enum": ["FEASIBLE", "PARTIAL", "BLOCKED"]},
        "evidence_requirements": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 4}},
        "resource_constraints": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 4}},
        "validation_plan": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 4}},
        "hypotheses": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 6}},
        "falsification_criteria": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 6}},
        "required_claims": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 6}},
        "completion_contracts": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 6}},
        "external_dependencies": {"type": "array", "items": {"type": "string", "minLength": 3}},
        "replication_tolerance": {"type": "object"},
    },
}

FEASIBILITY_OPERATIONALIZATION_SCHEMA = {
    "type": "object", "required": ["observable_test"], "additionalProperties": False,
    "properties": {"observable_test": {"type": "string", "minLength": 12}},
}
FEASIBILITY_ROUTE_SCHEMA = {
    "type": "object", "required": ["approach", "reason"], "additionalProperties": False,
    "properties": {
        "approach": {"type": "string", "enum": ["secondary_data_analysis", "simulation", "primary_measurement", "controlled_experiment", "systematic_evidence_analysis", "other"]},
        "reason": {"type": "string", "minLength": 12},
    },
}
FEASIBILITY_REQUIREMENT_SCHEMA = {
    "type": "object", "required": ["requirement_type", "requirement"], "additionalProperties": False,
    "properties": {
        "requirement_type": {"type": "string", "enum": ["data", "measurement", "software", "compute", "external_service", "apparatus", "method", "other"]},
        "requirement": {"type": "string", "minLength": 8},
    },
}
FEASIBILITY_FIT_SCHEMA = {
    "type": "object", "required": ["fit", "reason"], "additionalProperties": False,
    "properties": {
        "fit": {"type": "string", "enum": ["good", "partial", "poor", "uncertain"]},
        "reason": {"type": "string", "minLength": 12},
    },
}


def minimal_literature_context(records, limit=8):
    excerpts = []
    for record in records[:limit]:
        excerpts.append({
            "identifier": record.get("identifier"),
            "title": record.get("title"),
            "year": record.get("year"),
            "abstract_excerpt": (record.get("abstract") or "")[:800],
            "verification_status": record.get("verification_status"),
        })
    return excerpts


def compact_literature_cards(records, limit=4):
    """Small, lossless-by-reference cards for weak-model planning prompts."""
    cards = []
    for index, record in enumerate(records[:limit], 1):
        abstract = re.sub(r"\s+", " ", _text(record.get("abstract")))
        cards.append({
            "card": index,
            "identifier": record.get("identifier"),
            "title": record.get("title"),
            "year": record.get("year"),
            "observation": abstract[:180],
        })
    return cards


def record_relevance(record, topic, candidates=None):
    candidates = candidates or []
    query = record.get("search_query") or ""
    text = " ".join(str(record.get(key) or "") for key in ("title", "abstract", "venue"))
    usable_query = "" if placeholder_like(query) else query
    anchors = [topic, usable_query]
    for candidate in candidates:
        if isinstance(candidate, dict):
            anchors.append(candidate.get("question") or "")
        else:
            anchors.append(str(candidate))
    topic_score = topic_overlap_score(text, [topic])
    query_score = topic_overlap_score(text, [usable_query])
    candidate_score = topic_overlap_score(text, anchors)
    score = max(topic_score, query_score, candidate_score)
    return {
        "identifier": record.get("identifier"),
        "title": record.get("title"),
        "search_query": query,
        "verification_status": record.get("verification_status"),
        "topic_overlap": round(topic_score, 4),
        "query_overlap": round(query_score, 4),
        "candidate_overlap": round(candidate_score, 4),
        "score": round(score, 4),
        "relevant": score >= 0.2,
    }


def literature_relevance_report(records, topic, candidates=None, min_relevant=1, min_fraction=0.25):
    diagnostics = [record_relevance(record, topic, candidates) for record in records]
    relevant = [diag for diag in diagnostics if diag["relevant"]]
    fraction = len(relevant) / len(diagnostics) if diagnostics else 0.0
    return {
        "record_count": len(records),
        "relevant_count": len(relevant),
        "relevant_fraction": round(fraction, 4),
        "min_relevant": min_relevant,
        "min_fraction": min_fraction,
        "usable": len(relevant) >= min_relevant and fraction >= min_fraction,
        "diagnostics": diagnostics,
        "created_at": now_iso(),
    }


def _text(value):
    return value.strip() if isinstance(value, str) else ""


PLACEHOLDER_PHRASES = {
    "question",
    "why_interesting",
    "falsifiability",
    "local_executability",
    "search query",
    "broad discovery query",
    "targeted query",
    "novelty challenge query",
    "non-empty string",
    "string",
}


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when", "where", "whether", "with",
}


def lexical_terms(text):
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", _text(text).lower())
        if token not in STOPWORDS
    }


def placeholder_like(value):
    text = _text(value).lower()
    if not text:
        return True
    normalized = re.sub(r"[\s_:-]+", " ", text).strip()
    if normalized in PLACEHOLDER_PHRASES:
        return True
    tokens = lexical_terms(text)
    if not tokens:
        return True
    schema_terms = {"question", "query", "search", "targeted", "broad", "novelty", "interesting", "falsifiability", "executable", "string", "placeholder"}
    return tokens.issubset(schema_terms)


def local_guidance_high():
    default = "high" if os.environ.get("RESEARCH_LLM_PROVIDER", "").lower() == "local" else "off"
    return os.environ.get("RESEARCH_LOCAL_GUIDANCE", default).lower() in {"high", "1", "true", "yes"}


def token_count(text):
    return len(str(text).split())


def topic_overlap_score(text, anchors):
    terms = lexical_terms(text)
    anchor_terms = set()
    for anchor in anchors:
        anchor_terms.update(lexical_terms(anchor))
    if not terms or not anchor_terms:
        return 0.0
    return len(terms & anchor_terms) / max(1, min(len(terms), len(anchor_terms)))


def validate_question_discovery_semantics(data, topic=None):
    errors = []
    topic_text = _text(topic).lower()
    questions = data.get("candidate_questions", []) if isinstance(data, dict) else []
    seen = set()
    for idx, item in enumerate(questions):
        question = _text(item.get("question") if isinstance(item, dict) else "")
        if not question.endswith("?"):
            errors.append(f"candidate_questions[{idx}].question should be phrased as a question")
        if topic_text and question.lower() == topic_text:
            errors.append(f"candidate_questions[{idx}].question merely repeats the broad topic")
        if question.lower() in seen:
            errors.append(f"candidate_questions[{idx}].question duplicates another candidate")
        if placeholder_like(question):
            errors.append(f"candidate_questions[{idx}].question is placeholder or content-free")
        validation = candidate_question_semantic_validation(question, topic)
        if not validation["substantive_question"]:
            errors.extend(
                f"candidate_questions[{idx}].question {reason}"
                for reason in validation["rejection_reasons"]
            )
        seen.add(question.lower())
    queries = data.get("search_queries", []) if isinstance(data, dict) else []
    if not any(_text(q) for q in queries):
        errors.append("search_queries must contain at least one non-empty query")
    for idx, query in enumerate(queries):
        query_text = _text(query.get("query") if isinstance(query, dict) else query)
        if placeholder_like(query_text):
            errors.append(f"search_queries[{idx}] is placeholder or content-free")
        elif topic_text and topic_overlap_score(query_text, [topic_text] + [(_text(q.get("question")) if isinstance(q, dict) else "") for q in questions]) < 0.2:
            errors.append(f"search_queries[{idx}] is disconnected from topic/candidate questions")
    return errors


def validate_question_refinement_semantics(data, topic=None):
    errors = []
    selected = _text(data.get("selected_question") if isinstance(data, dict) else "")
    evaluations = data.get("candidate_evaluations", []) if isinstance(data, dict) else []
    rationale = _text(data.get("rationale") if isinstance(data, dict) else "")
    if not selected:
        errors.append("selected_question must be a non-empty string")
    if selected and _text(topic).lower() == selected.lower():
        errors.append("selected_question merely repeats the broad topic")
    if len(rationale.split()) < 4:
        errors.append("rationale is too short to justify selection")
    eval_questions = []
    for idx, evaluation in enumerate(evaluations):
        question = _text(evaluation.get("question") if isinstance(evaluation, dict) else "")
        eval_questions.append(question)
        for key in ("feasibility", "novelty_potential", "falsifiability", "evidence_accessibility"):
            value = evaluation.get(key) if isinstance(evaluation, dict) else None
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or value > 1:
                errors.append(f"candidate_evaluations[{idx}].{key} must be a number from 0 to 1")
        if len(_text(evaluation.get("rationale") if isinstance(evaluation, dict) else "").split()) < 3:
            errors.append(f"candidate_evaluations[{idx}].rationale is too short")
    if selected and eval_questions and selected not in eval_questions:
        errors.append("selected_question must appear in candidate_evaluations.question")
    return errors


def empirical_answer_like_planning_reason(reason):
    text = _text(reason).lower()
    if not text:
        return True
    assertive_patterns = (
        r"\b(proves|demonstrates|establishes|confirms|shows that)\b",
        r"\b(has|have|is|are|reduces|increases|improves|decreases|causes|leads to)\b.{0,80}\b(impact|effect|result|outcome|performance|rate|risk|measurement)\b",
    )
    planning_markers = ("could", "may", "can", "would", "assess", "evaluate", "test", "planning", "metadata", "available", "requires")
    if any(marker in text for marker in planning_markers):
        return False
    return any(re.search(pattern, text) for pattern in assertive_patterns)


def validate_atomic_planning_score(data, topic=None):
    errors = validate_json_schema_subset(data, SCORE_REASON_SCHEMA)
    if errors:
        return errors
    reason = _text(data.get("reason"))
    if len(reason.split()) < 4:
        errors.append("reason is too short")
    if empirical_answer_like_planning_reason(reason):
        errors.append("reason appears to answer the research question instead of assessing planning suitability")
    procedural = (
        "the task was to", "the previous response", "satisfies the schema",
        "json object", "validation error", "the question is to assess", "answer should specify",
        "answer should be", "i will assess",
        "must explain what observable result",
    )
    if any(marker in reason.lower() for marker in procedural) and not any(
        marker in _text(topic).lower() for marker in ("json", "schema", "validation", "software format")
    ):
        errors.append("reason is a procedural repair artifact rather than a substantive planning judgment")
    return errors


def validate_atomic_semantic_value(data, key, topic=None):
    value = _text(data.get(key) if isinstance(data, dict) else "")
    errors = []
    if not value or value.lower() in {"ok", "unknown", "n/a", "none"}:
        errors.append(f"{key} must contain a substantive semantic value")
    procedural = ("json object", "the task was", "previous response", "schema", "validation error", "instructions say")
    if any(marker in value.lower() for marker in procedural) and not any(
        marker in _text(topic).lower() for marker in ("json", "schema", "software format")
    ):
        errors.append(f"{key} is a procedural/schema artifact")
    if key == "measurement" and (not any(char.isalpha() for char in value) or value.lower() in {"value", "result", "outcome", "number", "impact", "effect"}):
        errors.append("measurement must name an observable computational quantity, not a literal or generic placeholder")
    if key == "measurement" and not any(marker in value.lower() for marker in (
        "time", "duration", "memory", "operation", "accuracy", "correct", "throughput", "cost", "count", "rate", "error", "output", "size",
    )):
        errors.append("measurement must identify a measurable computational quantity")
    if key == "observation":
        hypothetical = ("if ", "would", "could", "no difference", "different", "varies", "across")
        if not any(marker in value.lower() for marker in hypothetical):
            errors.append("observation must describe a possible distinguishing outcome rather than assert an answer")
        if empirical_answer_like_planning_reason(value) or any(marker in value.lower() for marker in ("directly impacts", "direct relationship", "leading to", "therefore improves", "therefore reduces", "reveals how", "reveals the")):
            errors.append("observation answers the empirical question instead of describing a possible outcome")
    return errors


def validate_controllable_variable_candidate(value):
    normalized = _text(value).lower()
    if normalized in {"number", "value", "input", "parameter", "variable", "result", "outcome", "amount"}:
        return ["proposed variable is underspecified and not representable as a directly controlled input parameter"]
    if not any(char.isalpha() for char in normalized):
        return ["proposed variable does not identify a controllable input parameter"]
    return []


def semantic_value_tokens(value):
    return re.findall(r"[a-z0-9]+", _text(value).lower())


def validate_experimental_skeleton_semantics(skeleton, topic=None):
    errors = validate_controllable_variable_candidate(skeleton.get("independent_variable"))
    errors += validate_atomic_semantic_value({"measurement": skeleton.get("dependent_measurement")}, "measurement", topic)
    if semantic_value_tokens(skeleton.get("independent_variable")) == semantic_value_tokens(skeleton.get("dependent_measurement")):
        errors.append("independent variable and dependent measurement are semantically identical")
    if skeleton.get("distinguishing_observation"):
        errors += validate_atomic_semantic_value({"observation": skeleton.get("distinguishing_observation")}, "observation", topic)
    return errors


def validate_candidate_clarification(data, original_question, topic=None):
    question = normalize_atomic_question(data.get("question") if isinstance(data, dict) else "")
    errors = validate_single_question(question, topic, []) + validate_bounded_computational_question(question)
    if " ".join(question.lower().split()) == " ".join(_text(original_question).lower().split()):
        errors.append("clarification must operationally change the ambiguous candidate")
    return errors


def legacy_falsifiability_from_testability(status):
    mapping = {"TESTABLE": 1.0, "NOT_TESTABLE": 0.0, "UNCERTAIN": 0.0}
    return {"value": mapping[status], "source": "deterministic_mapping_from_typed_testability",
            "authoritative_field": "testability_status", "scientific_measurement": False}


def validate_dimension_score(data, dimension, topic=None):
    errors = validate_atomic_planning_score(data, topic)
    if errors:
        return errors
    reason = _text(data.get("reason")).lower()
    literature_basis = (
        "not enough papers", "few papers", "number of papers", "literature is not",
        "literature was not", "lack of literature", "no literature", "no sufficient evidence", "insufficient evidence",
    )
    if dimension == "falsifiability":
        falsifiability_terms = (
            "observ", "measur", "compar", "contradict", "against", "no association",
            "no effect", "lack of association", "distinguish", "constraint", "fail to support",
        )
        negates_observability = any(marker in reason for marker in ("not possible to measure", "not possible to observe", "not observable", "not measurable"))
        outcome_against = any(term in reason for term in ("would count against", "would contradict", "no association", "no effect", "fail to support", "difference would"))
        wrong_only = any(marker in reason for marker in literature_basis) and not outcome_against
        if wrong_only or negates_observability or not any(term in reason for term in falsifiability_terms):
            errors.append("falsifiability reason must explain what observable result could count against or constrain the relationship")
    elif dimension == "feasibility":
        if not any(term in reason for term in ("resource", "tool", "data", "local", "free", "comput", "experiment", "practical", "available")):
            errors.append("feasibility reason must address practical data, tools, resources, or execution constraints")
    return errors


def validate_novelty_assessment(data, topic=None):
    errors = validate_json_schema_subset(data, NOVELTY_ASSESSMENT_SCHEMA)
    if errors:
        return errors
    reason = _text(data.get("reason"))
    if len(reason.split()) < 4:
        errors.append("novelty reason is too short")
    if empirical_answer_like_planning_reason(reason):
        errors.append("novelty reason answers the empirical question instead of assessing prior-work coverage")
    if not any(term in reason.lower() for term in ("prior", "work", "stud", "literature", "coverage", "gap", "comparison", "setting", "measurement", "similar", "retrieved")):
        errors.append("novelty reason must address similarity, prior coverage, or a specific gap")
    if data.get("assessment") == "plausible_gap" and any(marker in reason.lower() for marker in ("no provided information", "not enough information", "cannot determine", "insufficient information")):
        errors.append("insufficient coverage information must be assessed as uncertain, not plausible_gap")
    return errors


def validate_feasibility_semantics(data):
    errors = []
    if not data.get("evidence_requirements"):
        errors.append("feasibility must identify evidence requirements")
    if not data.get("resource_constraints"):
        errors.append("feasibility must consider resource constraints")
    if not data.get("validation_plan"):
        errors.append("feasibility must propose validation or falsification path")
    verdict = data.get("feasibility_verdict")
    blockers = [item for item in data.get("resource_constraints", []) if "unavailable" in str(item).lower() or "blocked" in str(item).lower()]
    if verdict == "FEASIBLE" and blockers:
        errors.append("FEASIBLE verdict contradicts listed unavailable/blocked resources")
    if data.get("required_claims") and not data.get("completion_contracts"):
        errors.append("required claims need completion contracts")
    return errors


def validate_observable_test(data, topic=None):
    errors = validate_json_schema_subset(data, FEASIBILITY_OPERATIONALIZATION_SCHEMA)
    text = _text(data.get("observable_test")) if isinstance(data, dict) else ""
    if errors:
        return errors
    if not any(term in text.lower() for term in ("observ", "measur", "compar", "difference", "association", "relationship", "effect", "change", "rate")):
        errors.append("observable_test must describe an observable relationship, measurement, or comparison")
    if empirical_answer_like_planning_reason(text):
        errors.append("observable_test asserts an empirical result instead of describing a planned test")
    if any(term in text.lower() for term in ("data are available", "dataset is available", "measurements exist", "results show")):
        errors.append("observable_test claims an unverified resource or result exists")
    return errors


def validate_feasibility_route(data, topic=None):
    errors = validate_json_schema_subset(data, FEASIBILITY_ROUTE_SCHEMA)
    if errors:
        return errors
    reason = _text(data.get("reason"))
    errors.extend(validate_atomic_planning_score({"score": 0.5, "reason": reason}, topic))
    if any(marker in reason.lower() for marker in ("have been shown to", "has been shown to", "are generally lower", "are generally higher", "results demonstrate")):
        errors.append("route reason asserts an empirical result instead of proposing an investigation method")
    approach_terms = {
        "secondary_data_analysis": ("data", "dataset", "analysis", "record"),
        "simulation": ("simulat", "model", "comput"),
        "primary_measurement": ("measur", "observ", "field"),
        "controlled_experiment": ("experiment", "control", "compar"),
        "systematic_evidence_analysis": ("systematic", "evidence", "literature", "review"),
        "other": ("route", "method", "approach"),
    }
    if not any(term in reason.lower() for term in approach_terms.get(data.get("approach"), ())):
        errors.append("route reason must explain how the named approach would investigate the observable test")
    return errors


def validate_feasibility_requirement(data):
    errors = validate_json_schema_subset(data, FEASIBILITY_REQUIREMENT_SCHEMA)
    requirement = _text(data.get("requirement")) if isinstance(data, dict) else ""
    words = re.findall(r"[A-Za-z0-9]+", requirement)
    enum_labels = {"data", "measurement", "software", "compute", "external", "service", "apparatus", "method", "other"}
    if requirement and (len(words) < 4 or (set(word.lower() for word in words) <= enum_labels)):
        errors.append("requirement must be a substantive description, not an enum/schema echo or identifier slug")
    if requirement.rstrip().endswith("?"):
        errors.append("requirement must state a needed resource or capability, not repeat a research question")
    if requirement.lstrip().startswith(("{", "[")) or '"requirement_type"' in requirement:
        errors.append("requirement must be plain substantive text, not a serialized schema or nested response")
    if requirement and any(term in requirement.lower() for term in ("is available", "are available", "has been acquired", "already exists")):
        errors.append("requirement must be a proposal, not an assertion of availability")
    if requirement and empirical_answer_like_planning_reason(requirement):
        errors.append("requirement asserts an empirical result instead of proposing a needed resource or capability")
    if requirement and any(marker in requirement.lower() for marker in ("the task was to", "previous response", "the response should", "response must",
                                                                           "json", "schema", "validation error")):
        errors.append("requirement is a procedural repair artifact")
    return errors


def validate_feasibility_fit(data, topic=None):
    errors = validate_json_schema_subset(data, FEASIBILITY_FIT_SCHEMA)
    if errors:
        return errors
    reason = _text(data.get("reason"))
    errors.extend(validate_atomic_planning_score({"score": 0.5, "reason": reason}, topic))
    if not any(term in reason.lower() for term in ("address", "test", "fit", "comparison", "observable", "partial", "route")):
        errors.append("scientific-fit reason must explain whether the route addresses the observable test")
    if any(marker in reason.lower() for marker in ("supports the conclusion", "support the conclusion", "evidence is sufficient", "evidence is adequate", "results show")):
        errors.append("scientific-fit reason asserts empirical support instead of judging whether the route addresses the question")
    return errors


def revalidate_planning_state(state, stages=None):
    stages = set(stages or ("question_discovery", "question_refinement"))
    invalid = []
    if "question_discovery" in stages and state.get("candidate_questions"):
        discovery = {"candidate_questions": state.get("candidate_questions"), "search_queries": state.get("search_strategy", {}).get("queries", [])}
        errors = validate_json_schema_subset(discovery, QUESTION_DISCOVERY_SCHEMA) + validate_question_discovery_semantics(discovery, state.get("topic"))
        if errors:
            invalid.append(("question_discovery", errors))
    if "question_refinement" in stages and (state.get("selected_question") is not None or state.get("candidate_evaluations")):
        refinement = {
            "selected_question": state.get("selected_question"),
            "candidate_evaluations": state.get("candidate_evaluations", []),
            "rationale": state.get("question_refinement_rationale", ""),
        }
        errors = validate_json_schema_subset(refinement, QUESTION_REFINEMENT_SCHEMA) + validate_question_refinement_semantics(refinement, state.get("topic"))
        if errors:
            invalid.append(("question_refinement", errors))
    return invalid


def clear_node_outputs(state, node_id):
    if node_id == "question_refinement":
        state["selected_question"] = None
        state["candidate_evaluations"] = []
        state.pop("question_refinement_rationale", None)
    if node_id == "feasibility_analysis":
        state.pop("research_spec", None)


def response_schema_for_node(node_id):
    if node_id == "question_discovery":
        return QUESTION_DISCOVERY_SCHEMA
    if node_id == "question_refinement":
        return QUESTION_REFINEMENT_SCHEMA
    if node_id == "feasibility_analysis":
        return FEASIBILITY_SCHEMA
    return {"type": "object"}


def semantic_errors_for_node(node_id, data, state):
    if node_id == "question_discovery":
        return validate_question_discovery_semantics(data, state.get("topic"))
    if node_id == "question_refinement":
        return validate_question_refinement_semantics(data, state.get("topic"))
    if node_id == "feasibility_analysis":
        return validate_feasibility_semantics(data)
    return []


def validate_human_response_for_node(node_id, data, state):
    schema_errors = validate_json_schema_subset(data, response_schema_for_node(node_id))
    semantic_errors = [] if schema_errors else semantic_errors_for_node(node_id, data, state)
    return schema_errors, semantic_errors


def artifact_contract_for_node(node):
    return node.get("contract", {})


def candidate_question_context(state):
    candidates = state.get("candidate_questions", [])
    warnings = []
    placeholder_labels = {"question", "why_interesting", "falsifiability", "local_executability"}
    if candidates and all(isinstance(item, str) for item in candidates) and set(candidates).issubset(placeholder_labels):
        warnings.append("Persisted candidate_questions appear to be schema placeholder labels, not substantive research questions.")
        return [], candidates, warnings
    return candidates, candidates, warnings


def fallback_search_queries(topic, candidates=None):
    candidates = candidates or []
    queries = []
    topic_text = _text(topic)
    if topic_text and not placeholder_like(topic_text):
        queries.append(topic_text)
        terms = sorted(lexical_terms(topic_text))
        if len(terms) >= 2:
            queries.append(" ".join(terms[:6]))
    for candidate in candidates:
        question = candidate.get("question") if isinstance(candidate, dict) else str(candidate)
        if question and not placeholder_like(question):
            terms = sorted(lexical_terms(question))
            if terms:
                queries.append(" ".join(terms[:8]))
    unique = []
    seen = set()
    for query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique[:5]


def normalize_guided_action(action):
    normalized = deepcopy(action or {})
    rules = []
    args = normalized.get("arguments")
    if normalized.get("action") == "tool" and isinstance(args, dict):
        nested_tool = args.get("tool") or args.get("method")
        if nested_tool and not normalized.get("tool") and nested_tool in {"literature_search", "web_search", "fetch_web_source", "run_python", "run_bash", "read_artifact"}:
            normalized["tool"] = nested_tool
            rules.append("arguments.tool_or_method_to_top_level_tool")
        if "query" not in args and normalized.get("query"):
            args["query"] = normalized["query"]
            rules.append("top_level_query_to_arguments_query")
    if normalized.get("action") in {"literature_search", "web_search", "fetch_web_source", "run_python", "run_bash", "read_artifact"}:
        tool = normalized["action"]
        normalized = {"action": "tool", "tool": tool, "arguments": {k: v for k, v in normalized.items() if k not in {"action", "tool"}}}
        rules.append("tool_name_action_to_tool_envelope")
    return normalized, rules


def validate_search_query(query, topic, candidates=None):
    query_text = _text(query)
    if placeholder_like(query_text):
        return ["query is placeholder or content-free"]
    anchors = [topic] + [c.get("question", "") for c in candidates or [] if isinstance(c, dict)]
    if topic_overlap_score(query_text, anchors) < 0.2:
        return ["query is disconnected from topic/candidates"]
    return []


def validate_novelty_search_query(query, topic, question):
    errors = validate_search_query(query, topic, [{"question": question}])
    normalized = re.sub(r"_+", " ", _text(query)).strip()
    if len(normalized.split()) > 12:
        errors.append("novelty search query is too long")
    if re.search(r"\d{8,}", normalized):
        errors.append("novelty search query contains an implausible numeric token")
    return errors


def normalize_atomic_question(question):
    if not isinstance(question, str):
        return question
    normalized = question.strip()
    normalized = re.sub(r"\?\s*[,.;:]+$", "?", normalized)
    return normalized


def topic_is_meta_research(topic):
    text = _text(topic).lower()
    markers = (
        "meta-analysis", "systematic review", "evidence synthesis", "literature review",
        "bibliometric", "scientific publishing", "publication bias", "research literature",
        "citation analysis", "research evidence",
    )
    return any(marker in text for marker in markers)


def candidate_question_semantic_validation(question, topic, records=None):
    text = _text(question)
    lower = text.lower()
    meta_patterns = (
        r"^what (?:does|do) (?:the )?literature (?:say|show)",
        r"^what evidence in (?:the )?retrieved literature",
        r"^what evidence (?:exists|is available)(?: in| for| about)?",
        r"^can (?:we|one) find evidence",
        r"^what sources (?:are|were) available",
        r"^how can (?:this|the) (?:topic|subject) be researched",
        r"\b(?:sources|literature) (?:are|is) available for\b",
    )
    meta = any(re.search(pattern, lower) for pattern in meta_patterns)
    meta_topic = topic_is_meta_research(topic)
    anchors = [topic]
    for record in records or []:
        anchors.extend((record.get("title") or "", record.get("abstract") or ""))
    connected = topic_overlap_score(text, anchors) >= 0.2 if _text(topic) else True
    question_form = text.endswith("?") and not placeholder_like(text)
    observable_markers = (
        "effect", "relationship", "association", "difference", "compare", "comparison",
        "measure", "rate", "change", "influence", "predict", "mechanism", "correlat",
        "increase", "decrease", "vary", "extent", "how much", "whether",
    )
    testable = question_form and (meta_topic or any(marker in lower for marker in observable_markers) or (not meta and connected))
    reasons = []
    if meta and not meta_topic:
        reasons.append("is a meta-research/planning question for an ordinary empirical topic")
    if not connected:
        reasons.append("is disconnected from the topic and retrieved literature")
    if not testable:
        reasons.append("does not state an empirically testable question")
    substantive = question_form and connected and (not meta or meta_topic) and testable
    return {
        "substantive_question": substantive,
        "meta_research_question": meta,
        "topic_connected": connected,
        "testability": 1.0 if testable else 0.0,
        "rejection_reasons": reasons,
    }


def validate_single_question(question, topic, records=None):
    question_text = _text(question)
    if placeholder_like(question_text):
        return ["question is placeholder or content-free"]
    if not question_text.endswith("?"):
        return ["question should be phrased as a question"]
    anchors = [topic]
    for record in records or []:
        anchors.append(record.get("title") or "")
        anchors.append(record.get("abstract") or "")
    if topic_overlap_score(question_text, anchors) < 0.2:
        return ["question is disconnected from topic/retrieved records"]
    validation = candidate_question_semantic_validation(question_text, topic, records)
    if not validation["substantive_question"]:
        return validation["rejection_reasons"]
    return []


class WebSearchProvider:
    provider_name = "disabled"

    def search(self, query, limit=5):
        raise RuntimeError("WEB_SEARCH_PROVIDER_NOT_CONFIGURED")

    def fetch(self, url):
        raise RuntimeError("WEB_FETCH_PROVIDER_NOT_CONFIGURED")


class SimpleHTTPWebProvider(WebSearchProvider):
    provider_name = "http"

    def search(self, query, limit=5):
        raise RuntimeError("WEB_SEARCH_PROVIDER_NOT_CONFIGURED")

    def fetch(self, url):
        import hashlib
        import urllib.request

        started = now_iso()
        req = urllib.request.Request(url, headers={"User-Agent": "researchGPT/0.1"})
        with urllib.request.urlopen(req, timeout=int(os.environ.get("RESEARCH_WEB_TIMEOUT", "15"))) as response:
            body = response.read(int(os.environ.get("RESEARCH_WEB_MAX_BYTES", "200000")))
            final_url = response.geturl()
            content_type = response.headers.get("content-type", "")
        text = body.decode("utf-8", errors="replace")
        return {
            "requested_url": url,
            "final_url": final_url,
            "retrieval_timestamp": started,
            "status": "SUCCESS",
            "content_type": content_type,
            "sha256": hashlib.sha256(body).hexdigest(),
            "text": text,
            "source": self.provider_name,
        }


def web_provider_from_env():
    if os.environ.get("RESEARCH_WEB_FETCH", "1") == "1":
        return SimpleHTTPWebProvider()
    return WebSearchProvider()


def safe_bash_command(command):
    lowered = str(command).lower()
    dangerous = ["sudo", "rm ", "rm\t", "mkfs", "dd ", "chmod 777", "chown", "apt ", "apt-get", "dnf ", "yum ", "pacman", "curl ", "wget ", "ssh ", "scp ", ">", ">>"]
    return bool(str(command).strip()) and not any(token in lowered for token in dangerous)


class GuidedToolController:
    def __init__(self, store, run_id, work_root, literature_provider, web_provider=None):
        self.store = store
        self.run_id = run_id
        self.work_root = Path(work_root) / run_id / "guided_tools"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.literature_provider = literature_provider
        self.web_provider = web_provider or web_provider_from_env()

    def execute(self, action):
        raw_action = deepcopy(action or {})
        action, normalization_rules = normalize_guided_action(action)
        tool = action.get("tool")
        args = action.get("arguments") or {}
        started = time.time()
        if tool == "literature_search":
            query = str(args.get("query", ""))
            result = self.literature_provider.search(query, limit=int(args.get("limit", os.environ.get("RESEARCH_LITERATURE_LIMIT", "5"))))
            return self._persist_tool_result(tool, args, result, self._compact_literature(result), started, raw_action, action, normalization_rules)
        if tool == "web_search":
            query = str(args.get("query", ""))
            result = self.web_provider.search(query, limit=int(args.get("limit", 5)))
            return self._persist_tool_result(tool, args, result, f"web_search returned {len(result.get('results', []))} results", started, raw_action, action, normalization_rules)
        if tool == "fetch_web_source":
            result = self.web_provider.fetch(str(args.get("url", "")))
            text = result.get("text", "")
            observation = f"fetched {result.get('final_url')} bytes={len(text.encode('utf-8'))} excerpt={text[:500]}"
            return self._persist_tool_result(tool, args, result, observation, started, raw_action, action, normalization_rules)
        if tool == "run_python":
            return self._run_python(str(args.get("code", "")), started)
        if tool == "run_bash":
            command = str(args.get("command", ""))
            if not safe_bash_command(command):
                return self._persist_tool_result(tool, args, {"status": "HUMAN_REQUIRED", "reason": "unsafe or privileged shell command"}, "shell command requires human approval", started, raw_action, action, normalization_rules)
            result = subprocess.run(command, cwd=self.work_root, shell=True, capture_output=True, text=True, timeout=int(os.environ.get("RESEARCH_TOOL_TIMEOUT", "30")))
            payload = {"command": command, "cwd": str(self.work_root), "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            return self._persist_tool_result(tool, args, payload, f"exit={result.returncode} stdout={result.stdout[:500]} stderr={result.stderr[:300]}", started, raw_action, action, normalization_rules)
        if tool == "read_artifact":
            path = str(args.get("path", ""))
            artifact_path = Path(self.store.get_artifact_path(self.run_id, path))
            text = artifact_path.read_text(encoding="utf-8", errors="replace")[:4000]
            return self._persist_tool_result(tool, args, {"path": path, "text": text}, f"read {path}: {text[:700]}", started, raw_action, action, normalization_rules)
        return self._persist_tool_result(tool or "unknown", args, {"status": "INVALID_TOOL"}, "invalid tool request", started, raw_action, action, normalization_rules)

    def _run_python(self, code, started):
        script = self.work_root / f"tool-{uuid4().hex[:8]}.py"
        script.write_text(code, encoding="utf-8")
        before = {p: p.stat().st_mtime_ns for p in self.work_root.rglob("*") if p.is_file()}
        result = subprocess.run(["python3", str(script)], cwd=self.work_root, capture_output=True, text=True, timeout=int(os.environ.get("RESEARCH_TOOL_TIMEOUT", "30")))
        after = {p: p.stat().st_mtime_ns for p in self.work_root.rglob("*") if p.is_file()}
        changed = [str(p.relative_to(self.work_root)) for p, mtime in after.items() if before.get(p) != mtime]
        payload = {"script": str(script), "cwd": str(self.work_root), "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "changed_files": changed}
        return self._persist_tool_result("run_python", {"code": code}, payload, f"exit={result.returncode} stdout={result.stdout[:500]} stderr={result.stderr[:300]} changed={changed[:5]}", started, {"tool": "run_python", "arguments": {"code": code}}, {"tool": "run_python", "arguments": {"code": code}}, [])

    def _persist_tool_result(self, tool, args, payload, observation, started, raw_action=None, normalized_action=None, normalization_rules=None):
        rel = f"guided_tools/{tool}-{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}.json"
        local = self.work_root / rel
        write_json(local, {
            "tool": tool,
            "arguments": args,
            "raw_model_action": raw_action or {},
            "normalized_action": normalized_action or {},
            "normalization_rules_applied": normalization_rules or [],
            "result": payload,
            "observation": observation,
            "duration_seconds": time.time() - started,
            "created_at": now_iso(),
        })
        artifact = self.store.put_artifact(self.run_id, local, rel, f"guided_tool_{tool}")
        return {"tool": tool, "artifact": artifact["path"], "observation": observation, "result": payload}

    def _compact_literature(self, result):
        records = result.get("records", [])
        lines = [f"literature_search returned {len(records)} records for query: {result.get('query')}"]
        for idx, record in enumerate(records[:5], 1):
            lines.append(f"{idx}. {record.get('title')} ({record.get('year')}) id={record.get('identifier')}")
        return "\n".join(lines)


def llm_attempts_for_stage(state, stage, limit=8):
    attempts = []
    for call in state.get("budget", {}).get("calls", []):
        if call.get("stage") != stage:
            continue
        attempts.append({
            "attempt_number": call.get("attempt_number"),
            "repair_attempt": call.get("repair_attempt", False),
            "actual_model": call.get("actual_model"),
            "selected_configuration_id": call.get("selected_configuration_id"),
            "weight_quantization": call.get("weight_quantization"),
            "kv_quantization": call.get("kv_quantization"),
            "status": call.get("status"),
            "failure_type": call.get("failure_type"),
            "schema_errors": call.get("schema_errors", []),
            "semantic_errors": call.get("semantic_errors", []),
            "assistant_generation_text": (call.get("assistant_generation_text") or call.get("isolated_assistant_text") or "")[:1000],
            "extraction_strategy": call.get("extraction_strategy"),
            "raw_response": (call.get("raw_response") or "")[:4000],
            "parsed_response": call.get("parsed_response"),
        })
    return attempts[-limit:]


def external_reasoning_context(state, node, reason):
    display_candidates, raw_candidates, candidate_warnings = candidate_question_context(state)
    return {
        "run_id": state["run_id"],
        "topic": state.get("topic"),
        "node_id": node["node_id"],
        "stage_objective": {
            "question_discovery": "Generate candidate research questions and search queries.",
            "question_refinement": "Select one feasible, scoped research question from candidates using retrieved literature.",
            "feasibility_analysis": "Assess feasibility and produce a research specification with evidence contracts.",
        }.get(node["node_id"], "Complete the blocked structured reasoning task."),
        "reason_local_inference_stopped": reason,
        "candidate_questions": display_candidates,
        "raw_candidate_questions": raw_candidates,
        "candidate_question_warnings": candidate_warnings,
        "selected_question": state.get("selected_question"),
        "prior_stage_outputs": {
            "search_strategy": state.get("search_strategy", {}),
            "question_refinement_rationale": state.get("question_refinement_rationale"),
            "evidence_discovery_attempts": state.get("evidence_discovery_attempts", []),
            "literature_relevance": state.get("literature_relevance", {}),
        },
        "criteria": {
            "question_refinement": [
                "support from retrieved research landscape",
                "local executability with available resources",
                "falsifiability/testability where appropriate",
                "clear scope",
                "no unsupported novelty or empirical claims",
            ],
            "feasibility_analysis": [
                "required evidence",
                "available tools/compute/data",
                "external resource dependence",
                "validation/falsification path",
                "replicability and bounded runtime/cost",
            ],
        }.get(node["node_id"], []),
        "constraints": [
            "Do not invent literature, measurements, experiments, citations, or evidence.",
            "This is planning/meta-research output, not empirical evidence for a paper.",
            "Return only JSON matching response_schema.json.",
            "Use only retrieved literature context supplied in this bundle.",
        ],
        "relevant_literature": minimal_literature_context(state.get("literature_cache", [])),
        "failed_local_attempts": llm_attempts_for_stage(state, node["node_id"]),
    }


def external_reasoning_prompt(context, schema):
    lines = [
        f"You are resolving the {context['node_id']} stage of a research run.",
        "",
        "Topic:",
        str(context.get("topic") or ""),
        "",
        "Stage objective:",
        context.get("stage_objective") or "",
        "",
        "Important constraints:",
    ]
    lines.extend(f"- {item}" for item in context.get("constraints", []))
    criteria = context.get("criteria", [])
    if criteria:
        lines.extend(["", "Evaluation criteria:"])
        lines.extend(f"- {item}" for item in criteria)
    candidates = context.get("candidate_questions", [])
    candidate_warnings = context.get("candidate_question_warnings", [])
    if candidate_warnings:
        lines.extend(["", "Candidate research question warnings:"])
        lines.extend(f"- {item}" for item in candidate_warnings)
        raw_candidates = context.get("raw_candidate_questions", [])
        if raw_candidates:
            lines.append(f"Raw persisted candidate_questions: {json.dumps(raw_candidates)}")
        lines.append("Because the prior candidate list is not substantive, propose/evaluate candidate questions from the topic and retrieved literature without inventing evidence.")
    if candidates:
        lines.extend(["", "Candidate research questions:"])
        for index, candidate in enumerate(candidates, 1):
            if isinstance(candidate, dict):
                question = candidate.get("question") or json.dumps(candidate, sort_keys=True)
                detail = {k: v for k, v in candidate.items() if k != "question"}
                lines.append(f"{index}. {question}")
                if detail:
                    lines.append(f"   Context: {json.dumps(detail, sort_keys=True)}")
            else:
                lines.append(f"{index}. {candidate}")
    literature = context.get("relevant_literature", [])
    if literature:
        lines.extend(["", "Relevant retrieved literature:"])
        for record in literature:
            identifier = record.get("identifier") or record.get("doi") or record.get("stable_url") or "unknown-id"
            title = record.get("title") or "untitled"
            year = record.get("year") or "unknown-year"
            excerpt = record.get("abstract_excerpt") or record.get("abstract") or ""
            lines.append(f"- [{identifier}] {title} ({year})")
            if excerpt:
                lines.append(f"  Abstract excerpt: {excerpt}")
    failed = context.get("failed_local_attempts", [])
    if failed:
        lines.extend(["", "Recent failed local model attempts and validator errors:"])
        for attempt in failed[-3:]:
            lines.append(
                f"- {attempt.get('actual_model') or 'no-model'} attempt={attempt.get('attempt_number')} "
                f"status={attempt.get('status')} failure={attempt.get('failure_type')}"
            )
            errors = (attempt.get("schema_errors") or []) + (attempt.get("semantic_errors") or [])
            if errors:
                lines.append(f"  Errors: {json.dumps(errors)}")
    lines.extend([
        "",
        "Expected structured response:",
        json.dumps(schema, indent=2, sort_keys=True),
        "",
        "Return ONLY one JSON object matching response_schema.json. Do not include markdown fences or prose outside JSON.",
    ])
    return "\n".join(lines) + "\n"


def external_reasoning_options(attempts, reason):
    if "ATOMIC_LOCAL_REASONING_EXHAUSTED" in reason:
        return {
            "question": "A bounded atomic planning judgment exhausted Q3/Q4 attempts. Provide that one structured judgment or stop for inspection?",
            "options": [
                {"id": "A", "description": "Provide the one requested atomic judgment using the persisted prompt.", "benefits": ["preserves supervisor-controlled assembly"], "risks": ["requires external scientific planning judgment"]},
                {"id": "B", "description": "Stop and inspect the atomic attempts.", "benefits": ["preserves all provenance"], "risks": ["run remains blocked"]},
            ],
            "recommended_option": "A", "recommendation_confidence": 0.8,
        }
    if "CANDIDATE_QUESTION_UNSUITABLE" in reason:
        return {
            "question": "Candidate-question regeneration was exhausted. Provide a substantive candidate question or stop for scientific review?",
            "options": [
                {"id": "A", "description": "Provide one substantive question grounded in the persisted literature.", "benefits": ["resumes refinement without repeating retrieval"], "risks": ["requires scientific judgment"]},
                {"id": "B", "description": "Stop and inspect candidate-generation history.", "benefits": ["preserves all attempts for diagnosis"], "risks": ["run remains blocked"]},
            ],
            "recommended_option": "A",
            "recommendation_confidence": 0.75,
        }
    if "INSUFFICIENT_RELEVANT_EVIDENCE" in reason:
        return {
            "question": "Bounded automatic literature query repair did not find usable topical records. Provide search guidance or stop the run?",
            "options": [
                {"id": "A", "description": "Provide corrected search terms or guidance as JSON/text for a future recovery pass.", "benefits": ["keeps the run grounded in real retrieved evidence"], "risks": ["run remains blocked until retrieval succeeds"]},
                {"id": "B", "description": "Stop and inspect attempted queries and relevance diagnostics.", "benefits": ["avoids unsupported research"], "risks": ["no further autonomous progress"]},
            ],
            "recommended_option": "B",
            "recommendation_confidence": 0.8,
        }
    failure_types = {attempt.get("failure_type") for attempt in attempts if attempt.get("failure_type")}
    weight_quants = {attempt.get("weight_quantization") for attempt in attempts if attempt.get("weight_quantization")}
    saw_quality = any(str(quant).startswith("Q4") for quant in weight_quants)
    infrastructure_failure = (
        "STRUCTURED_DECODING_CONFIGURATION_FAILURE" in failure_types
        or "UNKNOWN_TASK_CLASS" in reason
        or "NO_PROFILE_AVAILABLE" in reason
    )
    if infrastructure_failure:
        return {
            "question": "Local structured decoding or routing configuration blocked this node. Repair the engineering/configuration issue or provide the structured output manually?",
            "options": [
                {"id": "A", "description": "Repair the local inference/configuration issue and resume.", "benefits": ["keeps the scientific runtime automated"], "risks": ["requires repository or runtime configuration work"]},
                {"id": "B", "description": "Use the persisted prompt with ChatGPT and submit the result as free text.", "benefits": ["unblocks this run without changing models"], "risks": ["human answer must match schema and still pass validators"]},
            ],
            "recommended_option": "A",
            "recommendation_confidence": 0.8,
        }
    if saw_quality and ("SEMANTIC_VALIDATION_FAILURE" in failure_types or "SCHEMA_VALIDATION_FAILURE" in failure_types or "MODEL_OUTPUT_INVALID" in failure_types):
        return {
            "question": "Local Q3/Q4 inference exhausted this structured reasoning task. Provide the validated structured output?",
            "options": [
                {"id": "A", "description": "Use the persisted prompt with ChatGPT and submit the result as free text.", "benefits": ["unblocks the run"], "risks": ["human answer must match schema and pass semantic validators"]},
                {"id": "B", "description": "Stop and inspect the failed local outputs before deciding.", "benefits": ["preserves scientific caution"], "risks": ["run remains blocked"]},
            ],
            "recommended_option": "A",
            "recommendation_confidence": 0.75,
        }
    return {
        "question": "Local inference could not complete this node. Provide structured reasoning output or improve local model configuration?",
        "options": [
            {"id": "A", "description": "Use the persisted prompt with ChatGPT and submit the result as free text.", "benefits": ["unblocks the run"], "risks": ["human answer must match schema and pass semantic validators"]},
            {"id": "B", "description": "Install/register a stronger local model and resume.", "benefits": ["keeps inference local"], "risks": ["uses disk/RAM and may still fail"]},
        ],
        "recommended_option": "B",
        "recommendation_confidence": 0.6,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_external_reasoning_bundle(store, state, node, reason, response_contract=None):
    run_root = Path(store.run_root(state["run_id"]))
    bundle_id = f"{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}"
    bundle_rel = f"external_reasoning/{node['node_id']}/{bundle_id}"
    request_dir = run_root / bundle_rel
    request_dir.mkdir(parents=True, exist_ok=True)
    schema = deepcopy((response_contract or {}).get("response_schema") or response_schema_for_node(node["node_id"]))
    context = external_reasoning_context(state, node, reason)
    prompt_text = (response_contract or {}).get("prompt") or external_reasoning_prompt(context, schema)
    (request_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
    write_json(request_dir / "context.json", context)
    write_json(request_dir / "response_schema.json", schema)
    write_json(request_dir / "artifact_contract.json", artifact_contract_for_node(node))
    if response_contract:
        write_json(request_dir / "response_contract.json", response_contract)
    write_json(request_dir / "metadata.json", {"status": "EXTERNAL_REASONING_REQUIRED", "bundle_id": bundle_id,
        "bundle_path": bundle_rel, "response_kind": (response_contract or {}).get("response_kind", "NODE_LEVEL_RESPONSE"),
        "semantic_task": (response_contract or {}).get("semantic_task"), "created_at": now_iso()})
    artifacts = []
    names = ["prompt.md", "context.json", "response_schema.json", "artifact_contract.json", "metadata.json"]
    if response_contract:
        names.append("response_contract.json")
    for name in names:
        artifacts.append(store.put_artifact(state["run_id"], request_dir / name, f"{bundle_rel}/{name}", "local_model_router"))
    state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].extend(artifacts)
    return bundle_rel


def computational_measurement_decision_contract(state, node_id="question_refinement"):
    skeleton = state.get("computational_experimental_skeleton") or {}
    history = state.get("measurement_kind_attempt_history") or []
    unusable = [item.get("measurement_kind") for item in history
                if item.get("status") in {"UNUSABLE_SOURCE_MISMATCH", "UNUSABLE_REPLACEMENT"}]
    allowed = [kind for kind in MEASUREMENT_KIND_SCHEMA["properties"]["measurement_kind"]["enum"]
               if kind not in set(unusable) and kind != "uncertain"]
    if not skeleton.get("independent_variable") or not allowed:
        return None
    return {
        "contract_version": "atomic-semantic-response-v1",
        "response_kind": "ATOMIC_SEMANTIC_RESPONSE",
        "semantic_task": "computational_measurement_kind_selection",
        "response_schema": {"type": "object", "required": ["measurement_kind"], "additionalProperties": False,
            "properties": {"measurement_kind": {"type": "string", "enum": allowed}}},
        "validation_policy": {"schema": "JSON_SCHEMA_SUBSET", "downstream_semantic_validation": True,
                              "node_completion_on_acceptance": False},
        "continuation": {"type": "COMPUTATIONAL_MEASUREMENT_KIND_RECOVERY_V1", "node_id": node_id,
                         "state_refs": ["computational_experimental_skeleton", "measurement_kind_attempt_history"]},
        "prompt": ("Provide one neutral computational measurement kind for the persisted bounded experimental skeleton. "
                   "Return only one JSON object matching response_schema.json. Do not answer the research question."),
    }


def infer_atomic_response_contract(state, node, reason):
    # Migration is based on persisted suspended-state structure, not a decision id
    # or parsing a human-facing reason string.
    if (node.get("node_id") == "question_refinement"
            and state.get("computational_experimental_skeleton")
            and any(item.get("status") in {"UNUSABLE_SOURCE_MISMATCH", "UNUSABLE_REPLACEMENT"}
                    for item in state.get("measurement_kind_attempt_history", []))):
        return computational_measurement_decision_contract(state, node["node_id"])
    return None


def regenerate_external_reasoning_bundle(store, state, decision_id):
    for decision in list_open_decisions(state):
        if decision.get("decision_id") != decision_id:
            continue
        blocked_nodes = decision.get("blocked_nodes") or []
        if not blocked_nodes:
            raise ValueError(f"decision has no blocked node: {decision_id}")
        node = state.get("dag", {}).get("nodes", {}).get(blocked_nodes[0])
        if not node:
            raise KeyError(blocked_nodes[0])
        response_contract = decision.get("response_contract") or infer_atomic_response_contract(
            state, node, decision.get("why_human_is_needed", "human reasoning required"))
        if decision.get("response_kind") == "ATOMIC_SEMANTIC_RESPONSE" and not response_contract:
            raise ValueError(f"DECISION_CONTRACT_ERROR decision_id={decision_id} atomic response contract is missing")
        bundle_rel = write_external_reasoning_bundle(
            store, state, node, decision.get("why_human_is_needed", "human reasoning required"), response_contract)
        old_bundle = decision.get("external_reasoning_bundle")
        if old_bundle:
            decision.setdefault("superseded_external_reasoning_bundles", []).append(old_bundle)
        decision["external_reasoning_bundle"] = bundle_rel
        decision["evidence"] = [bundle_rel]
        if response_contract:
            decision["response_kind"] = "ATOMIC_SEMANTIC_RESPONSE"
            decision["response_contract"] = deepcopy(response_contract)
            decision["continuation"] = deepcopy(response_contract.get("continuation"))
            decision.setdefault("contract_amendments", []).append({
                "kind": "IMMUTABLE_BUNDLE_SUPERSESSION", "source_bundle": old_bundle,
                "replacement_bundle": bundle_rel, "contract_version": response_contract.get("contract_version"),
                "created_at": now_iso()})
        decision["updated_at"] = now_iso()
        state["updated_at"] = now_iso()
        return {"decision_id": decision_id, "external_reasoning_bundle": bundle_rel}
    raise KeyError(decision_id)


def apply_human_response_for_node(store, state, node_id, data):
    schema_errors, semantic_errors = validate_human_response_for_node(node_id, data, state)
    if schema_errors or semantic_errors:
        raise ValueError(json.dumps({"schema_errors": schema_errors, "semantic_errors": semantic_errors}, sort_keys=True))
    run_root = Path(store.run_root(state["run_id"]))
    if node_id == "question_refinement":
        state["selected_question"] = data["selected_question"]
        state["candidate_evaluations"] = data["candidate_evaluations"]
        state["question_refinement_rationale"] = data["rationale"]
        state.setdefault("known_limitations", []).extend(data.get("limitations", []))
        local_path = run_root / "human_responses" / node_id / f"{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}.json"
        write_json(local_path, data)
        artifact_path = f"human_responses/{node_id}/{local_path.name}"
        artifact = store.put_artifact(state["run_id"], local_path, artifact_path, "human_external_reasoning")
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
        complete_node(state, node_id, [artifact["path"]])
        return artifact
    if node_id == "question_discovery":
        state["candidate_questions"] = data["candidate_questions"]
        state["search_strategy"] = {"queries": data["search_queries"]}
        local_path = run_root / "human_responses" / node_id / f"{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}.json"
        write_json(local_path, data)
        artifact_path = f"human_responses/{node_id}/{local_path.name}"
        artifact = store.put_artifact(state["run_id"], local_path, artifact_path, "human_external_reasoning")
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
        complete_node(state, node_id, [artifact["path"]])
        return artifact
    if node_id == "feasibility_analysis":
        state["research_spec"] = data
        local_path = run_root / "human_responses" / node_id / f"{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}.json"
        write_json(local_path, data)
        artifact_path = f"human_responses/{node_id}/{local_path.name}"
        artifact = store.put_artifact(state["run_id"], local_path, artifact_path, "human_external_reasoning")
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
        complete_node(state, node_id, [artifact["path"]])
        return artifact
    return None


def validate_decision_response_contract(decision, data):
    kind = decision.get("response_kind")
    contract = decision.get("response_contract")
    if kind == "ATOMIC_SEMANTIC_RESPONSE":
        if not isinstance(contract, dict) or not isinstance(contract.get("response_schema"), dict):
            raise ValueError(
                f"DECISION_CONTRACT_ERROR decision_id={decision.get('decision_id')} "
                "response_kind=ATOMIC_SEMANTIC_RESPONSE response contract is missing")
        errors = validate_json_schema_subset(data, contract["response_schema"])
        if errors:
            raise ValueError(
                f"DECISION_RESPONSE_VALIDATION_FAILURE decision_id={decision.get('decision_id')} "
                f"response_kind=ATOMIC_SEMANTIC_RESPONSE errors={json.dumps(errors, sort_keys=True)}")
        return contract
    return None


def continue_atomic_decision_response(state, decision, data):
    """Install a validated semantic value at a stable supervisor continuation point."""
    continuation = decision.get("continuation") or (decision.get("response_contract") or {}).get("continuation")
    if not isinstance(continuation, dict) or not continuation.get("type"):
        raise ValueError(
            f"DECISION_CONTRACT_ERROR decision_id={decision.get('decision_id')} atomic continuation is missing")
    continuation_type = continuation["type"]
    if continuation_type == "COMPUTATIONAL_MEASUREMENT_KIND_RECOVERY_V1":
        skeleton = state.get("computational_experimental_skeleton")
        if not isinstance(skeleton, dict) or not skeleton.get("independent_variable"):
            raise ValueError(
                f"DECISION_CONTINUATION_ERROR decision_id={decision.get('decision_id')} "
                "persisted computational skeleton is unavailable")
        state["computational_measurement_recovery"] = {
            "skeleton": deepcopy(skeleton),
            "measurement_kind_reselection": True,
            "excluded_measurement_kinds": [item.get("measurement_kind") for item in
                state.get("measurement_kind_attempt_history", [])
                if item.get("status") in {"UNUSABLE_SOURCE_MISMATCH", "UNUSABLE_REPLACEMENT"}],
            "external_measurement_kind": data["measurement_kind"],
            "external_response_decision_id": decision["decision_id"],
            "original_candidate": skeleton.get("original_candidate"),
        }
        node_id = continuation.get("node_id") or "question_refinement"
        node = state.get("dag", {}).get("nodes", {}).get(node_id)
        if not node:
            raise ValueError(f"DECISION_CONTINUATION_ERROR decision_id={decision.get('decision_id')} node={node_id}")
        node["status"] = "PENDING"
        node["lease"] = None
        node["failure_reason"] = None
        return node_id
    raise ValueError(
        f"DECISION_CONTRACT_ERROR decision_id={decision.get('decision_id')} "
        f"unsupported continuation type={continuation_type}")


def persist_external_atomic_response(store, state, decision, option, raw_text, data, continuation_target):
    submitted_at = now_iso()
    record = {
        "decision_id": decision["decision_id"], "raw_response": raw_text,
        "parsed_response": deepcopy(data),
        "response_contract_version": decision["response_contract"].get("contract_version"),
        "validation_result": "ACCEPTED_FOR_SEMANTIC_CONTINUATION",
        "semantic_task": decision["response_contract"].get("semantic_task"),
        "origin": "EXTERNAL_REASONING", "submitted_option": option,
        "submitted_at": submitted_at, "source_bundle": decision.get("external_reasoning_bundle"),
        "continuation_target": continuation_target,
    }
    run_root = Path(store.run_root(state["run_id"]))
    local_path = run_root / "external_responses" / decision["decision_id"] / f"{submitted_at.replace(':', '').replace('.', '')}.json"
    write_json(local_path, record)
    artifact_path = f"external_responses/{decision['decision_id']}/{local_path.name}"
    artifact = store.put_artifact(state["run_id"], local_path, artifact_path, "human_external_reasoning")
    state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
    return artifact


def external_continuation_identity(decision_id, artifact):
    return f"{decision_id}:{artifact.get('sha256') or artifact.get('path')}"


def record_external_continuation(state, decision, artifact, parsed_response, continuation_target):
    identity = external_continuation_identity(decision["decision_id"], artifact)
    existing = next((item for item in state.get("external_decision_continuations", [])
                     if item.get("continuation_id") == identity), None)
    if existing:
        return existing
    timestamp = now_iso()
    record = {
        "continuation_id": identity, "decision_id": decision["decision_id"],
        "response_artifact": artifact["path"], "response_artifact_sha256": artifact.get("sha256"),
        "semantic_task": (decision.get("response_contract") or {}).get("semantic_task"),
        "response_kind": decision.get("response_kind"), "parsed_response": deepcopy(parsed_response),
        "response_validation": "PASSED", "continuation_type": (decision.get("continuation") or {}).get("type"),
        "continuation_target": continuation_target, "status": "CONTINUATION_PENDING",
        "events": [
            {"event": "RESPONSE_VALIDATED", "timestamp": timestamp},
            {"event": "RESPONSE_PERSISTED", "timestamp": timestamp, "artifact": artifact["path"]},
        ],
        "downstream_semantic_allowance": {
            "semantic_tasks": ["measurement_informativeness"], "used_semantic_tasks": [],
            "purpose": "validate an accepted external semantic value; not a replacement generation budget",
        },
        "created_at": timestamp, "updated_at": timestamp,
    }
    state.setdefault("external_decision_continuations", []).append(record)
    return record


def continuation_record(state, continuation_id):
    return next((item for item in state.get("external_decision_continuations", [])
                 if item.get("continuation_id") == continuation_id), None)


def add_node_failure_record(state, node_id, failure_class, reason, operation, artifact_refs=None,
                            recoverability="RECOVERABLE"):
    record = {"node_id": node_id, "failure_class": failure_class, "failure_reason": reason,
              "triggering_operation": operation, "relevant_artifact_refs": artifact_refs or [],
              "recoverability": recoverability, "timestamp": now_iso()}
    state.setdefault("node_failures", []).append(record)
    return record


def submit_research_decision(store, state, decision_id, option=None, free_text=None):
    decision = next((d for d in state.get("decisions", []) if d.get("decision_id") == decision_id), None)
    if not decision:
        raise KeyError(decision_id)
    if decision.get("status") != "WAITING_FOR_HUMAN":
        raise ValueError(f"decision already resolved: {decision_id}")
    if decision.get("external_reasoning_bundle") and option == "A":
        if not free_text:
            raise ValueError("structured external reasoning decisions require --text or --response-file")
        try:
            data = json.loads(free_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"human response is not valid JSON: {exc}") from exc
        contract = validate_decision_response_contract(decision, data)
        blocked_nodes = decision.get("blocked_nodes") or []
        if not blocked_nodes:
            raise ValueError(f"decision has no blocked node: {decision_id}")
        if contract:
            # Assemble all state changes on a copy so validation/continuation failures
            # leave the open decision fully resubmittable.
            next_state = deepcopy(state)
            next_decision = next(d for d in next_state["decisions"] if d.get("decision_id") == decision_id)
            continuation_target = continue_atomic_decision_response(next_state, next_decision, data)
            artifact = persist_external_atomic_response(
                store, next_state, next_decision, option, free_text, data, continuation_target)
            lifecycle = record_external_continuation(
                next_state, next_decision, artifact, data, continuation_target)
            next_state["computational_measurement_recovery"]["external_continuation_id"] = lifecycle["continuation_id"]
            next_decision["status"] = "RESOLVED"
            next_decision["selected_option"] = option
            next_decision["resolved_at"] = now_iso()
            next_decision["external_response_artifact"] = artifact["path"]
            next_decision["continuation_id"] = lifecycle["continuation_id"]
            next_state.setdefault("decision_history", []).append(deepcopy(next_decision))
            if not list_open_decisions(next_state):
                next_state["status"] = "PLANNED_RESEARCH"
            next_state["updated_at"] = now_iso()
            state.clear()
            state.update(next_state)
            return next_decision
        artifact = apply_human_response_for_node(store, state, blocked_nodes[0], data)
        decision["status"] = "RESOLVED"
        decision["selected_option"] = option
        decision["free_text"] = free_text
        decision["resolved_at"] = now_iso()
        decision["human_response_artifact"] = artifact["path"] if artifact else None
        state.setdefault("decision_history", []).append(deepcopy(decision))
        if not list_open_decisions(state):
            state["status"] = "PLANNED_RESEARCH"
        state["updated_at"] = now_iso()
        return decision
    return submit_decision(state, decision_id, option, free_text)


def current_local_backend_precondition():
    discovered = RuntimeRegistry().discover()
    selected = next((item for item in discovered
                     if item.get("available") and item.get("backend_compatibility") == "COMPATIBLE"), None)
    return {"status": "AVAILABLE" if selected else "UNAVAILABLE",
            "selected_backend": selected, "discovered_backends": discovered, "checked_at": now_iso()}


def reconcile_external_decision_continuation(store, state, decision_id=None, backend_probe=None):
    """Idempotently restore a persisted accepted atomic response to its continuation checkpoint."""
    candidates = [item for item in state.get("decisions", [])
                  if item.get("response_kind") == "ATOMIC_SEMANTIC_RESPONSE"
                  and item.get("status") == "RESOLVED" and item.get("external_response_artifact")]
    if decision_id:
        candidates = [item for item in candidates if item.get("decision_id") == decision_id]
    if not candidates:
        raise ValueError(f"no resolved atomic external response found for decision_id={decision_id}")
    results = []
    for decision in candidates:
        artifact_path = decision["external_response_artifact"]
        full_path = Path(store.run_root(state["run_id"])) / artifact_path
        response = json.loads(full_path.read_text(encoding="utf-8"))
        parsed = response["parsed_response"]
        validate_decision_response_contract(decision, parsed)
        descriptor = next((item for item in state.get("artifact_manifest", {}).get("artifacts", [])
                           if item.get("path") == artifact_path), None)
        if descriptor is None:
            raise ValueError(f"accepted external response is absent from artifact manifest: {artifact_path}")
        lifecycle = record_external_continuation(
            state, decision, descriptor, parsed,
            (decision.get("continuation") or {}).get("node_id") or "question_refinement")
        node_id = (decision.get("continuation") or {}).get("node_id") or "question_refinement"
        node = state["dag"]["nodes"][node_id]
        standard_measurement = measurement_from_kind(parsed.get("measurement_kind"))
        if standard_measurement and not lifecycle.get("semantic_handler_result"):
            lifecycle["semantic_handler_result"] = standard_measurement
            lifecycle.setdefault("events", []).append({"event": "STANDARD_MEASUREMENT_SEMANTICS_RECONCILED",
                "timestamp": now_iso(), "measurement_kind": parsed.get("measurement_kind"),
                "measurement": deepcopy(standard_measurement)})
        if lifecycle.get("status") == "CONTINUATION_APPLIED":
            results.append({"decision_id": decision["decision_id"], "continuation_id": lifecycle["continuation_id"],
                "action": "ALREADY_APPLIED", "previous_continuation_status": "CONTINUATION_APPLIED",
                "attempt_number": len(lifecycle.get("attempts", [])), "selected_backend": None,
                "resulting_continuation_status": "CONTINUATION_APPLIED", "response_artifact": artifact_path})
            continue
        unresolved_runtime_block = any(
            request.get("status") == "OPEN" and request.get("problem") == "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE"
            for request in state.get("engineering_requests", [])) and any(
                event.get("failure_class") == "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE"
                for event in lifecycle.get("events", []))
        if ((node.get("status") == "BLOCKED_ENGINEERING_REQUIRED"
             and lifecycle.get("status") in {"CONTINUATION_STARTED", "CONTINUATION_BLOCKED_ENGINEERING"})
                or unresolved_runtime_block):
            precondition = (backend_probe or current_local_backend_precondition)()
            previous_status = lifecycle.get("status")
            attempts = lifecycle.setdefault("attempts", [])
            attempt = {"attempt_number": len(attempts) + 1, "attempt_id": f"{lifecycle['continuation_id']}:attempt-{len(attempts) + 1}",
                       "previous_status": previous_status, "precondition_recheck": deepcopy(precondition),
                       "started_at": now_iso(), "status": "PRECONDITION_CHECKED"}
            attempts.append(attempt)
            lifecycle.setdefault("events", []).append({"event": "RECOVERY_PRECONDITION_CHECK",
                "timestamp": now_iso(), "attempt_id": attempt["attempt_id"], "result": deepcopy(precondition)})
            allowance = lifecycle.get("downstream_semantic_allowance") or {}
            allowance["used_semantic_tasks"] = [task for task in allowance.get("used_semantic_tasks", [])
                                                 if task != "measurement_informativeness"]
            state.pop("active_external_continuation", None)
            if precondition.get("status") != "AVAILABLE":
                attempt["status"] = "SKIPPED_STILL_BLOCKED"
                attempt["completed_at"] = now_iso()
                lifecycle["status"] = "CONTINUATION_BLOCKED_ENGINEERING"
                lifecycle["continuation_validation_result"] = "NOT_COMPLETED"
                lifecycle["updated_at"] = now_iso()
                node["status"] = "BLOCKED_ENGINEERING_REQUIRED"; node["lease"] = None
                node["failure_reason"] = lifecycle.get("failure_reason") or "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE"
                state["status"] = "BLOCKED_ENGINEERING_REQUIRED"
                results.append({"decision_id": decision["decision_id"], "continuation_id": lifecycle["continuation_id"],
                    "action": "SKIPPED_STILL_BLOCKED", "previous_continuation_status": previous_status,
                    "precondition_recheck": precondition, "attempt_number": attempt["attempt_number"],
                    "selected_backend": None, "resulting_continuation_status": lifecycle["status"],
                    "response_artifact": artifact_path})
                continue
            attempt["status"] = "RETRY_READY"
            lifecycle["status"] = "CONTINUATION_PENDING"
            lifecycle["failure_reason"] = None
            lifecycle["updated_at"] = now_iso()
            lifecycle.setdefault("events", []).append({"event": "RECOVERY_PRECONDITION_SATISFIED",
                "timestamp": now_iso(), "attempt_id": attempt["attempt_id"],
                "selected_backend": deepcopy(precondition.get("selected_backend"))})
            continue_atomic_decision_response(state, decision, parsed)
            state["computational_measurement_recovery"]["external_continuation_id"] = lifecycle["continuation_id"]
            state["computational_measurement_recovery"]["continuation_attempt_id"] = attempt["attempt_id"]
            node["status"] = "PENDING"; node["lease"] = None; node["failure_reason"] = None
            state["status"] = "PLANNED_RESEARCH"
            results.append({"decision_id": decision["decision_id"], "continuation_id": lifecycle["continuation_id"],
                "action": "RETRY_STARTED", "previous_continuation_status": previous_status,
                "precondition_recheck": precondition, "attempt_number": attempt["attempt_number"],
                "selected_backend": precondition.get("selected_backend"),
                "resulting_continuation_status": lifecycle["status"], "response_artifact": artifact_path})
            continue
        if node.get("status") == "FAILED":
            add_node_failure_record(state, node_id, "LEGACY_CONTINUATION_TERMINAL_FAILURE",
                node.get("failure_reason") or "terminal failure without reason",
                (decision.get("continuation") or {}).get("type"), [artifact_path], "RECOVERED")
        if lifecycle.get("status") not in {"CONTINUATION_APPLIED", "CONTINUATION_VALIDATED"}:
            continue_atomic_decision_response(state, decision, parsed)
            state["computational_measurement_recovery"]["external_continuation_id"] = lifecycle["continuation_id"]
            lifecycle["status"] = "CONTINUATION_PENDING"
            lifecycle["updated_at"] = now_iso()
            lifecycle.setdefault("events", []).append({"event": "CONTINUATION_RECONCILED", "timestamp": now_iso()})
            state["status"] = "PLANNED_RESEARCH"
        results.append({"decision_id": decision["decision_id"], "continuation_id": lifecycle["continuation_id"],
                        "action": "ALREADY_APPLIED" if lifecycle["status"] == "CONTINUATION_APPLIED" else "RETRY_STARTED",
                        "resulting_continuation_status": lifecycle["status"], "response_artifact": artifact_path})
    state["updated_at"] = now_iso()
    return results


def llm_task_class_for_node(node_id):
    return NODE_LLM_TASK_CLASS.get(node_id)


def validate_node_llm_task_class(node):
    task_class = node.get("llm_task_class") or llm_task_class_for_node(node["node_id"])
    node["llm_task_class"] = task_class
    handler_uses_llm = node["node_id"] in {"question_discovery", "question_refinement", "feasibility_analysis"}
    if not handler_uses_llm:
        return None
    if not task_class:
        return (
            f"UNKNOWN_LLM_TASK_CLASS node={node['node_id']} requested_class=None "
            f"available_classes={sorted(CANONICAL_LLM_TASK_CLASSES)}"
        )
    if task_class not in CANONICAL_LLM_TASK_CLASSES:
        return (
            f"UNKNOWN_LLM_TASK_CLASS node={node['node_id']} requested_class={task_class} "
            f"available_classes={sorted(CANONICAL_LLM_TASK_CLASSES)}"
        )
    return None


def invalidate_stale_local_routing_decisions(state):
    invalidated = []
    for decision in state.get("decisions", []):
        if decision.get("status") != "WAITING_FOR_HUMAN":
            continue
        why = decision.get("why_human_is_needed", "")
        if "no eligible local model configuration" not in why and "NO_ELIGIBLE_LOCAL_MODEL" not in why:
            continue
        blocked = decision.get("blocked_nodes", [])
        for node_id in blocked:
            node = state.get("dag", {}).get("nodes", {}).get(node_id)
            if not node:
                continue
            task_class = llm_task_class_for_node(node_id) or node.get("llm_task_class")
            if task_class in CANONICAL_LLM_TASK_CLASSES and node.get("status") == "WAITING_FOR_HUMAN":
                node["status"] = "PENDING"
                node["lease"] = None
                node["failure_reason"] = None
                node["llm_task_class"] = task_class
                decision["status"] = "INVALIDATED_ROUTING_FIX"
                decision["invalidated_at"] = now_iso()
                decision["invalidation_reason"] = "canonical LLM task mapping was added after this local-routing decision was created"
                invalidated.append(decision["decision_id"])
    if invalidated:
        state.setdefault("decision_history", []).append({
            "event": "INVALIDATED_STALE_LOCAL_ROUTING_DECISIONS",
            "decision_ids": invalidated,
            "created_at": now_iso(),
        })
        if not any(d.get("status") == "WAITING_FOR_HUMAN" for d in state.get("decisions", [])):
            state["status"] = "PLANNED_RESEARCH"
        state["updated_at"] = now_iso()
    return invalidated


def repair_recoverable_structured_generation_failures(state):
    repaired = []
    invalidated = []
    dependency_invalidations = []
    skeleton = state.get("computational_experimental_skeleton") or {}
    if skeleton and not skeleton.get("clarification"):
        preserved_clarification = next((deepcopy(item) for item in reversed(state.get("candidate_clarification_history", []))
                                        if item.get("rewritten_candidate") == skeleton.get("candidate_question")), None)
        if preserved_clarification:
            skeleton["clarification"] = preserved_clarification
    skeleton_errors = (experimental_skeleton_validation(skeleton) if isinstance(skeleton.get("independent_variable"), dict)
                       else ["legacy untyped variable/measurement representation is stale"]) if skeleton else []
    active_measurement = skeleton.get("dependent_measurement", {}) if skeleton else {}
    active_observable = active_measurement.get("measurement_observable", {}) if isinstance(active_measurement, dict) else {}
    normalized_active_observable = normalize_semantic_value(active_observable.get("display_text")) if active_observable else None
    if (normalized_active_observable and active_measurement.get("measurement_kind") in {"output_value", "other"}
            and normalized_active_observable["normalized_display_text"] != active_observable.get("display_text")):
        skeleton_errors.append("parameterized measurement observable requires deterministic surface normalization")
    refinement_node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
    if (refinement_node and refinement_node.get("status") == "FAILED"
            and refinement_node.get("failure_reason") == "MAX_AGENT_ITERATIONS"
            and state.get("candidate_questions")):
        refinement_node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
        repaired.append("question_refinement")
        dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
    if skeleton and skeleton_errors and refinement_node and refinement_node.get("status") == "COMPLETED":
        state.setdefault("computational_experimental_skeleton_history", []).append({
            "skeleton": deepcopy(skeleton), "validation_errors": skeleton_errors,
            "invalidation_reason": "operational semantic-role validation repair", "created_at": now_iso()})
        if (isinstance(skeleton.get("dependent_measurement"), dict)
                and skeleton["dependent_measurement"].get("measurement_kind") in {"output_value", "other"}):
            normalized_measurement = measurement_from_kind(
                skeleton["dependent_measurement"]["measurement_kind"],
                observable_display=skeleton["dependent_measurement"].get("measurement_observable", {}).get("display_text"))
            if normalized_measurement:
                normalized_measurement["informativeness"] = skeleton["dependent_measurement"].get("informativeness")
            source_compatible = bool(normalized_measurement and measurement_source_compatible(
                normalized_measurement.get("measurement_kind"), normalized_measurement.get("measurement_source")))
            state.setdefault("measurement_kind_attempt_history", []).append({
                "measurement_kind": skeleton["dependent_measurement"].get("measurement_kind"),
                "observable": deepcopy(skeleton["dependent_measurement"].get("measurement_observable")),
                "previous_informativeness": skeleton["dependent_measurement"].get("informativeness"),
                "reevaluated_source": normalized_measurement.get("measurement_source") if normalized_measurement else "UNCERTAIN",
                "status": "PRESERVED_ACCEPTED" if source_compatible else "UNUSABLE_SOURCE_MISMATCH",
                "failure_reason": None if source_compatible else "measurement kind/source incompatibility",
                "created_at": now_iso()})
            state["computational_measurement_recovery"] = {"skeleton": deepcopy(skeleton),
                "original_candidate": skeleton.get("original_candidate"), "repair_reason": "measurement observable/informativeness required",
                "recovered_measurement": normalized_measurement if source_compatible else None,
                "measurement_kind_reselection": not source_compatible,
                "excluded_measurement_kinds": [skeleton["dependent_measurement"].get("measurement_kind")] if not source_compatible else [],
                "created_at": now_iso()}
        elif (skeleton.get("candidate_disposition") == "REGENERATED_SKELETON_FIRST"
              and semantic_display_errors(skeleton.get("independent_variable", {}).get("display_text"), "independent variable")):
            state["fresh_skeleton_regeneration_count"] = 0
            state["computational_measurement_recovery"] = {
                "skeleton": deepcopy(skeleton), "original_candidate": skeleton.get("original_candidate"),
                "skip_current_candidate_recovery": True,
                "retirement_reason": "fresh variable was an identifier placeholder rejected by strengthened normalization",
                "created_at": now_iso()}
        state.pop("computational_experimental_skeleton", None)
        clear_node_outputs(state, "question_refinement")
        refinement_node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
        repaired.append("question_refinement")
        dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
    candidates = state.get("candidate_questions") or []
    if candidates and state.get("prior_work_coverage", {}).get("status") in {"UNKNOWN", "UNAVAILABLE"}:
        question = candidates[0].get("question", "") if isinstance(candidates[0], dict) else str(candidates[0])
        errors = validate_bounded_computational_question(question)
        node = state.get("dag", {}).get("nodes", {}).get("question_discovery")
        if errors and node and node.get("status") == "COMPLETED":
            state.setdefault("candidate_question_history", []).append({
                "candidate_questions": deepcopy(candidates), "validation_errors": errors,
                "trigger": "bounded_computational_semantic_revalidation", "created_at": now_iso()})
            state["candidate_questions"] = []
            state.get("candidate_evidence_contracts", {}).pop(question, None)
            node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
            repaired.append("question_discovery")
            dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_discovery"))
    for decision in state.get("decisions", []):
        reason = decision.get("why_human_is_needed", "")
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and reason == "NO_COHERENT_COMPUTATIONAL_CANDIDATE semantic_task=measurement_kind_selection"):
            calls = state.get("budget", {}).get("calls", [])
            kind_index = next((i for i in range(len(calls) - 1, -1, -1)
                               if isinstance(calls[i].get("parsed_response"), dict)
                               and calls[i]["parsed_response"].get("measurement_kind")), None)
            info_call = next((call for call in calls[(kind_index + 1 if kind_index is not None else len(calls)):]
                              if isinstance(call.get("parsed_response"), dict)
                              and call["parsed_response"].get("assessment") in
                                  {"INFORMATIVE", "WEAKLY_INFORMATIVE", "UNINFORMATIVE", "UNCERTAIN"}), None)
            kind = calls[kind_index]["parsed_response"]["measurement_kind"] if kind_index is not None else None
            measurement = measurement_from_kind(kind) if kind else None
            if not measurement or not info_call:
                continue
            measurement["informativeness"] = info_call["parsed_response"]["assessment"]
            prior = deepcopy((state.get("computational_experimental_skeleton_history") or [{}])[-1].get("skeleton") or {})
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
            state["computational_measurement_recovery"] = {
                "skeleton": prior, "original_candidate": prior.get("original_candidate"),
                "recovered_measurement": measurement, "measurement_kind_reselection": True,
                "excluded_measurement_kinds": ["output_value"],
                "repair_reason": "persist and adjudicate completed bounded replacement measurement attempt",
                "created_at": now_iso()}
            decision["status"] = "INVALIDATED_MEASUREMENT_RESELECTION_PERSISTENCE_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "completed replacement measurement is now preserved before atomic escalation"
            invalidated.append(decision["decision_id"])
            continue
        current_skeleton = state.get("computational_experimental_skeleton") or {}
        current_measurement = current_skeleton.get("dependent_measurement") or {}
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and current_skeleton.get("candidate_disposition") == "REGENERATED_SKELETON_FIRST"
                and observable_specificity_errors((current_measurement.get("measurement_observable") or {}).get("display_text"),
                                                  current_measurement.get("measurement_kind"))):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            variable = current_skeleton.get("independent_variable") or {}
            recovered = semantic_value_record(variable.get("raw_model_value") or variable.get("display_text"))
            recovered.update({"semantic_source": "HISTORICAL_MODEL_OUTPUT_REEVALUATED",
                              "original_attempt_id": (state.get("computational_measurement_recovery") or {}).get(
                                  "recovered_historical_variable", {}).get("original_attempt_id"),
                              "semantic_validation_status": "VALID"})
            state.setdefault("computational_experimental_skeleton_history", []).append({
                "skeleton": deepcopy(current_skeleton), "validation_errors": current_skeleton.get("validation_errors", []),
                "invalidation_reason": "parameterized observable category echo normalized before specificity validation",
                "created_at": now_iso()})
            state["fresh_skeleton_regeneration_count"] = 0
            state["computational_measurement_recovery"] = {
                "skeleton": deepcopy(current_skeleton), "original_candidate": current_skeleton.get("original_candidate"),
                "skip_current_candidate_recovery": True, "recovered_historical_variable": recovered,
                "recovered_control_type": variable.get("control_type"),
                "recovered_measurement_kind": current_measurement.get("measurement_kind"),
                "retirement_reason": "retry only the normalized parameterized observable", "created_at": now_iso()}
            decision["status"] = "INVALIDATED_OBSERVABLE_SURFACE_NORMALIZATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "observable category echoes are now checked after deterministic surface normalization"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("decision_id") == "D34131d1d"
                and decision.get("blocked_nodes") == ["question_refinement"]):
            historical = []
            for index, call in enumerate(state.get("budget", {}).get("calls", [])):
                if (call.get("status") == "FAILED" and call.get("schema", {}).get("required") == ["variable"]
                        and isinstance(call.get("parsed_response"), dict) and "variable" in call["parsed_response"]):
                    historical.append((index, call))
            historical = historical[-3:]
            reevaluations = []
            for index, call in historical:
                record = semantic_value_record(call["parsed_response"]["variable"], "independent_variable")
                record.update({"semantic_source": "HISTORICAL_MODEL_OUTPUT_REEVALUATED",
                               "original_attempt_id": f"budget_call:{index}", "original_attempt_status": call.get("status"),
                               "reevaluated_at": now_iso()})
                reevaluations.append(record)
            state.setdefault("semantic_value_reevaluations", []).append({
                "decision_id": decision["decision_id"], "field": "independent_variable",
                "records": deepcopy(reevaluations), "created_at": now_iso()})
            recovered = next((item for item in reevaluations if item["semantic_validation_status"] == "VALID"), None)
            if not recovered:
                continue
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            state["fresh_skeleton_regeneration_count"] = 0
            state["computational_measurement_recovery"] = {
                "skeleton": deepcopy((state.get("computational_experimental_skeleton_history") or [{}])[-1].get("skeleton") or {}),
                "original_candidate": (state.get("retired_candidate_history") or [{}])[-1].get("candidate", {}).get("question"),
                "skip_current_candidate_recovery": True, "recovered_historical_variable": deepcopy(recovered),
                "retirement_reason": "resume fresh skeleton from normalized historical semantic value", "created_at": now_iso()}
            decision["status"] = "INVALIDATED_SEMANTIC_SURFACE_NORMALIZATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "historical fresh-variable output became usable after deterministic surface normalization and role validation"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and "derived-variable source display text is missing" in reason
                and state.get("computational_experimental_skeleton", {}).get("candidate_disposition") == "REGENERATED_SKELETON_FIRST"):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            failed = deepcopy(state.get("computational_experimental_skeleton") or {})
            state.setdefault("computational_experimental_skeleton_history", []).append({
                "skeleton": failed, "validation_errors": failed.get("validation_errors", []),
                "invalidation_reason": "fresh derived-variable source microstep was not executed",
                "created_at": now_iso()})
            state["fresh_skeleton_regeneration_count"] = 0
            state["computational_measurement_recovery"] = {
                "skeleton": failed, "original_candidate": failed.get("original_candidate"),
                "skip_current_candidate_recovery": True,
                "retirement_reason": "continue the same bounded fresh attempt with derived-source validation",
                "created_at": now_iso()}
            decision["status"] = "INVALIDATED_DERIVED_SOURCE_MICROSTEP_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "fresh skeleton generation now rejects identifier placeholders and generates required derived source controls"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and "experimental_skeleton missing semantic field" in reason
                and state.get("computational_experimental_skeleton", {}).get("candidate_disposition") == "RECOVERED"
                and state.get("computational_experimental_skeleton", {}).get("dependent_measurement", {}).get("informativeness")
                    in {"WEAKLY_INFORMATIVE", "UNINFORMATIVE", "UNCERTAIN"}):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            failed = deepcopy(state.get("computational_experimental_skeleton") or {})
            state.setdefault("computational_experimental_skeleton_history", []).append({
                "skeleton": failed, "validation_errors": failed.get("validation_errors", []),
                "invalidation_reason": "uninformative observable now activates bounded fresh skeleton fallback",
                "created_at": now_iso()})
            state["fresh_skeleton_regeneration_count"] = 0
            state["computational_measurement_recovery"] = {
                "skeleton": failed, "original_candidate": failed.get("original_candidate"),
                "skip_current_candidate_recovery": True,
                "retirement_reason": "; ".join(failed.get("validation_errors", [])),
                "created_at": now_iso()}
            decision["status"] = "INVALIDATED_MEASUREMENT_INFORMATIVENESS_FALLBACK_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "failed current-candidate observable recovery now proceeds to the single bounded skeleton-first regeneration"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=fresh_variable_generation" in reason
                and any(call.get("schema", {}).get("required") == ["variable"] for call in state.get("budget", {}).get("calls", [])[-3:])):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
            state["fresh_skeleton_regeneration_count"] = 0
            state["computational_measurement_recovery"] = {"skeleton": deepcopy(
                (state.get("computational_experimental_skeleton_history") or [{}])[-1].get("skeleton") or {}),
                "original_candidate": (state.get("retired_candidate_history") or [{}])[-1].get("candidate", {}).get("question"),
                "repair_reason": "continue same bounded fresh regeneration with semantic retries", "created_at": now_iso()}
            decision["status"] = "INVALIDATED_FRESH_VARIABLE_SEMANTIC_RETRY_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "fresh variable validation now participates in bounded Q3/retry/Q4 handling"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("decision_id") == "Dccc54b84"
                and decision.get("blocked_nodes") == ["question_refinement"]):
            invalid_skeleton = deepcopy(state.get("computational_experimental_skeleton") or
                                        (state.get("computational_experimental_skeleton_history") or [{}])[-1].get("skeleton") or {})
            state["computational_measurement_recovery"] = {"skeleton": invalid_skeleton,
                "original_candidate": invalid_skeleton.get("original_candidate") or
                    (state.get("candidate_question_history") or [{}])[0].get("candidate_questions", [{}])[0].get("question"),
                "decision_id": decision["decision_id"], "created_at": now_iso()}
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            decision["status"] = "INVALIDATED_MEASUREMENT_SELECTION_SKELETON_FIRST_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "invalid dependent measurement now uses enum-first field recovery and one skeleton-first fallback"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("decision_id") == "D42d2453b"
                and decision.get("blocked_nodes") == ["question_refinement"]):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            decision["status"] = "INVALIDATED_COMPUTATIONAL_TESTABILITY_ATOMICIZATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "computational refinement now uses typed variable, measurement, controllability, and testability microsteps"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and "SEMANTIC_VALIDATION_FAILURE local structured generation exhausted" in reason
                and any(call.get("stage") == "question_refinement" and isinstance(call.get("parsed_response"), dict)
                        and "measurement" in call.get("parsed_response", {}) for call in state.get("budget", {}).get("calls", [])[-6:])):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_refinement"))
            decision["status"] = "INVALIDATED_COMPUTATIONAL_OPERATIONALIZATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "failed measurement extraction now triggers the bounded candidate-clarification path"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and "STRUCTURED_OUTPUT_INCOMPLETE local structured generation exhausted" in reason
                and any(call.get("schema", {}).get("required") == ["variable", "control_type"]
                        for call in state.get("budget", {}).get("calls", [])[-4:])):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
            decision["status"] = "INVALIDATED_TYPED_SKELETON_FAILURE_CLASSIFICATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "atomic typed-skeleton failures now identify the exact missing semantic field"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_refinement"]
                and "ATOMIC_LOCAL_REASONING_EXHAUSTED missing semantic field: independent_variable_contract" in reason
                and any(call.get("schema", {}).get("required") == ["variable", "control_type"]
                        for call in state.get("budget", {}).get("calls", [])[-6:])):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_refinement")
            decision["status"] = "INVALIDATED_SINGLE_SEMANTIC_VALUE_ATOMICIZATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "variable display text and control type are now separate atomic calls"
            invalidated.append(decision["decision_id"])
            continue
        if (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_discovery"]
                and "INSUFFICIENT_RELEVANT_EVIDENCE guided search executed" in reason):
            node = state.get("dag", {}).get("nodes", {}).get("question_discovery")
            if node:
                node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                repaired.append("question_discovery")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_discovery"))
            attempts = state.get("research_source_attempts", [])
            if not attempts:
                guided = [item for item in state.get("guided_agent_steps", []) if item.get("microstep") == "literature_search"]
                attempts = [{"provider_id": getattr(state, "literature_provider", "preserved_provider"),
                             "status": "ZERO_RESULTS" if not item.get("record_count") else "RESULTS_IRRELEVANT",
                             "raw_query": item.get("raw_query"), "normalized_query": item.get("normalized_query"),
                             "record_count": item.get("record_count", 0), "retrieval_timestamp": item.get("created_at")}
                            for item in guided]
                state["research_source_attempts"] = attempts
            state["prior_work_coverage"] = classify_prior_work_coverage(attempts)
            decision["status"] = "INVALIDATED_LITERATURE_DEPENDENCY_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "bounded retrieval failure no longer asserts nonexistence or blocks high-closure computational question generation"
            invalidated.append(decision["decision_id"])
        elif (decision.get("status") == "WAITING_FOR_HUMAN" and decision.get("blocked_nodes") == ["question_discovery"]
              and "SEMANTIC_VALIDATION_FAILURE local structured generation exhausted" in reason):
            preserved_call = next((call for call in reversed(state.get("budget", {}).get("calls", []))
                                   if call.get("stage") == "question_discovery"
                                   and isinstance(call.get("parsed_response"), dict)
                                   and not validate_bounded_computational_question(call["parsed_response"].get("question"))), None)
            if preserved_call:
                state["candidate_generation_resume"] = {
                    "question": preserved_call["parsed_response"]["question"],
                    "actual_model": preserved_call.get("actual_model"), "attempt": deepcopy(preserved_call),
                }
                node = state.get("dag", {}).get("nodes", {}).get("question_discovery")
                if node:
                    node.update({"status": "PENDING", "lease": None, "failure_reason": None, "attempts": 0})
                    repaired.append("question_discovery")
                    dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_discovery"))
                decision["status"] = "INVALIDATED_COMPUTATIONAL_QUESTION_PROTOCOL_REPAIR"
                decision["invalidated_at"] = now_iso()
                decision["invalidation_reason"] = "preserved model output passes the corrected generic bounded-computation validator"
                invalidated.append(decision["decision_id"])
    handoff_repair = repair_guided_candidate_question_handoff(state)
    if handoff_repair:
        dependency_invalidations.extend(mark_dependent_nodes_pending(state, "question_discovery"))
        repaired.extend(handoff_repair.get("reset_nodes", []))
    feasibility_node = state.get("dag", {}).get("nodes", {}).get("feasibility_analysis")
    feasibility_routes = state.get("feasibility_routes", [])
    if feasibility_node and feasibility_node.get("status") == "COMPLETED" and feasibility_routes:
        fit_errors = [{"route": route.get("approach"), "errors": errors} for route in feasibility_routes
                      if (errors := validate_feasibility_fit(route.get("scientific_fit", {}), state.get("topic")))]
        requirement_errors = [
            {"requirement": item, "errors": validate_feasibility_requirement(item)}
            for route in feasibility_routes for item in route.get("model_proposed_requirements", [])
            if validate_feasibility_requirement(item)
        ]
        if fit_errors or requirement_errors:
            state.setdefault("feasibility_history", []).append({
                "research_spec": deepcopy(state.get("research_spec")),
                "routes": deepcopy(feasibility_routes), "resource_matches": deepcopy(state.get("feasibility_resource_matches", [])),
                "invalidation_reason": "atomic feasibility semantic validation tightened",
                "validation_errors": fit_errors, "requirement_validation_errors": requirement_errors, "created_at": now_iso(),
            })
            state["feasibility_resume_context"] = {
                "input_snapshot": deepcopy(state.get("feasibility_input_snapshot")),
                "operationalization": state.get("feasibility_operationalization"),
                "route": deepcopy(feasibility_routes[0].get("route") or {
                    "approach": feasibility_routes[0].get("approach"), "reason": feasibility_routes[0].get("raw_reason", "")}),
                "scientific_fit": deepcopy(feasibility_routes[0].get("scientific_fit")),
                "skip_optional_requirements": True, "skip_alternative_route": True,
                "origin": "atomic_feasibility_semantic_revalidation", "created_at": now_iso(),
            }
            feasibility_node["status"] = "PENDING"
            feasibility_node["lease"] = None
            feasibility_node["failure_reason"] = None
            feasibility_node["attempts"] = 0
            clear_node_outputs(state, "feasibility_analysis")
            repaired.append("feasibility_analysis")
            dependency_invalidations.extend(mark_dependent_nodes_pending(state, "feasibility_analysis"))
    for decision in state.get("decisions", []):
        if (
            decision.get("status") == "WAITING_FOR_HUMAN"
            and decision.get("blocked_nodes") == ["feasibility_analysis"]
            and "NO_ELIGIBLE_LOCAL_MODEL task_class=research_feasibility_analysis" in decision.get("why_human_is_needed", "")
        ):
            node = state.get("dag", {}).get("nodes", {}).get("feasibility_analysis")
            if node:
                node["status"] = "PENDING"
                node["lease"] = None
                node["failure_reason"] = None
                node["attempts"] = 0
                clear_node_outputs(state, "feasibility_analysis")
                repaired.append("feasibility_analysis")
                dependency_invalidations.extend(mark_dependent_nodes_pending(state, "feasibility_analysis"))
            decision["status"] = "INVALIDATED_FEASIBILITY_ATOMICIZATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "high-guidance feasibility now uses vetted atomic planning calls and supervisor assembly"
            invalidated.append(decision["decision_id"])
    for decision in state.get("decisions", []):
        if (
            decision.get("status") == "WAITING_FOR_HUMAN"
            and decision.get("blocked_nodes") == ["feasibility_analysis"]
            and "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=feasibility_scientific_fit"
                in decision.get("why_human_is_needed", "")
        ):
            preserved = next((entry for entry in reversed(state.get("feasibility_history", [])) if entry.get("routes")), None)
            prior_route = deepcopy(preserved["routes"][0]) if preserved else None
            prior_fit = prior_route.get("scientific_fit", {}) if prior_route else {}
            if prior_route and not validate_feasibility_fit(prior_fit, state.get("topic")):
                state["feasibility_resume_context"] = {
                    "input_snapshot": deepcopy(state.get("feasibility_input_snapshot") or preserved.get("input_snapshot")),
                    "operationalization": state.get("feasibility_operationalization") or preserved.get("operationalization"),
                    "route": deepcopy(prior_route.get("route") or {"approach": prior_route.get("approach"), "reason": prior_route.get("raw_reason", "")}),
                    "scientific_fit": deepcopy(prior_fit), "origin": "preserved_valid_scientific_fit_reuse", "created_at": now_iso(),
                    "skip_optional_requirements": True, "skip_alternative_route": True,
                }
                node = state.get("dag", {}).get("nodes", {}).get("feasibility_analysis")
                if node:
                    node["status"] = "PENDING"; node["lease"] = None; node["failure_reason"] = None; node["attempts"] = 0
                    clear_node_outputs(state, "feasibility_analysis"); repaired.append("feasibility_analysis")
                decision["status"] = "INVALIDATED_REDUNDANT_ATOMIC_FIT_RETRY"
                decision["invalidated_at"] = now_iso()
                decision["invalidation_reason"] = "the unchanged route already had a preserved semantically valid scientific-fit judgment"
                invalidated.append(decision["decision_id"])
    feasibility_node = state.get("dag", {}).get("nodes", {}).get("feasibility_analysis")
    if feasibility_node and feasibility_node.get("status") == "FAILED" and feasibility_node.get("failure_reason") == "MAX_AGENT_ITERATIONS":
        preserved = next((entry for entry in reversed(state.get("feasibility_history", [])) if entry.get("routes")), None)
        prior_route = deepcopy(preserved["routes"][0]) if preserved else None
        prior_fit = prior_route.get("scientific_fit", {}) if prior_route else {}
        if prior_route and not validate_feasibility_fit(prior_fit, state.get("topic")):
            state["feasibility_resume_context"] = {
                "input_snapshot": deepcopy(state.get("feasibility_input_snapshot")),
                "operationalization": state.get("feasibility_operationalization") or preserved.get("operationalization"),
                "route": deepcopy(prior_route.get("route") or {"approach": prior_route.get("approach"), "reason": prior_route.get("raw_reason", "")}),
                "scientific_fit": deepcopy(prior_fit), "skip_optional_requirements": True, "skip_alternative_route": True,
                "origin": "bounded_optional_enrichment_exhaustion_repair", "created_at": now_iso(),
            }
            feasibility_node["status"] = "PENDING"; feasibility_node["lease"] = None
            feasibility_node["failure_reason"] = None; feasibility_node["attempts"] = 0
            clear_node_outputs(state, "feasibility_analysis"); repaired.append("feasibility_analysis")
    for decision in state.get("decisions", []):
        if (
            decision.get("status") == "WAITING_FOR_HUMAN"
            and decision.get("blocked_nodes") == ["feasibility_analysis"]
            and "STRUCTURED_OUTPUT_INCOMPLETE" in decision.get("why_human_is_needed", "")
            and any(call.get("semantic_task", "").startswith("feasibility_") for call in state.get("budget", {}).get("calls", [])[-6:])
        ):
            node = state.get("dag", {}).get("nodes", {}).get("feasibility_analysis")
            if node:
                node["status"] = "PENDING"; node["lease"] = None; node["failure_reason"] = None; node["attempts"] = 0
                clear_node_outputs(state, "feasibility_analysis")
                repaired.append("feasibility_analysis")
            decision["status"] = "INVALIDATED_ATOMIC_FAILURE_CLASSIFICATION_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "atomic feasibility exhaustion now has a distinct human-checkpoint policy"
            invalidated.append(decision["decision_id"])
    for decision in state.get("decisions", []):
        if (
            decision.get("status") == "WAITING_FOR_HUMAN"
            and decision.get("blocked_nodes") == ["feasibility_analysis"]
            and "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=feasibility_requirement_generation"
                in decision.get("why_human_is_needed", "")
        ):
            steps = state.get("feasibility_atomic_steps", [])
            operational = next((step.get("structured", {}).get("observable_test") for step in reversed(steps)
                                if step.get("semantic_task") == "feasibility_operationalization"), None)
            route = next((deepcopy(step.get("structured")) for step in reversed(steps)
                          if step.get("semantic_task") == "feasibility_route_generation"), None)
            state["feasibility_resume_context"] = {
                "input_snapshot": deepcopy(state.get("feasibility_input_snapshot")),
                "operationalization": operational, "route": route,
                "origin": "typed_requirement_route_semantics_repair", "created_at": now_iso(),
            }
            state.setdefault("feasibility_history", []).append({
                "decision": deepcopy(decision), "input_snapshot": deepcopy(state.get("feasibility_input_snapshot")),
                "operationalization": operational, "route": route,
                "atomic_steps": deepcopy(steps), "created_at": now_iso(),
            })
            node = state.get("dag", {}).get("nodes", {}).get("feasibility_analysis")
            if node:
                node["status"] = "PENDING"; node["lease"] = None; node["failure_reason"] = None; node["attempts"] = 0
                clear_node_outputs(state, "feasibility_analysis")
                repaired.append("feasibility_analysis")
            decision["status"] = "INVALIDATED_TYPED_REQUIREMENT_ROUTE_SEMANTICS_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "typed requirements and deterministic route-inherent requirements replace lexical resource markers"
            invalidated.append(decision["decision_id"])
    for decision in state.get("decisions", []):
        if (
            decision.get("status") == "WAITING_FOR_HUMAN"
            and decision.get("blocked_nodes") == ["question_refinement"]
            and (
                "SEMANTIC_VALIDATION_FAILURE" in decision.get("why_human_is_needed", "")
                or "SCIENTIFIC_JUDGMENT_REQUIRED" in decision.get("why_human_is_needed", "")
            )
        ):
            node = state.get("dag", {}).get("nodes", {}).get("question_refinement")
            if node:
                node["status"] = "PENDING"
                node["lease"] = None
                node["failure_reason"] = None
                node["attempts"] = 0
                clear_node_outputs(state, "question_refinement")
                repaired.append("question_refinement")
            decision["status"] = "INVALIDATED_REFINEMENT_SEMANTICS_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "dimension-specific refinement semantics and novelty challenge flow were repaired"
            invalidated.append(decision["decision_id"])
    current_candidates = deepcopy(state.get("candidate_questions", []))
    current_question = current_candidates[0].get("question", "") if current_candidates and isinstance(current_candidates[0], dict) else ""
    current_validation = candidate_question_semantic_validation(
        current_question, state.get("topic"), state.get("literature_cache", [])
    ) if current_question else None
    unsuitable_checkpoint = any(
        decision.get("status") == "WAITING_FOR_HUMAN"
        and "question_refinement" in decision.get("blocked_nodes", [])
        and (
            "QUESTION_REFINEMENT_ATOMIC_SCORES_BELOW_THRESHOLD" in decision.get("why_human_is_needed", "")
            or "CANDIDATE_QUESTION_UNSUITABLE" in decision.get("why_human_is_needed", "")
            or "SCIENTIFIC_JUDGMENT_REQUIRED atomic scores below threshold" in decision.get("why_human_is_needed", "")
        )
        for decision in state.get("decisions", [])
    )
    recent_refinement_reasons = " ".join(
        _text(step.get("reason")).lower()
        for step in state.get("guided_refinement_steps", [])[-3:]
    )
    candidate_issue_markers = (
        "not empirically testable", "planning question", "meta question", "too vague",
        "not locally investigable", "not feasible to investigate locally", "not directly observable",
        "not directly measurable", "disconnected from", "not testable",
    )
    candidate_issue = any(marker in recent_refinement_reasons for marker in candidate_issue_markers)
    if unsuitable_checkpoint and current_validation and (not current_validation["substantive_question"] or candidate_issue):
        state.setdefault("candidate_question_history", []).append({
            "candidate_questions": current_candidates,
            "validation": current_validation,
            "trigger": "repair_stale_candidate_unsuitability_checkpoint",
            "refinement_reasons": recent_refinement_reasons,
            "created_at": now_iso(),
        })
        state["candidate_questions"] = []
        state["candidate_question_validation"] = current_validation
        clear_node_outputs(state, "question_refinement")
        for node_id in ("question_discovery", "question_refinement"):
            node = state.get("dag", {}).get("nodes", {}).get(node_id)
            if node:
                node["status"] = "PENDING"
                node["lease"] = None
                node["failure_reason"] = None
                node["attempts"] = 0
                repaired.append(node_id)
        for decision in state.get("decisions", []):
            if decision.get("status") == "WAITING_FOR_HUMAN" and "question_refinement" in decision.get("blocked_nodes", []):
                decision["status"] = "INVALIDATED_CANDIDATE_UNSUITABILITY_REPAIR"
                decision["invalidated_at"] = now_iso()
                decision["invalidation_reason"] = "candidate was a meta-research planning prompt; regenerate from preserved literature"
                invalidated.append(decision["decision_id"])
    for decision in state.get("decisions", []):
        if decision.get("status") != "WAITING_FOR_HUMAN":
            continue
        why = (decision.get("why_human_is_needed") or "").lower()
        blocked_nodes = decision.get("blocked_nodes", [])
        runtime_environment_retryable = any(retryable_local_runtime_environment_failure(state, node_id) for node_id in blocked_nodes)
        if (
            "model_output_invalid" not in why
            and "schema_validation_failure" not in why
            and "schema validation" not in why
            and not runtime_environment_retryable
        ):
            continue
        for node_id in blocked_nodes:
            node = state.get("dag", {}).get("nodes", {}).get(node_id)
            if node and node.get("status") == "WAITING_FOR_HUMAN":
                node["status"] = "PENDING"
                node["lease"] = None
                node["failure_reason"] = None
                node["attempts"] = 0
                repaired.append(node_id)
        decision["status"] = "INVALIDATED_STRUCTURED_DECODING_FIX"
        decision["invalidated_at"] = now_iso()
        decision["invalidation_reason"] = "structured JSON decoding integration was repaired after this decision was created"
        invalidated.append(decision["decision_id"])
    for node_id, node in state.get("dag", {}).get("nodes", {}).items():
        reason = node.get("failure_reason") or ""
        if node.get("status") != "FAILED":
            continue
        if (
            "MODEL_OUTPUT_INVALID" not in reason
            and "schema validation" not in reason
            and "malformed structured LLM output" not in reason
            and not (node_id == "question_refinement" and "MAX_AGENT_ITERATIONS" in reason)
        ):
            continue
        node["status"] = "PENDING"
        node["lease"] = None
        node["failure_reason"] = None
        node["attempts"] = 0
        repaired.append(node_id)
    for node_id, errors in revalidate_planning_state(state, stages=("question_refinement",)):
        node = state.get("dag", {}).get("nodes", {}).get(node_id)
        if node:
            node["status"] = "PENDING"
            node["lease"] = None
            node["failure_reason"] = "RETRY_AFTER_SEMANTIC_REVALIDATION: " + "; ".join(errors[:5])
            node["attempts"] = 0
            clear_node_outputs(state, node_id)
            repaired.append(node_id)
            dependency_invalidations.extend(mark_dependent_nodes_pending(state, node_id))
        state.setdefault("planning_revalidation", []).append({
            "node_id": node_id,
            "errors": errors,
            "created_at": now_iso(),
        })
    if repaired and state.get("status") in {"FAILED", "WAITING_FOR_HUMAN"}:
        state["status"] = "PLANNED_RESEARCH"
    budget = state.setdefault("budget", {"llm_usd": 0.0, "strong_calls": 0, "calls": []})
    budget["llm_usd"] = round(sum(float(call.get("actual_cost", 0.0) or 0.0) for call in budget.get("calls", [])), 8)
    budget["estimated_llm_usd"] = round(sum(float(call.get("estimated_cost", 0.0) or 0.0) for call in budget.get("calls", [])), 8)
    if repaired:
        state.setdefault("repair_history", []).append({
            "event": "REPAIRED_RECOVERABLE_STRUCTURED_GENERATION_FAILURES",
            "node_ids": repaired,
            "decision_ids": invalidated,
            "dependent_node_ids": dependency_invalidations,
            "guided_candidate_handoff": handoff_repair,
            "created_at": now_iso(),
        })
        state["updated_at"] = now_iso()
    return repaired


def retryable_local_runtime_environment_failure(state, node_id):
    for call in reversed(state.get("budget", {}).get("calls", [])):
        if call.get("stage") != node_id:
            continue
        if call.get("failure_type") != "MODEL_EXECUTION_FAILED":
            continue
        text = " ".join(str(item) for item in call.get("schema_errors", []))
        text += " " + str(call.get("stderr") or "")
        text += " " + str(call.get("raw_response") or "")
        if "failed to get a free port" in text.lower():
            return True
    return False


def repair_guided_candidate_question_handoff(state):
    candidates = state.get("candidate_questions") or []
    if not candidates or not isinstance(candidates[0], dict):
        return None
    current_question = candidates[0].get("question")
    fallback_question = f"What evidence in retrieved literature can bound or test {state.get('topic')}?"
    if current_question != fallback_question:
        return None
    records = state.get("literature_cache", [])
    for call in reversed(state.get("budget", {}).get("calls", [])):
        if call.get("stage") != "question_discovery" or call.get("status") != "SUCCESS":
            continue
        parsed = call.get("parsed_response")
        if not isinstance(parsed, dict) or not parsed.get("question"):
            continue
        question = normalize_atomic_question(parsed.get("question"))
        errors = validate_single_question(question, state.get("topic"), records)
        if errors:
            continue
        candidates[0]["question"] = question
        provenance = {
            "question": {
                "origin": "local_model",
                "source": "repaired_from_preserved_atomic_question",
                "selected_configuration_id": call.get("selected_configuration_id"),
                "actual_model": call.get("actual_model"),
                "raw_response": call.get("raw_response"),
            },
            "why_interesting": {
                "origin": "deterministic_supervisor",
                "source": "guided_question_discovery_metadata",
            },
            "falsifiability": {
                "origin": "deterministic_supervisor",
                "source": "guided_question_discovery_metadata",
            },
            "local_executability": {
                "origin": "deterministic_supervisor",
                "source": "guided_question_discovery_metadata",
            },
        }
        state["candidate_question_field_provenance"] = [provenance]
        state.setdefault("guided_agent_steps", []).append({
            "node_id": "question_discovery",
            "microstep": "candidate_question_handoff_repair",
            "question": question,
            "origin": "local_model",
            "created_at": now_iso(),
        })
        reset_nodes = []
        for node_id in ("question_refinement",):
            node = state.get("dag", {}).get("nodes", {}).get(node_id)
            if node:
                node["status"] = "PENDING"
                node["lease"] = None
                node["failure_reason"] = None
                node["attempts"] = 0
                reset_nodes.append(node_id)
        return {
            "question": question,
            "previous_question": current_question,
            "reset_nodes": reset_nodes,
        }
    return None


def repair_invalid_evidence_relevance(state):
    nodes = state.get("dag", {}).get("nodes", {})
    evidence_node = nodes.get("evidence_discovery")
    if not evidence_node:
        return {"repaired": False, "reason": "missing evidence_discovery node"}
    question_node = nodes.get("question_discovery")
    stale_attempt_failure = (
        question_node
        and question_node.get("status") == "FAILED"
        and question_node.get("failure_reason") == "MAX_AGENT_ITERATIONS"
        and not state.get("candidate_questions")
        and not state.get("literature_cache")
        and state.get("invalidated_evidence_history")
    )
    guided_retry_waiting = (
        question_node
        and question_node.get("status") == "WAITING_FOR_HUMAN"
        and not state.get("candidate_questions")
        and not state.get("literature_cache")
        and any(
            decision.get("status") == "WAITING_FOR_HUMAN"
            and "question_discovery" in decision.get("blocked_nodes", [])
            for decision in state.get("decisions", [])
        )
    )
    queries = state.get("search_strategy", {}).get("queries") or []
    candidates = state.get("candidate_questions", [])
    invalid_queries = []
    for query in queries:
        query_text = query.get("query") if isinstance(query, dict) else str(query)
        if placeholder_like(query_text) or topic_overlap_score(query_text, [state.get("topic", "")] + [c.get("question", "") for c in candidates if isinstance(c, dict)]) < 0.2:
            invalid_queries.append(query_text)
    records = state.get("literature_cache", [])
    relevance = literature_relevance_report(records, state.get("topic", ""), candidates) if records else state.get("literature_relevance", {})
    relevance_invalid = bool(records) and not relevance.get("usable")
    candidate_warnings = candidate_question_context(state)[2]
    if not invalid_queries and not relevance_invalid and not candidate_warnings and not stale_attempt_failure and not guided_retry_waiting:
        return {"repaired": False, "reason": "current evidence/search state passed relevance checks"}
    preserved = {
        "search_strategy": deepcopy(state.get("search_strategy", {})),
        "literature_cache": deepcopy(state.get("literature_cache", [])),
        "literature_retrievals": deepcopy(state.get("literature_retrievals", [])),
        "literature_relevance": deepcopy(relevance),
        "invalid_queries": invalid_queries,
        "stale_attempt_failure": bool(stale_attempt_failure),
        "guided_retry_waiting": bool(guided_retry_waiting),
        "created_at": now_iso(),
    }
    state.setdefault("invalidated_evidence_history", []).append(preserved)
    state["search_strategy"] = {}
    if candidate_warnings:
        state["candidate_questions"] = []
    state["literature_cache"] = []
    state["literature_retrievals"] = []
    state["literature_relevance"] = literature_relevance_report([], state.get("topic", ""), candidates)
    state.setdefault("evidence_discovery_attempts", []).append({
        "source": "repair_invalid_evidence_relevance",
        "queries": queries,
        "record_count": len(records),
        "relevance": relevance,
        "retry_reason": "placeholder_search_strategy_or_irrelevant_retrieval",
        "created_at": now_iso(),
    })
    invalidated_nodes = []
    earliest_nodes = ("question_discovery", "evidence_discovery", "question_refinement") if candidate_warnings or stale_attempt_failure or guided_retry_waiting else ("evidence_discovery", "question_refinement")
    for node_id in earliest_nodes:
        node = nodes.get(node_id)
        if node:
            node["status"] = "PENDING"
            node["lease"] = None
            node["failure_reason"] = None
            node["attempts"] = 0
            node["artifacts"] = []
            invalidated_nodes.append(node_id)
            clear_node_outputs(state, node_id)
    invalidated_nodes.extend(mark_dependent_nodes_pending(state, "evidence_discovery"))
    for decision in state.get("decisions", []):
        if decision.get("status") == "WAITING_FOR_HUMAN" and any(node in invalidated_nodes for node in decision.get("blocked_nodes", [])):
            decision["status"] = "INVALIDATED_UPSTREAM_EVIDENCE_REPAIR"
            decision["invalidated_at"] = now_iso()
            decision["invalidation_reason"] = "upstream search/retrieval was placeholder or topically irrelevant"
    if not any(d.get("status") == "WAITING_FOR_HUMAN" for d in state.get("decisions", [])):
        state["status"] = "PLANNED_RESEARCH"
    state.setdefault("repair_history", []).append({
        "event": "REPAIRED_INVALID_EVIDENCE_RELEVANCE",
        "invalidated_nodes": sorted(set(invalidated_nodes)),
        "invalid_queries": invalid_queries,
        "candidate_warnings": candidate_warnings,
        "relevance": relevance,
        "created_at": now_iso(),
    })
    state["updated_at"] = now_iso()
    return {
        "repaired": True,
        "invalidated_nodes": sorted(set(invalidated_nodes)),
        "invalid_queries": invalid_queries,
        "preserved_record_count": len(records),
        "relevance": relevance,
    }


def mark_dependent_nodes_pending(state, node_id):
    nodes = state.get("dag", {}).get("nodes", {})
    changed = []
    frontier = [node_id]
    seen = set()
    while frontier:
        current = frontier.pop()
        for candidate_id, candidate in nodes.items():
            if candidate_id in seen:
                continue
            if current in candidate.get("dependencies", []):
                seen.add(candidate_id)
                frontier.append(candidate_id)
                if candidate.get("status") not in {"PENDING", "LEASED"}:
                    candidate["status"] = "PENDING"
                    candidate["lease"] = None
                    candidate["failure_reason"] = None
                    changed.append(candidate_id)
                elif candidate.get("status") == "WAITING_FOR_HUMAN":
                    candidate["status"] = "PENDING"
                    candidate["lease"] = None
                    candidate["failure_reason"] = None
                    changed.append(candidate_id)
    if changed and state.get("status") == "WAITING_FOR_HUMAN":
        for decision in state.get("decisions", []):
            if decision.get("status") == "WAITING_FOR_HUMAN" and any(n in changed for n in decision.get("blocked_nodes", [])):
                decision["status"] = "INVALIDATED_DEPENDENCY_REVALIDATION"
                decision["invalidated_at"] = now_iso()
                decision["invalidation_reason"] = f"upstream node {node_id} was invalidated by semantic revalidation"
        if not any(d.get("status") == "WAITING_FOR_HUMAN" for d in state.get("decisions", [])):
            state["status"] = "PLANNED_RESEARCH"
    return changed


class GenericResearchRuntime:
    def __init__(self, store, work_root=None, literature_provider=None, gateway=None):
        self.store = store
        self.work_root = Path(work_root or os.environ.get("RESEARCH_WORK_ROOT", "/tmp/researchGPT-worker"))
        self.literature_provider = literature_provider or literature_provider_from_env()
        self.gateway = gateway or ModelGateway()

    def execute(self, state, node):
        handler = getattr(self, f"_node_{node['node_id']}", None)
        if not handler:
            return False
        validation_error = validate_node_llm_task_class(node)
        if validation_error:
            block_node(state, node["node_id"], "BLOCKED_ENGINEERING_REQUIRED", validation_error)
            return True
        try:
            handler(state, node)
        except MissingLLMProvider as exc:
            block_node(state, node["node_id"], "BLOCKED_MISSING_LLM", str(exc))
        except MalformedStructuredOutput as exc:
            block_node(state, node["node_id"], "FAILED", f"malformed structured LLM output: {exc}")
        except UnknownLLMTaskClass as exc:
            block_node(state, node["node_id"], "BLOCKED_ENGINEERING_REQUIRED", str(exc))
        except StructuredDecodingConfigurationFailure as exc:
            clear_node_outputs(state, node["node_id"])
            block_node(state, node["node_id"], "BLOCKED_ENGINEERING_REQUIRED", str(exc))
        except LocalRuntimeInfrastructureFailure as exc:
            clear_node_outputs(state, node["node_id"])
            active = state.get("active_external_continuation")
            if active:
                active["status"] = "CONTINUATION_BLOCKED_ENGINEERING"
                active["continuation_validation_result"] = "NOT_COMPLETED"
                active["failure_reason"] = str(exc)
                active["updated_at"] = now_iso()
                active.setdefault("events", []).append({"event": "CONTINUATION_EXCEPTION", "timestamp": now_iso(),
                    "failure_class": "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE", "reason": str(exc)})
                # No assistant generation occurred; the scientific validation operation
                # remains unconsumed and can be retried after runtime recovery.
                allowance = active.get("downstream_semantic_allowance") or {}
                allowance["used_semantic_tasks"] = [task for task in allowance.get("used_semantic_tasks", [])
                                                     if task != "measurement_informativeness"]
                add_node_failure_record(state, node["node_id"], "LOCAL_RUNTIME_RESOURCE_UNAVAILABLE",
                    str(exc), active.get("continuation_type"), [active.get("response_artifact")])
                attempt_id = (state.get("computational_measurement_recovery") or {}).get("continuation_attempt_id")
                attempt = next((item for item in active.get("attempts", []) if item.get("attempt_id") == attempt_id), None)
                if attempt:
                    attempt["status"] = "RETRY_BLOCKED_ENGINEERING"; attempt["completed_at"] = now_iso()
                    attempt["failure_reason"] = str(exc)
            create_engineering_request(
                state, "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE", str(exc),
                relevant_files=["src/local_inference.py", "src/llm_gateway.py"],
                required_behavior="Restore local runtime resource availability and retry the unchanged semantic task.",
                acceptance_tests=["local inference invocation starts and produces assistant generation"])
            block_node(state, node["node_id"], "BLOCKED_ENGINEERING_REQUIRED", str(exc))
            state.pop("active_external_continuation", None)
        except StructuredGenerationExhausted as exc:
            clear_node_outputs(state, node["node_id"])
            self._external_reasoning_required(state, node, str(exc))
        except BudgetExceeded as exc:
            active = state.get("active_external_continuation")
            if active:
                active["status"] = "CONTINUATION_BLOCKED_ENGINEERING"
                active["continuation_validation_result"] = "NOT_COMPLETED"
                active["failure_reason"] = str(exc)
                active["updated_at"] = now_iso()
                active.setdefault("events", []).append({"event": "CONTINUATION_EXCEPTION", "timestamp": now_iso(),
                    "failure_class": "CONTINUATION_RUNTIME_OR_BUDGET_FAILURE", "reason": str(exc)})
                add_node_failure_record(state, node["node_id"], "CONTINUATION_RUNTIME_OR_BUDGET_FAILURE",
                    str(exc), active.get("continuation_type"), [active.get("response_artifact")])
                create_engineering_request(state, "EXTERNAL_DECISION_CONTINUATION_UNAVAILABLE", str(exc),
                    relevant_files=["src/research_runtime.py", "src/llm_gateway.py"],
                    required_behavior="Resume the persisted accepted response without consuming its local generation budget.")
                block_node(state, node["node_id"], "BLOCKED_ENGINEERING_REQUIRED", str(exc))
            else:
                from src.budget_control import agent_iteration_usage, authorized_limit, record_budget_block
                initial_limit = self.gateway.budget_manager.max_agent_iterations
                record_budget_block(
                    state, node["node_id"], "agent_iterations",
                    authorized_limit(state, "agent_iterations", initial_limit),
                    agent_iteration_usage(state), node.get("llm_task_class") or node["node_id"], str(exc))
                block_node(state, node["node_id"], "BLOCKED_BUDGET", str(exc))
        except NoEligibleLocalModel as exc:
            self._external_reasoning_required(state, node, str(exc))
        except Exception as exc:
            message = str(exc)
            active = state.get("active_external_continuation")
            if active:
                active["status"] = "CONTINUATION_BLOCKED_ENGINEERING"
                active["continuation_validation_result"] = "NOT_COMPLETED"
                active["failure_reason"] = message
                active["updated_at"] = now_iso()
                active.setdefault("events", []).append({"event": "CONTINUATION_EXCEPTION", "timestamp": now_iso(),
                    "failure_class": "CONTINUATION_IMPLEMENTATION_FAILURE", "reason": message})
                add_node_failure_record(state, node["node_id"], "CONTINUATION_IMPLEMENTATION_FAILURE",
                    message, active.get("continuation_type"), [active.get("response_artifact")])
                attempt_id = (state.get("computational_measurement_recovery") or {}).get("continuation_attempt_id")
                attempt = next((item for item in active.get("attempts", []) if item.get("attempt_id") == attempt_id), None)
                if attempt:
                    attempt["status"] = "RETRY_BLOCKED_ENGINEERING"
                    attempt["completed_at"] = now_iso()
                    attempt["failure_reason"] = message
                create_engineering_request(state, "EXTERNAL_DECISION_CONTINUATION_FAILURE", message,
                    relevant_files=["src/research_runtime.py"],
                    required_behavior="Idempotently resume the persisted external semantic response checkpoint.")
                block_node(state, node["node_id"], "BLOCKED_ENGINEERING_REQUIRED", message)
                return True
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                block_node(state, node["node_id"], "BLOCKED_BUDGET", message)
                return True
            if "no eligible local model configuration" in message or "NO_ELIGIBLE_LOCAL_MODEL" in message:
                self._external_reasoning_required(state, node, message)
                return True
            block_node(state, node["node_id"], "FAILED", str(exc))
        return True

    def _external_reasoning_required(self, state, node, reason, response_contract=None):
        bundle_rel = write_external_reasoning_bundle(self.store, state, node, reason, response_contract)
        attempts = llm_attempts_for_stage(state, node["node_id"])
        decision_policy = external_reasoning_options(attempts, reason)
        decision = DecisionEngine().resolve_or_request(state, {
            "stage": "EXTERNAL_REASONING_REQUIRED",
            "severity": "HIGH",
            "question": f"Node '{node['node_id']}': {decision_policy['question']}",
            "why_human_is_needed": reason,
            "options": decision_policy["options"],
            "recommended_option": decision_policy["recommended_option"],
            "recommendation_confidence": decision_policy["recommendation_confidence"],
            "blocked_nodes": [node["node_id"]],
            "evidence": [bundle_rel],
            "external_reasoning_bundle": bundle_rel,
            "response_kind": (response_contract or {}).get("response_kind", "NODE_LEVEL_RESPONSE"),
            "response_contract": deepcopy(response_contract) if response_contract else None,
            "continuation": deepcopy((response_contract or {}).get("continuation")),
            "risk": "high",
            "material_scientific_impact": True,
        })
        node["status"] = "WAITING_FOR_HUMAN"
        node["lease"] = None
        node["failure_reason"] = f"EXTERNAL_REASONING_REQUIRED: {decision['decision_id']}"
        state["status"] = "WAITING_FOR_HUMAN"

    def _node_question_discovery(self, state, node):
        if local_guidance_high():
            return self._guided_question_discovery(state, node)
        prompt = {
            "task": "Generate generic research question candidates and iterative scholarly search queries. Return JSON only.",
            "topic": state["topic"],
            "required_fields": {
                "candidate_questions": ["question", "why_interesting", "falsifiability", "local_executability"],
                "search_queries": "list of substantive literature queries",
            },
            "constraints": [
                "Do not claim novelty.",
                "Do not invent citations.",
                "Keep questions small enough for local executable evidence or systematic literature analysis.",
            ],
        }
        response = self.gateway.generate_structured(
            state,
            LLMRequest(json.dumps(prompt), stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            required_keys=["candidate_questions", "search_queries"],
            schema=QUESTION_DISCOVERY_SCHEMA,
            semantic_validator=lambda data: validate_question_discovery_semantics(data, state.get("topic")),
            estimated_cost=0.002,
        )
        data = response["structured"]
        state["candidate_questions"] = data["candidate_questions"]
        state["search_strategy"] = {"queries": data["search_queries"], "created_at": now_iso()}
        self._persist_node_artifact(state, node, "research_spec.json", {
            "topic": state["topic"],
            "candidate_questions": data["candidate_questions"],
            "search_strategy": state["search_strategy"],
        })

    def _guided_question_discovery(self, state, node):
        controller = GuidedToolController(self.store, state["run_id"], self.work_root, self.literature_provider)
        observations = []
        tool_calls = []
        existing_records = dedupe_records(state.get("literature_cache", []))
        existing_relevance = state.get("literature_relevance", {})
        reuse_evidence = bool(existing_records and existing_relevance.get("usable"))
        existing_search = state.get("search_strategy", {})
        existing_queries = existing_search.get("queries") or []
        query = existing_search.get("raw_query") or (existing_queries[0] if existing_queries else None)
        query_origin = existing_search.get("query_origin", "preserved_search_provenance")
        if not reuse_evidence:
            query_prompt = self._atomic_prompt(
                objective="Write one scholarly search query for TOPIC.",
                context={"topic": state["topic"]},
                expected="Return JSON with field: query.",
            )
            self._record_prompt_telemetry(state, node, 1, query_prompt, "search_query")
            query = None
            query_origin = "model"
            try:
                response = self.gateway.generate_structured(
                    state,
                    LLMRequest(query_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                    schema=SEARCH_QUERY_SCHEMA,
                    estimated_cost=0.001,
                )
                query = response["structured"].get("query")
            except Exception as exc:
                state.setdefault("guided_agent_steps", []).append({"node_id": node["node_id"], "step": 1, "microstep": "search_query", "status": "MODEL_QUERY_FAILED", "error": str(exc), "created_at": now_iso()})
        query_errors = validate_search_query(query, state["topic"])
        if query_errors:
            fallback = fallback_search_queries(state["topic"])
            if not fallback:
                raise StructuredGenerationExhausted("MODEL_OUTPUT_INVALID guided search query failed and deterministic fallback unavailable", [])
            query = fallback[0]
            query_origin = "deterministic_fallback"
            query_errors = validate_search_query(query, state["topic"])
            if query_errors:
                raise StructuredGenerationExhausted("SEMANTIC_VALIDATION_FAILURE deterministic fallback search query invalid", [])
        raw_query = query
        normalized_query = normalize_research_query(query)
        state.setdefault("guided_agent_steps", []).append({
            "node_id": node["node_id"],
            "step": 1,
            "microstep": "search_query",
            "query": normalized_query,
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "origin": query_origin,
            "validation_errors": query_errors,
            "created_at": now_iso(),
        })

        if reuse_evidence:
            records = existing_records
            relevance = existing_relevance
            observations.append(f"reused {len(records)} active literature records")
            tool_artifact = None
        else:
            search_action = {"tool": "literature_search", "arguments": {"query": normalized_query, "limit": int(os.environ.get("RESEARCH_LITERATURE_LIMIT", "5"))}}
            descriptor = provider_descriptor(getattr(self.literature_provider, "provider_name", "scholarly_provider"),
                                             "scholarly_source", ["scholarly_search"], "scholarly_metadata")
            try:
                search_result = controller.execute(search_action)
                tool_calls.append(search_result)
                observations.append(search_result["observation"])
                state.setdefault("guided_literature_results", []).append(search_result["result"])
                records = dedupe_records(search_result["result"].get("records", []))
                relevance = literature_relevance_report(records, state["topic"], [])
                provider_outcome = classify_provider_search_outcome(descriptor, search_result["result"], relevance=relevance)
                tool_artifact = search_result["artifact"]
            except Exception as exc:
                records = []
                relevance = literature_relevance_report([], state["topic"], [])
                provider_outcome = classify_provider_search_outcome(descriptor, error=exc)
                tool_artifact = None
                observations.append(f"scholarly retrieval failed: {type(exc).__name__}")
            state.setdefault("research_source_attempts", []).append({
                **provider_outcome, "operation": "scholarly_search", "raw_query": raw_query,
                "normalized_query": normalized_query, "tool_artifact": tool_artifact,
            })
        state.setdefault("guided_agent_steps", []).append({
            "node_id": node["node_id"],
            "step": 2,
            "microstep": "literature_search",
            "query": normalized_query,
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "tool_artifact": tool_artifact,
            "record_count": len(records),
            "relevance": relevance,
            "created_at": now_iso(),
        })

        state["prior_work_coverage"] = classify_prior_work_coverage(
            state.get("research_source_attempts", []), relevance.get("relevant_count", 0), bool(records and relevance.get("usable")))
        if records and relevance["usable"]:
            question, response, validation = self._generate_guided_candidate_question(state, node, records, step=3)
            question_context = "retrieved_literature"
        else:
            provisional = ["executable_computation"]
            allowed, closure = may_continue_without_literature(
                provisional, self._current_capability_semantics(state), state.get("question_selection_policy"),
                state.get("verified_evidence_artifacts", []))
            if not allowed:
                self._external_reasoning_required(state, node, "LITERATURE_RETRIEVAL_UNSUCCESSFUL bounded retrieval did not obtain usable records; prior-work coverage remains unknown")
                return
            question, response, validation = self._generate_topic_computational_question(state, node, step=3)
            question_context = "topic_and_verified_modality"
        question_origin = "local_model"
        question_errors = validation["rejection_reasons"]
        state.setdefault("guided_agent_steps", []).append({
            "node_id": node["node_id"],
            "step": 3,
            "microstep": "candidate_question",
            "question": question,
            "origin": question_origin,
            "validation_errors": question_errors,
            "candidate_question_validation": validation,
            "created_at": now_iso(),
        })
        if question_errors:
            self._external_reasoning_required(state, node, "MODEL_OUTPUT_INVALID question generation exhausted without a substantive candidate question")
            return
        scope_contract = question_scope_modalities(question)
        modality_assessment = assess_automation_closure(scope_contract["required_evidence_modalities"],
                                                        self._current_capability_semantics(state),
                                                        state.get("verified_evidence_artifacts", []))
        candidate = {
            "question": question,
            "why_interesting": "A substantive candidate suitable for later feasibility and evidence checks.",
            "falsifiability": "The retrieved evidence may fail to support a measurable or testable relationship.",
            "local_executability": "Initial retrieval and metadata checks can run with local/free tools.",
        }
        candidate_field_provenance = {
            "question": {
                "origin": question_origin,
                "node_id": node["node_id"],
                "microstep": "candidate_question",
            },
            "why_interesting": {
                "origin": "deterministic_supervisor",
                "node_id": node["node_id"],
                "source": "guided_question_discovery_metadata",
            },
            "falsifiability": {
                "origin": "deterministic_supervisor",
                "node_id": node["node_id"],
                "source": "guided_question_discovery_metadata",
            },
            "local_executability": {
                "origin": "deterministic_supervisor",
                "node_id": node["node_id"],
                "source": "guided_question_discovery_metadata",
            },
        }
        assembled = {"candidate_questions": [candidate], "search_queries": [normalized_query]}
        schema_errors = validate_json_schema_subset(assembled, QUESTION_DISCOVERY_SCHEMA)
        semantic_errors = [] if schema_errors else validate_question_discovery_semantics(assembled, state.get("topic"))
        if schema_errors or semantic_errors:
            raise StructuredGenerationExhausted("SEMANTIC_VALIDATION_FAILURE guided assembled question discovery invalid", [{
                "stage": node["node_id"],
                "task_class": node["llm_task_class"],
                "status": "FAILED",
                "failure_type": "SEMANTIC_VALIDATION_FAILURE",
                "schema_errors": schema_errors,
                "semantic_errors": semantic_errors,
                "raw_response": json.dumps(assembled),
                "actual_cost": 0.0,
            }])
        state["candidate_questions"] = assembled["candidate_questions"]
        state.setdefault("candidate_evidence_contracts", {})[question] = {
            **scope_contract, **modality_assessment, "prior_work_coverage": state["prior_work_coverage"]["status"],
            "novelty_status": state["prior_work_coverage"]["novelty_status"], "question_context": question_context,
        }
        state["candidate_question_validation"] = validation
        state["candidate_question_field_provenance"] = [candidate_field_provenance]
        state["search_strategy"] = {
            "queries": assembled["search_queries"], "created_at": now_iso(), "guided": True,
            "query_origin": query_origin, "raw_query": raw_query, "normalized_query": normalized_query,
        }
        self._persist_node_artifact(state, node, "research_spec.json", {
            "topic": state["topic"],
            "candidate_questions": state["candidate_questions"],
            "candidate_question_field_provenance": state["candidate_question_field_provenance"],
            "search_strategy": state["search_strategy"],
            "guided_observations": observations,
            "tool_artifacts": [call["artifact"] for call in tool_calls],
            "search_relevance": relevance,
            "candidate_question_validation": validation,
            "prior_work_coverage": state["prior_work_coverage"],
            "candidate_evidence_contract": state["candidate_evidence_contracts"][question],
        })

    def _generate_topic_computational_question(self, state, node, step=1):
        prompt = self._atomic_prompt(
            objective="Write one substantive computational research question for TOPIC that is falsifiable by a bounded experiment.",
            context={"topic": state["topic"]}, expected='Return exactly {"question":"..."}.',
            constraints=["Use an explicit bounded input or benchmark regime.", "Do not claim novelty.",
                         "Do not mention missing literature.", "Do not make the question about benchmarking methodology.",
                         "Do not answer the question.", "Keep it short."],
        )
        self._record_prompt_telemetry(state, node, step, prompt, "candidate_computational_question")
        def validator(data):
            question = normalize_atomic_question(data.get("question"))
            errors = validate_single_question(question, state.get("topic"), [])
            errors.extend(validate_bounded_computational_question(question))
            return errors
        preserved = state.pop("candidate_generation_resume", None)
        if preserved and not validator({"question": preserved.get("question")}):
            response = {"structured": {"question": preserved["question"]}, "model": preserved.get("actual_model"),
                        "provenance": "preserved_model_output_revalidated_after_semantic_protocol_repair"}
        else:
            response = self.gateway.generate_structured(
                state, LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=ONE_QUESTION_SCHEMA, semantic_validator=validator, estimated_cost=0.001)
        question = normalize_atomic_question(response["structured"].get("question"))
        validation = candidate_question_semantic_validation(question, state.get("topic"), [])
        validation["rejection_reasons"].extend(validate_bounded_computational_question(question))
        validation["substantive_question"] = not validation["rejection_reasons"]
        validation["testability"] = 1.0 if validation["substantive_question"] else 0.0
        return question, response, validation

    def _generate_guided_candidate_question(self, state, node, records, step=1):
        prompt = self._atomic_prompt(
            objective="Write one empirically testable research question grounded in TOPIC and LITERATURE.",
            context={"topic": state["topic"], "literature": compact_literature_cards(records, limit=4)},
            expected='Return exactly {"question":"..."}.',
            constraints=[
                "Ask about an observable relationship, comparison, effect, measurement, mechanism, or difference.",
                "Do not ask what the literature says.",
                "Do not ask whether evidence exists.",
                "Do not describe the research process.",
                "Do not invent sources.",
                "Keep it short.",
            ],
        )
        self._record_prompt_telemetry(state, node, step, prompt, "candidate_question")
        response = self.gateway.generate_structured(
            state,
            LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=ONE_QUESTION_SCHEMA,
            semantic_validator=lambda data: validate_single_question(
                normalize_atomic_question(data.get("question")), state.get("topic"), records
            ),
            estimated_cost=0.001,
        )
        question = normalize_atomic_question(response["structured"].get("question"))
        return question, response, candidate_question_semantic_validation(question, state.get("topic"), records)

    def _tool_cards(self):
        cards = [
            "literature_search(query) Search scholarly metadata.",
            "run_python(code) Run Python in workspace.",
            "run_bash(command) Run safe shell in workspace.",
            "read_artifact(path) Read one run artifact.",
        ]
        if os.environ.get("RESEARCH_WEB_SEARCH_PROVIDER"):
            cards.insert(1, "web_search(query) Search public web pages.")
        if os.environ.get("RESEARCH_WEB_FETCH", "1") == "1":
            cards.insert(1, "fetch_web_source(url) Fetch one public source.")
        return "\n".join(cards)

    def _persist_guided_prompt(self, state, node, step, prompt):
        run_dir = self.work_root / state["run_id"] / node["node_id"] / "guided_prompts"
        run_dir.mkdir(parents=True, exist_ok=True)
        rel = f"guided_prompts/{node['node_id']}-step-{step}-{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}.md"
        path = run_dir / Path(rel).name
        path.write_text(prompt, encoding="utf-8")
        artifact = self.store.put_artifact(state["run_id"], path, rel, "guided_local_agent")
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
        return artifact["path"]

    def _record_prompt_telemetry(self, state, node, step, prompt, microstep):
        state.setdefault("prompt_telemetry", []).append({
            "stage": node["node_id"],
            "microstep": microstep,
            "guided": True,
            "instruction_tokens": token_count("OBJECTIVE CONSTRAINTS CONTEXT EXPECTED"),
            "context_tokens": token_count(prompt.split("CONTEXT", 1)[-1].split("EXPECTED", 1)[0] if "CONTEXT" in prompt else ""),
            "tool_description_tokens": 0,
            "prompt_tokens": token_count(prompt),
            "prompt_artifact": self._persist_guided_prompt(state, node, step, prompt),
            "created_at": now_iso(),
        })

    def _atomic_prompt(self, objective, context, expected, constraints=None):
        constraint_lines = constraints or ["Use concrete topic terms. Keep it short. Do not invent sources."]
        return "\n".join([
            "OBJECTIVE",
            objective,
            "CONSTRAINTS",
            *constraint_lines,
            "CONTEXT",
            json.dumps(context, sort_keys=True),
            "EXPECTED",
            expected,
        ])

    def _guided_prompt(self, objective, context, expected):
        return "\n".join([
            "OBJECTIVE",
            objective,
            "CONSTRAINTS",
            "Use substantive topic terms. Do not invent sources. Keep outputs short.",
            "TOOLS",
            self._tool_cards(),
            "CONTEXT",
            json.dumps(context, sort_keys=True),
            "EXPECTED",
            expected,
        ])

    def _validate_guided_action(self, data, observations, tool_calls_used, max_tool_calls):
        errors = []
        action = data.get("action")
        if action == "tool":
            if tool_calls_used >= max_tool_calls:
                errors.append("tool budget exhausted")
            if not data.get("tool"):
                errors.append("tool action requires tool")
            query = (data.get("arguments") or {}).get("query")
            if data.get("tool") in {"literature_search", "web_search"}:
                if placeholder_like(query):
                    errors.append("tool query is placeholder or content-free")
        elif action == "final":
            if not observations:
                errors.append("final answer requires at least one tool observation")
            if not isinstance(data.get("result"), dict):
                errors.append("final action requires result object")
        return errors

    def _node_evidence_discovery(self, state, node):
        candidates = state.get("candidate_questions", [])
        configured_queries = state.get("search_strategy", {}).get("queries") or []
        valid_queries = []
        invalid_queries = []
        for query in configured_queries:
            if isinstance(query, dict):
                query = query.get("query") or query.get("search_query") or json.dumps(query)
            query = str(query)
            if placeholder_like(query) or topic_overlap_score(query, [state["topic"]] + [c.get("question", "") for c in candidates if isinstance(c, dict)]) < 0.2:
                invalid_queries.append({"query": query, "reason": "placeholder_or_topic_disconnected"})
            else:
                valid_queries.append(query)
        query_attempts = []
        if valid_queries:
            query_attempts.append({"source": "planner", "queries": valid_queries[:5], "invalid_queries": invalid_queries})
        fallback_queries = fallback_search_queries(state["topic"], candidates)
        if fallback_queries and fallback_queries != valid_queries[:5]:
            query_attempts.append({"source": "automatic_rewrite", "queries": fallback_queries, "invalid_queries": invalid_queries})
        max_attempts = int(os.environ.get("RESEARCH_LITERATURE_RECOVERY_ATTEMPTS", "2"))
        all_attempt_reports = []
        for attempt in query_attempts[:max_attempts]:
            all_records = []
            retrievals = []
            for query in attempt["queries"][:5]:
                result = self.literature_provider.search(str(query), limit=int(os.environ.get("RESEARCH_LITERATURE_LIMIT", "8")))
                retrievals.append(result)
                all_records.extend(result.get("records", []))
            records = dedupe_records(all_records)
            relevance = literature_relevance_report(records, state["topic"], candidates)
            attempt_report = {**attempt, "retrievals": retrievals, "records": records, "relevance": relevance}
            all_attempt_reports.append(attempt_report)
            if records and relevance["usable"]:
                state["literature_cache"] = records
                state["literature_retrievals"] = retrievals
                state["literature_relevance"] = relevance
                state.setdefault("evidence_discovery_attempts", []).append({
                    "source": attempt["source"],
                    "queries": attempt["queries"],
                    "record_count": len(records),
                    "relevance": relevance,
                    "created_at": now_iso(),
                })
                self._persist_node_artifact(state, node, "evidence/discovery.json", {"attempts": all_attempt_reports, "retrievals": retrievals, "records": records, "relevance": relevance})
                return
        state.setdefault("evidence_discovery_attempts", []).extend({
            "source": attempt.get("source"),
            "queries": attempt.get("queries", []),
            "record_count": len(attempt.get("records", [])),
            "relevance": attempt.get("relevance", {}),
            "created_at": now_iso(),
        } for attempt in all_attempt_reports)
        state["literature_relevance"] = all_attempt_reports[-1]["relevance"] if all_attempt_reports else literature_relevance_report([], state["topic"], candidates)
        state["prior_work_coverage"] = classify_prior_work_coverage(
            state.get("research_source_attempts", []), state["literature_relevance"].get("relevant_count", 0), False)
        failure_artifact = {"attempts": all_attempt_reports, "invalid_queries": invalid_queries,
                            "prior_work_coverage": state["prior_work_coverage"]}
        contract = next(iter(state.get("candidate_evidence_contracts", {}).values()), {})
        allowed, assessment = may_continue_without_literature(
            contract.get("required_evidence_modalities", []), self._current_capability_semantics(state),
            state.get("question_selection_policy"), state.get("verified_evidence_artifacts", []))
        if allowed:
            state["candidate_modality_assessments"] = {contract.get("question", "candidate"): assessment}
            self._persist_node_artifact(state, node, "evidence/discovery.json", {
                **failure_artifact, "records": [], "retrieval_status": "UNSUCCESSFUL_BOUNDED",
                "execution_evidence_dependency": "NOT_REQUIRED_FOR_EXECUTABLE_COMPUTATION",
            })
            return
        self._persist_node_artifact(state, node, "evidence/discovery_failed.json", failure_artifact)
        self._external_reasoning_required(state, node, "LITERATURE_RETRIEVAL_UNSUCCESSFUL bounded retrieval did not produce usable topical records; execution modality requires literature")

    def _node_question_refinement(self, state, node):
        relevance = state.get("literature_relevance", {})
        contracts = state.get("candidate_evidence_contracts", {})
        first_contract = next(iter(contracts.values()), {})
        literature_optional, _ = may_continue_without_literature(
            first_contract.get("required_evidence_modalities", []), self._current_capability_semantics(state),
            state.get("question_selection_policy"), state.get("verified_evidence_artifacts", []))
        if (not state.get("literature_cache") or not relevance.get("usable")) and not literature_optional:
            block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", "question refinement requires usable relevant literature")
            return
        display_candidates, _, warnings = candidate_question_context(state)
        if warnings or not display_candidates:
            block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", "question refinement requires substantive candidate questions")
            return
        ranked_candidates = []
        for index, candidate in enumerate(display_candidates):
            question = candidate.get("question") if isinstance(candidate, dict) else candidate
            contract = state.get("candidate_evidence_contracts", {}).get(question, {})
            assessment = assess_automation_closure(contract.get("required_evidence_modalities", []),
                                                   self._current_capability_semantics(state), state.get("verified_evidence_artifacts", []))
            ranked_candidates.append({"question_id": str(index), "question": question, "candidate": candidate,
                                      "scientific_score": contract.get("scientific_score", 0.0), **assessment})
        ranked_candidates = rank_candidate_questions(ranked_candidates, state.get("question_selection_policy"))
        display_candidates = [item["candidate"] for item in ranked_candidates]
        state["candidate_modality_assessments"] = {item["question"]: {key: item[key] for key in (
            "required_evidence_modalities", "currently_available_modalities", "missing_modalities", "automation_closure", "assessment_origin")}
            for item in ranked_candidates}
        validation = candidate_question_semantic_validation(
            display_candidates[0].get("question") if isinstance(display_candidates[0], dict) else display_candidates[0],
            state.get("topic"), state.get("literature_cache", []),
        )
        state["candidate_question_validation"] = validation
        if not validation["substantive_question"]:
            if self._regenerate_candidate_from_evidence(state, node, "semantic_validation", validation["rejection_reasons"]):
                display_candidates, _, _ = candidate_question_context(state)
            else:
                self._external_reasoning_required(state, node, "CANDIDATE_QUESTION_UNSUITABLE candidate regeneration exhausted: " + "; ".join(validation["rejection_reasons"]))
                return
        if local_guidance_high():
            return self._guided_question_refinement(state, node, display_candidates)
        prompt = {
            "task": "Select one feasible research question from candidates using only retrieved literature metadata. Return JSON only.",
            "topic": state["topic"],
            "candidate_questions": state.get("candidate_questions", []),
            "literature": minimal_literature_context(state.get("literature_cache", [])),
            "evaluation_dimensions": [
                "novelty_potential",
                "scientific_relevance",
                "falsifiability",
                "available_evidence",
                "local_executability",
                "resource_cost",
                "replicability",
                "scope",
            ],
            "required_fields": {
                "selected_question": "chosen candidate question",
                "candidate_evaluations": ["question", "feasibility", "novelty_potential", "falsifiability", "evidence_accessibility", "rationale"],
                "rationale": "selection justification",
                "limitations": "optional list",
            },
        }
        response = self.gateway.generate_structured(
            state,
            LLMRequest(json.dumps(prompt), stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            required_keys=["selected_question", "candidate_evaluations", "rationale"],
            schema=QUESTION_REFINEMENT_SCHEMA,
            semantic_validator=lambda data: validate_question_refinement_semantics(data, state.get("topic")),
            estimated_cost=0.003,
        )
        data = response["structured"]
        for evaluation in data.get("candidate_evaluations", []):
            question = evaluation.get("question", "")
            contract = state.get("candidate_evidence_contracts", {}).get(question, {})
            assessment = assess_automation_closure(contract.get("required_evidence_modalities", []),
                                                   self._current_capability_semantics(state), state.get("verified_evidence_artifacts", []))
            evaluation.update(assessment)
            state.setdefault("candidate_modality_assessments", {})[question] = deepcopy(assessment)
        state["selected_question"] = data["selected_question"]
        state["candidate_evaluations"] = data["candidate_evaluations"]
        state["question_refinement_rationale"] = data["rationale"]
        state.setdefault("known_limitations", []).extend(data.get("limitations", []))
        self._persist_node_artifact(state, node, "specification.json", data)

    def _guided_question_refinement(self, state, node, display_candidates):
        candidate = display_candidates[0]
        question = candidate["question"]
        records = state.get("literature_cache", [])
        contract = state.get("candidate_evidence_contracts", {}).get(question, {})
        if contract.get("scope_type") == "BOUNDED_COMPUTATIONAL" and contract.get("required_evidence_modalities") == ["executable_computation"]:
            return self._guided_computational_question_refinement(state, node, candidate, contract)
        modality_assessment = assess_automation_closure(contract.get("required_evidence_modalities", []),
                                                        self._current_capability_semantics(state),
                                                        state.get("verified_evidence_artifacts", []))
        relevant_count = int(state.get("literature_relevance", {}).get("relevant_count") or len(records))
        total_count = max(1, len(records))
        evidence_score = (1.0 if modality_assessment["automation_closure"] == "HIGH" else
                          min(1.0, relevant_count / max(1, min(total_count, 5))))
        evidence_reason = ("The required evidence modality is available through a verified local capability; actual claim support still requires execution and validation."
                           if modality_assessment["automation_closure"] == "HIGH" else
                           f"{relevant_count} relevant metadata records are available for planning; full empirical evidence still requires later execution and validation.")
        dimensions = {
            "evidence_accessibility": {
                "score": round(evidence_score, 3),
                "reason": evidence_reason,
                "origin": "deterministic_supervisor",
                "evidence": ["literature_relevance", "literature_cache"],
            }
        }
        model_dimensions = [
            ("falsifiability", "Judge whether this research question can in principle be contradicted or constrained by observable results."),
            ("feasibility", "Assess whether this research question is practical to investigate with available local/free tools, data, or experiments."),
        ]
        for step, (dimension, objective) in enumerate(model_dimensions, 1):
            prompt = self._atomic_refinement_prompt(objective, question, dimension)
            self._record_prompt_telemetry(state, node, step, prompt, f"refinement_{dimension}")
            response = self.gateway.generate_structured(
                state,
                LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=SCORE_REASON_SCHEMA,
                semantic_validator=lambda data, current=dimension: validate_dimension_score(data, current, state.get("topic")),
                estimated_cost=0.001,
            )
            data = response["structured"]
            dimensions[dimension] = {
                "score": float(data["score"]),
                "reason": data["reason"],
                "origin": "local_model",
                "model": response.get("model"),
            }
            state.setdefault("guided_refinement_steps", []).append({
                "node_id": node["node_id"],
                "microstep": dimension,
                "score": dimensions[dimension]["score"],
                "reason": dimensions[dimension]["reason"],
                "origin": dimensions[dimension]["origin"],
                "created_at": now_iso(),
            })

        prior = state.get("prior_work_coverage", {})
        novelty_required = prior.get("status") in {"SUFFICIENT", "PARTIAL"} or (not prior and bool(records))
        novelty = (self._assess_novelty_with_challenge(state, node, question) if novelty_required else {
            "score": 0.0, "assessment": "uncertain", "status": "NOT_ESTABLISHED", "confidence": 0.0,
            "reason": "Prior-work coverage is unavailable or unknown; novelty is not established.",
            "origin": "deterministic_prior_work_gate", "cards": [],
        })
        if novelty["score"] is None:
            self._external_reasoning_required(state, node, "SCIENTIFIC_JUDGMENT_REQUIRED novelty challenge remains uncertain after bounded Q3/Q4 assessment")
            return
        dimensions["novelty_potential"] = novelty
        state.setdefault("guided_refinement_steps", []).append({
            "node_id": node["node_id"], "microstep": "novelty_potential",
            "score": novelty["score"], "reason": novelty["reason"],
            "assessment": novelty["assessment"], "confidence": novelty["confidence"],
            "origin": novelty["origin"], "created_at": now_iso(),
        })
        if novelty["assessment"] == "well_covered":
            if self._regenerate_candidate_from_gap(state, node, question, novelty.get("cards", [])):
                return self._guided_question_refinement(state, node, state["candidate_questions"])
            self._external_reasoning_required(state, node, "SCIENTIFIC_JUDGMENT_REQUIRED candidate remains well covered after bounded gap-grounded regeneration")
            return

        thresholds = {
            "evidence_accessibility": 0.2,
            "falsifiability": 0.2,
            "feasibility": 0.2,
        }
        if novelty_required:
            thresholds["novelty_potential"] = 0.1
        failed = [key for key, minimum in thresholds.items() if dimensions[key]["score"] < minimum]
        if failed:
            reasons = " ".join(dimensions[key]["reason"] for key in failed).lower()
            unsuitable_markers = (
                "not empirically testable", "planning question", "meta question", "meta-question",
                "too vague", "not locally investigable", "not feasible to investigate locally",
                "not directly observable", "not directly measurable", "disconnected from", "not testable",
            )
            if any(marker in reasons for marker in unsuitable_markers) and self._regenerate_candidate_from_evidence(
                state, node, "atomic_refinement_scores", [dimensions[key]["reason"] for key in failed]
            ):
                return self._guided_question_refinement(state, node, state["candidate_questions"])
            classification = "CANDIDATE_QUESTION_UNSUITABLE" if any(marker in reasons for marker in unsuitable_markers) else "SCIENTIFIC_JUDGMENT_REQUIRED"
            self._external_reasoning_required(state, node, classification + " atomic scores below threshold: " + ", ".join(failed))
            return
        evaluation = {
            "question": question,
            "feasibility": dimensions["feasibility"]["score"],
            "novelty_potential": dimensions["novelty_potential"]["score"],
            "falsifiability": dimensions["falsifiability"]["score"],
            "evidence_accessibility": dimensions["evidence_accessibility"]["score"],
            "rationale": "; ".join([
                f"evidence_accessibility: {dimensions['evidence_accessibility']['reason']}",
                f"falsifiability: {dimensions['falsifiability']['reason']}",
                f"feasibility: {dimensions['feasibility']['reason']}",
                f"novelty_potential: {dimensions['novelty_potential']['reason']}",
            ]),
        }
        evaluation["novelty_status"] = novelty.get("status", "ASSESSED")
        modality_contract = state.get("candidate_evidence_contracts", {}).get(question, {})
        modality_assessment = assess_automation_closure(
            modality_contract.get("required_evidence_modalities", []), self._current_capability_semantics(state),
            state.get("verified_evidence_artifacts", []),
        )
        evaluation.update(modality_assessment)
        state["candidate_modality_assessments"] = {question: deepcopy(modality_assessment)}
        data = {
            "selected_question": question,
            "candidate_evaluations": [evaluation],
            "rationale": "Single candidate retained after atomic planning checks met configured evidence-accessibility, falsifiability, feasibility, and novelty-potential thresholds.",
            "limitations": self._question_refinement_limitations(state),
        }
        schema_errors = validate_json_schema_subset(data, QUESTION_REFINEMENT_SCHEMA)
        semantic_errors = [] if schema_errors else validate_question_refinement_semantics(data, state.get("topic"))
        if schema_errors or semantic_errors:
            raise StructuredGenerationExhausted("SEMANTIC_VALIDATION_FAILURE guided question refinement assembly invalid", [{
                "stage": node["node_id"],
                "task_class": node["llm_task_class"],
                "status": "FAILED",
                "failure_type": "SEMANTIC_VALIDATION_FAILURE",
                "schema_errors": schema_errors,
                "semantic_errors": semantic_errors,
                "raw_response": json.dumps(data),
                "actual_cost": 0.0,
            }])
        state["selected_question"] = data["selected_question"]
        state["candidate_evaluations"] = data["candidate_evaluations"]
        state["question_refinement_rationale"] = data["rationale"]
        state.setdefault("known_limitations", []).extend(data.get("limitations", []))
        state["question_refinement_field_provenance"] = {
            "selected_question": {
                "origin": state.get("candidate_question_field_provenance", [{}])[0].get("question", {}).get("origin", "existing_candidate"),
                "source": "single_candidate_supervisor_selection",
            },
            "candidate_evaluations": {
                key: {
                    "origin": value["origin"],
                    "model": value.get("model"),
                    "evidence": value.get("evidence", []),
                }
                for key, value in dimensions.items()
            },
            "rationale": {"origin": "deterministic_assembly"},
            "limitations": {"origin": "deterministic_state_diagnostics"},
        }
        self._persist_node_artifact(state, node, "specification.json", {
            **data,
            "field_provenance": state["question_refinement_field_provenance"],
        })

    def _atomic_semantic_field(self, state, node, step, objective, question, key, schema, extra_context=None):
        prompt = self._atomic_prompt(
            objective=objective, context={"question": question, **(extra_context or {})},
            expected=f'Return exactly {{"{key}":"..."}}.',
            constraints=["Do not answer the research question.", "Do not discuss novelty.",
                         "Do not claim an experiment or measurement has occurred.", "Do not describe JSON or instructions."],
        )
        self._record_prompt_telemetry(state, node, step, prompt, f"computational_{key}")
        response = self.gateway.generate_structured(
            state, LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=schema, semantic_validator=lambda data: validate_atomic_semantic_value(data, key, state.get("topic")),
            estimated_cost=0.001)
        return response["structured"][key], response

    def _extract_computational_skeleton_fields(self, state, node, question, step=1):
        variable_prompt = self._atomic_prompt(
            objective="Identify the single input quantity varied in this question.",
            context={"question": question},
            expected='Return exactly {"variable":"..."}.',
            constraints=["Use readable semantic text, not an identifier.", "Do not answer the question.", "Do not discuss novelty."],
        )
        self._record_prompt_telemetry(state, node, step, variable_prompt, "computational_variable_contract")
        try:
            variable_response = self.gateway.generate_structured(
                state, LLMRequest(variable_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=VARIABLE_VALUE_SCHEMA, semantic_validator=lambda data: semantic_display_errors(data.get("variable"), "independent variable"), estimated_cost=0.001)
        except StructuredGenerationExhausted as exc:
            raise StructuredGenerationExhausted("ATOMIC_LOCAL_REASONING_EXHAUSTED missing semantic field: independent_variable_contract", exc.attempts) from exc
        variable_data = variable_response["structured"]
        control_prompt = self._atomic_prompt(
            objective="Classify how the proposed input quantity is controlled in an experiment.",
            context={"question": question, "variable": variable_data["variable"]},
            expected="Return the constrained control_type field.", constraints=["Classify only."])
        self._record_prompt_telemetry(state, node, step + 1, control_prompt, "computational_variable_control_type")
        control_response = self.gateway.generate_structured(
            state, LLMRequest(control_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=CONTROL_TYPE_SCHEMA, estimated_cost=0.001)
        variable_data["control_type"] = control_response["structured"]["control_type"]
        source_data = None
        source_response = None
        if variable_data["control_type"] == "DERIVED_FROM_INPUT":
            source_prompt = self._atomic_prompt(
                objective="Identify the directly controlled input from which the proposed derived variable is determined.",
                context={"question": question, "derived_variable": variable_data["variable"]},
                expected="Return JSON fields source_variable and control_type matching the constrained schema.",
                constraints=["Use readable semantic text, not an identifier.", "Do not calculate a result."],
            )
            self._record_prompt_telemetry(state, node, step + 2, source_prompt, "computational_source_variable")
            try:
                source_response = self.gateway.generate_structured(
                    state, LLMRequest(source_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                    schema=SOURCE_VARIABLE_SCHEMA, semantic_validator=lambda data: semantic_display_errors(
                        data.get("source_variable"), "source variable"), estimated_cost=0.001)
            except StructuredGenerationExhausted as exc:
                raise StructuredGenerationExhausted("ATOMIC_LOCAL_REASONING_EXHAUSTED missing semantic field: derived_source_variable_contract", exc.attempts) from exc
            source_data = source_response["structured"]
        measurement_prompt = self._atomic_prompt(
            objective="Identify one neutral computational quantity to measure.",
            context={"question": question, "independent_variable": variable_data["variable"]},
            expected='Return exactly {"measurement":"..."}.',
            constraints=["Name an observable quantity only.", "Do not encode increase, decrease, improvement, or another expected result.",
                         "Do not repeat the independent variable as the measurement.",
                         "Generic measurement kinds include runtime, operation count, memory, correctness, or output value.",
                         "Use readable semantic text, not an identifier.", "Do not answer the question."],
        )
        self._record_prompt_telemetry(state, node, step + 3, measurement_prompt, "computational_measurement")
        try:
            measurement_response = self.gateway.generate_structured(
                state, LLMRequest(measurement_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=MEASUREMENT_VALUE_SCHEMA,
                semantic_validator=lambda data: (semantic_display_errors(data.get("measurement"), "dependent measurement")
                                                  + measurement_direction_errors(data.get("measurement"))),
                estimated_cost=0.001)
        except StructuredGenerationExhausted as exc:
            raise StructuredGenerationExhausted("ATOMIC_LOCAL_REASONING_EXHAUSTED missing semantic field: dependent_measurement_contract", exc.attempts) from exc
        measurement_data = measurement_response["structured"]
        kind_prompt = self._atomic_prompt(
            objective="Classify the kind of computational measurement.",
            context={"measurement": measurement_data["measurement"]},
            expected="Return the constrained measurement_kind field.", constraints=["Classify only."])
        self._record_prompt_telemetry(state, node, step + 4, kind_prompt, "computational_measurement_kind")
        kind_response = self.gateway.generate_structured(
            state, LLMRequest(kind_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=MEASUREMENT_KIND_SCHEMA, estimated_cost=0.001)
        measurement_data["measurement_kind"] = kind_response["structured"]["measurement_kind"]
        neutrality_prompt = self._atomic_prompt(
            objective="Classify whether the proposed dependent measurement is a neutral observable rather than an expected result.",
            context={"measurement": measurement_data["measurement"], "measurement_kind": measurement_data["measurement_kind"]},
            expected='Return exactly {"assessment":"NEUTRAL_MEASUREMENT|RESULT_EMBEDDED|NOT_MEASURABLE|UNCERTAIN"}.',
            constraints=["Classify only.", "Do not provide a rationale."],
        )
        self._record_prompt_telemetry(state, node, step + 5, neutrality_prompt, "computational_measurement_neutrality")
        try:
            neutrality_response = self.gateway.generate_structured(
                state, LLMRequest(neutrality_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=MEASUREMENT_NEUTRALITY_SCHEMA, estimated_cost=0.001)
        except StructuredGenerationExhausted as exc:
            raise StructuredGenerationExhausted("ATOMIC_LOCAL_REASONING_EXHAUSTED missing semantic field: measurement_neutrality", exc.attempts) from exc
        variable = {"display_text": variable_data["variable"],
                    "canonical_id": canonical_semantic_id(variable_data["variable"]),
                    "control_type": variable_data["control_type"]}
        if source_data:
            variable["source_variable"] = {"display_text": source_data["source_variable"],
                                           "canonical_id": canonical_semantic_id(source_data["source_variable"]),
                                           "control_type": source_data["control_type"]}
        measurement = measurement_from_kind(measurement_data["measurement_kind"],
                                            observable_display=measurement_data["measurement"])
        if measurement is None:
            measurement = {"display_text": measurement_data["measurement"],
                           "canonical_id": canonical_semantic_id(measurement_data["measurement"]),
                           "measurement_kind": measurement_data["measurement_kind"],
                           "measurement_source": "UNCERTAIN", "measurement_observable": {},
                           "semantic_requirement": "UNRESOLVED"}
        measurement["neutrality"] = neutrality_response["structured"]["assessment"]
        info_prompt = self._atomic_prompt(
            objective="Classify whether observing this quantity while varying the independent variable could materially constrain an answer.",
            context={"question": question, "independent_variable": variable, "measurement_observable": measurement["measurement_observable"]},
            expected="Return only the constrained assessment enum.", constraints=["Do not assess novelty.", "Return no rationale."])
        self._record_prompt_telemetry(state, node, step + 6, info_prompt, "measurement_informativeness")
        info_response = self.gateway.generate_structured(state, LLMRequest(info_prompt, stage=node["node_id"],
            requested_model_class="CHEAP", task_class=node["llm_task_class"]), schema=INFORMATIVENESS_SCHEMA, estimated_cost=0.001)
        measurement["informativeness"] = info_response["structured"]["assessment"]
        return variable, measurement, {"variable": variable_response, "source_variable": source_response,
                                       "control_type": control_response, "measurement": measurement_response,
                                       "measurement_kind": kind_response, "neutrality": neutrality_response,
                                       "informativeness": info_response}

    def _guided_computational_question_refinement(self, state, node, candidate, contract):
        recovery = state.get("computational_measurement_recovery")
        if recovery:
            continuation_id = recovery.get("external_continuation_id")
            lifecycle = continuation_record(state, continuation_id) if continuation_id else None
            if lifecycle:
                lifecycle["status"] = "CONTINUATION_STARTED"
                lifecycle["updated_at"] = now_iso()
                lifecycle.setdefault("events", []).append({"event": "CONTINUATION_STARTED", "timestamp": now_iso()})
                state["active_external_continuation"] = lifecycle
                attempt_id = recovery.get("continuation_attempt_id")
                attempt = next((item for item in lifecycle.get("attempts", []) if item.get("attempt_id") == attempt_id), None)
                if attempt:
                    attempt["status"] = "CONTINUATION_STARTED"
                    attempt["execution_started_at"] = now_iso()
            return self._recover_computational_measurement(state, node, candidate, contract, recovery)
        original_question = next((item.get("original_candidate") for item in state.get("candidate_clarification_history", [])
                                  if item.get("original_candidate")), candidate["question"])
        question = original_question
        if candidate["question"] != original_question:
            question = candidate["question"]
        modality = assess_automation_closure(["executable_computation"], self._current_capability_semantics(state),
                                             state.get("verified_evidence_artifacts", []))
        variable, measurement, atomic_provenance = self._extract_computational_skeleton_fields(state, node, question)
        clarification = next((deepcopy(item) for item in reversed(state.get("candidate_clarification_history", []))
                              if item.get("rewritten_candidate") == question), None)
        preliminary = finalize_skeleton_status({"independent_variable": variable, "dependent_measurement": measurement,
            "bounded_regime": contract.get("scope_type") == "BOUNDED_COMPUTATIONAL", "required_modality": "executable_computation",
            "automation_closure": modality["automation_closure"], "computational_testability": "UNCERTAIN"})
        operational_errors = [item for item in preliminary["validation_errors"] if "testability" not in item]
        if operational_errors:
            prompt = self._atomic_prompt(
                objective="Rewrite the question so it explicitly varies one controllable input parameter and measures one computational outcome under a bounded input regime.",
                context={"question": question, "rejection_reasons": operational_errors,
                         "proposed_variable": variable, "proposed_measurement": measurement},
                expected='Return exactly {"question":"..."}.',
                constraints=["Preserve the scientific intent.", "Remain topic-connected and bounded computational.",
                             "Do not claim novelty.", "Do not answer the question.", "Do not add a universal theorem claim."],
            )
            self._record_prompt_telemetry(state, node, 4, prompt, "computational_candidate_clarification")
            response = self.gateway.generate_structured(
                state, LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=ONE_QUESTION_SCHEMA,
                semantic_validator=lambda data: validate_candidate_clarification(data, original_question, state.get("topic")),
                estimated_cost=0.001)
            question = normalize_atomic_question(response["structured"]["question"])
            clarification = {"original_candidate": original_question,
                             "clarification_reason": "OPERATIONALIZATION_NEEDS_CLARIFICATION: " + "; ".join(operational_errors),
                             "rewritten_candidate": question, "rewrite_count": 1,
                             "provenance": {"origin": "local_model", "response": response}, "created_at": now_iso()}
            state.setdefault("candidate_clarification_history", []).append(deepcopy(clarification))
            variable, measurement, atomic_provenance = self._extract_computational_skeleton_fields(state, node, question, step=5)
        prompt = self._atomic_prompt(
            objective="Classify whether executable results could distinguish different answers to this bounded computational question.",
            context={"question": question, "independent_variable": variable,
                     "dependent_measurement": measurement, "bounded_scope": True},
            expected='Return exactly {"assessment":"TESTABLE|NOT_TESTABLE|UNCERTAIN"}.',
            constraints=["Classify only.", "Do not provide a rationale.", "Do not answer the research question."],
        )
        self._record_prompt_telemetry(state, node, 8, prompt, "computational_testability")
        test_response = self.gateway.generate_structured(
            state, LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=TESTABILITY_SCHEMA, estimated_cost=0.001)
        testability = test_response["structured"]["assessment"]
        legacy = legacy_falsifiability_from_testability(testability)
        observation = None
        observation_provenance = None
        if testability == "TESTABLE":
            try:
                observation, observation_provenance = self._atomic_semantic_field(
                    state, node, 9, "State one possible experimental observation that would meaningfully distinguish among possible answers.",
                    question, "observation", OBSERVATION_SCHEMA,
                    {"independent_variable": variable, "dependent_measurement": measurement})
            except StructuredGenerationExhausted as exc:
                observation_provenance = {"status": "OPTIONAL_ENRICHMENT_FAILED", "error": str(exc)}
        skeleton = {"candidate_question": question, "original_candidate": original_question,
                    "independent_variable": variable, "dependent_measurement": measurement,
                    "bounded_scope": True, "bounded_regime": {"scope_type": "BOUNDED_COMPUTATIONAL",
                        "reproducibility_requirement": "persisted generator parameters, bounded domain, and seeds where applicable"},
                    "required_modality": "executable_computation",
                    "computational_testability": testability, "distinguishing_observation": observation,
                    "automation_closure": modality["automation_closure"],
                    "artifact_type": "COMPUTATIONAL_EXPERIMENTAL_SKELETON", "not_full_experiment_contract": True,
                    "clarification": clarification,
                    "field_provenance": {"atomic_fields": atomic_provenance, "testability": test_response,
                                         "observation": observation_provenance, "legacy_falsifiability": legacy},
                    "created_at": now_iso()}
        finalize_skeleton_status(skeleton)
        if clarification:
            clarification.update(clarification_effective(operational_errors, skeleton))
            skeleton["clarification"] = clarification
        state["computational_experimental_skeleton"] = skeleton
        if skeleton["experimental_skeleton_status"] != "VALID":
            missing = "; ".join(skeleton["validation_errors"])
            self._external_reasoning_required(state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED experimental_skeleton missing semantic field: " + missing)
            return
        if question != original_question:
            state.setdefault("candidate_question_history", []).append({"candidate_questions": [deepcopy(candidate)],
                "trigger": "operationalization_clarification", "created_at": now_iso()})
            state["candidate_questions"] = [{**candidate, "question": question}]
            state["candidate_evidence_contracts"][question] = {**contract, "clarified_from": original_question}
        evaluation = {"question": question, "feasibility": 1.0, "novelty_potential": 0.0,
                      "novelty_status": state.get("prior_work_coverage", {}).get("novelty_status", "NOT_ESTABLISHED"),
                      "falsifiability": legacy["value"], "testability_status": testability,
                      "falsifiability_provenance": legacy, "evidence_accessibility": 1.0,
                      "rationale": "Bounded computational variables and measurements passed typed testability and verified modality checks.",
                      **modality}
        data = {"selected_question": question, "candidate_evaluations": [evaluation],
                "rationale": "The supervisor selected the bounded computational candidate after typed operationalization and automation checks.",
                "limitations": ["Novelty is not established because prior-work coverage remains unknown."]}
        schema_errors = validate_json_schema_subset(data, QUESTION_REFINEMENT_SCHEMA)
        semantic_errors = [] if schema_errors else validate_question_refinement_semantics(data, state.get("topic"))
        if schema_errors or semantic_errors:
            raise StructuredGenerationExhausted("SEMANTIC_VALIDATION_FAILURE computational refinement assembly invalid", [{
                "schema_errors": schema_errors, "semantic_errors": semantic_errors, "raw_response": json.dumps(data), "actual_cost": 0.0}])
        state["selected_question"] = question
        state["candidate_evaluations"] = data["candidate_evaluations"]
        state["question_refinement_rationale"] = data["rationale"]
        state.setdefault("known_limitations", []).extend(data["limitations"])
        self._persist_node_artifact(state, node, "specification.json", {**data, "experimental_skeleton": skeleton})

    def _select_measurement_kind(self, state, node, question, variable, step=1, forced_kind=None, excluded_kinds=None):
        excluded_kinds = set(excluded_kinds or [])
        prompt = self._atomic_prompt(
            objective="Choose one class of neutral computational quantity that could be observed while the independent variable changes.",
            context={"topic": state.get("topic"), "question": question, "independent_variable": variable,
                     "bounded_scope": "BOUNDED_COMPUTATIONAL"},
            expected="Return only the constrained measurement_kind enum.",
            constraints=["Do not predict a direction or result.", "Do not answer the question.",
                         "Do not discuss novelty.", "Return no rationale."],
        )
        if forced_kind:
            kind = forced_kind
            response = {"structured": {"measurement_kind": kind}, "origin": "preserved_measurement_kind"}
        else:
            self._record_prompt_telemetry(state, node, step, prompt, "measurement_kind_selection")
            allowed_kinds = [item for item in MEASUREMENT_KIND_SCHEMA["properties"]["measurement_kind"]["enum"]
                             if item not in excluded_kinds]
            selection_schema = deepcopy(MEASUREMENT_KIND_SCHEMA)
            selection_schema["properties"]["measurement_kind"]["enum"] = allowed_kinds
            response = self.gateway.generate_structured(
                state, LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=selection_schema, estimated_cost=0.001)
            kind = response["structured"]["measurement_kind"]
        other_response = None
        observable_display = None
        if kind in {"output_value", "other"}:
            observable_prompt = self._atomic_prompt(
                objective="Name the specific algorithm output property recorded in this bounded computational experiment.",
                context={"question": question, "independent_variable": variable, "measurement_kind": kind},
                expected='Return exactly {"observable":"..."}.',
                constraints=["Name an observable output only.", "Do not predict how it changes.", "Do not repeat the measurement category.",
                             "Do not answer the question.", "Do not claim novelty.", "Do not describe JSON or instructions."],
            )
            self._record_prompt_telemetry(state, node, step + 1, observable_prompt, "measurement_observable")
            other_response = self.gateway.generate_structured(
                state, LLMRequest(observable_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=OBSERVABLE_SCHEMA, semantic_validator=lambda data: observable_specificity_errors(data.get("observable"), kind)
                    + measurement_direction_errors(data.get("observable"))
                    + ([] if measurement_source_compatible(kind, classify_measurement_source(data.get("observable")))
                       else [f"observable source {classify_measurement_source(data.get('observable'))} is incompatible with measurement_kind {kind}"]),
                estimated_cost=0.001)
            observable_display = other_response["structured"]["observable"]
            errors = []
            if canonical_semantic_id(observable_display) == variable.get("canonical_id"):
                errors.append("dependent measurement must be distinct from the independent variable")
            if errors:
                raise StructuredGenerationExhausted("ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=other_measurement_value", [{
                    "status": "FAILED", "semantic_errors": errors, "parsed_response": {"observable": observable_display}}])
        measurement = measurement_from_kind(kind, observable_display=observable_display)
        if not measurement:
            return kind, None, {"kind": response, "observable": other_response}
        if not measurement_source_compatible(kind, measurement.get("measurement_source")):
            raise StructuredGenerationExhausted("ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=measurement_source_classification", [{
                "status": "FAILED", "parsed_response": {"measurement_kind": kind, "observable": observable_display,
                "source": measurement.get("measurement_source")}, "semantic_errors": ["measurement kind/source incompatibility"]}])
        info_prompt = self._atomic_prompt(
            objective="Classify whether observing this quantity while varying the independent variable could materially constrain an answer to the research question.",
            context={"question": question, "independent_variable": variable,
                     "measurement_observable": measurement["measurement_observable"]},
            expected="Return only the constrained assessment enum.",
            constraints=["Do not assess novelty.", "Do not answer the research question.", "Return no rationale."],
        )
        self._record_prompt_telemetry(state, node, step + 2, info_prompt, "measurement_informativeness")
        info_response = self.gateway.generate_structured(
            state, LLMRequest(info_prompt, stage=node["node_id"], requested_model_class="CHEAP",
                              task_class=node["llm_task_class"], semantic_task="measurement_informativeness"),
            schema=INFORMATIVENESS_SCHEMA, estimated_cost=0.001)
        measurement["informativeness"] = info_response["structured"]["assessment"]
        return kind, measurement, {"kind": response, "observable": other_response,
                                   "measurement_source": {"source": measurement.get("measurement_source"),
                                       "origin": "deterministic_measurement_source_registry"},
                                   "informativeness": info_response}

    def _complete_recovered_computational_refinement(self, state, node, candidate, contract, skeleton, provenance):
        skeleton["experiment_relation_coherence"] = experiment_relation_coherence(
            skeleton.get("independent_variable", {}), skeleton.get("dependent_measurement", {}))
        finalize_skeleton_status(skeleton)
        state["computational_experimental_skeleton"] = skeleton
        if skeleton["experimental_skeleton_status"] != "VALID":
            active = state.get("active_external_continuation")
            if active:
                active["status"] = "CONTINUATION_SEMANTIC_REJECTED"
                active["continuation_validation_result"] = deepcopy(skeleton.get("validation_errors", []))
                active["updated_at"] = now_iso()
                active.setdefault("events", []).append({"event": "CONTINUATION_VALIDATION_REJECTED",
                    "timestamp": now_iso(), "errors": deepcopy(skeleton.get("validation_errors", []))})
            self._external_reasoning_required(state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED experimental_skeleton missing semantic field: "
                                              + "; ".join(skeleton["validation_errors"]))
            return
        question = deterministic_bounded_question(skeleton["independent_variable"]["display_text"],
                                                  skeleton["dependent_measurement"]["display_text"])
        question_errors = validate_single_question(question, state.get("topic"), []) + validate_bounded_computational_question(question)
        if question_errors:
            self._external_reasoning_required(state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=final_question_surface_form")
            return
        skeleton["candidate_question"] = question
        skeleton["question_surface_origin"] = "deterministic_validated_skeleton_assembly"
        skeleton["field_provenance"].update(provenance)
        candidate = {**candidate, "question": question}
        state["candidate_questions"] = [candidate]
        state["candidate_evidence_contracts"][question] = {**contract, "skeleton_status": "VALID"}
        modality = assess_automation_closure(["executable_computation"], self._current_capability_semantics(state),
                                             state.get("verified_evidence_artifacts", []))
        legacy = legacy_falsifiability_from_testability("TESTABLE")
        evaluation = {"question": question, "feasibility": 1.0, "novelty_potential": 0.0,
                      "novelty_status": state.get("prior_work_coverage", {}).get("novelty_status", "NOT_ESTABLISHED"),
                      "falsifiability": legacy["value"], "testability_status": "TESTABLE",
                      "falsifiability_provenance": legacy, "evidence_accessibility": 1.0,
                      "rationale": "A valid bounded computational skeleton has distinct controlled and measured quantities.", **modality}
        data = {"selected_question": question, "candidate_evaluations": [evaluation],
                "rationale": "The supervisor assembled the question from a valid skeleton after bounded field recovery.",
                "limitations": ["Control availability remains proposed until feasibility verification.",
                                "Novelty is not established because prior-work coverage remains unknown."]}
        state["selected_question"] = question
        state["candidate_evaluations"] = [evaluation]
        state["question_refinement_rationale"] = data["rationale"]
        state.setdefault("known_limitations", []).extend(data["limitations"])
        self._persist_node_artifact(state, node, "specification.json", {**data, "experimental_skeleton": skeleton})
        active = state.get("active_external_continuation")
        if active:
            active["status"] = "CONTINUATION_APPLIED"
            active["continuation_validation_result"] = "PASSED"
            active["resulting_skeleton_status"] = skeleton.get("experimental_skeleton_status")
            active["updated_at"] = now_iso()
            active.setdefault("events", []).extend([
                {"event": "CONTINUATION_VALIDATED", "timestamp": now_iso()},
                {"event": "CONTINUATION_APPLIED", "timestamp": now_iso(), "node_id": node["node_id"]},
            ])
            attempt_id = (state.get("computational_measurement_recovery") or {}).get("continuation_attempt_id")
            attempt = next((item for item in active.get("attempts", []) if item.get("attempt_id") == attempt_id), None)
            if attempt:
                attempt["status"] = "RETRY_SUCCEEDED"
                attempt["completed_at"] = now_iso()
            for request in state.get("engineering_requests", []):
                if request.get("status") == "OPEN" and request.get("problem") == "LOCAL_INFERENCE_RUNTIME_UNAVAILABLE":
                    request["status"] = "RECOVERED"
                    request["recovered_at"] = now_iso()
                    request["recovered_by_continuation_id"] = active.get("continuation_id")
            state.pop("active_external_continuation", None)
            state.pop("computational_measurement_recovery", None)

    def _recover_computational_measurement(self, state, node, candidate, contract, recovery):
        prior_skeleton = deepcopy(recovery["skeleton"])
        if recovery.get("skip_current_candidate_recovery"):
            return self._fresh_skeleton_first_candidate(
                state, node, candidate, contract, recovery,
                recovery.get("retirement_reason", "current candidate field recovery was exhausted"))
        variable = deepcopy(prior_skeleton["independent_variable"])
        variable["control_availability"] = control_availability_for_refinement(variable.get("control_type"))
        if recovery.get("recovered_measurement"):
            measurement = deepcopy(recovery["recovered_measurement"])
            kind = measurement.get("measurement_kind")
            selection_provenance = {"semantic_source": "HISTORICAL_MODEL_OUTPUT_REEVALUATED",
                                    "normalization_only": True}
        else:
            try:
                external_kind = recovery.get("external_measurement_kind")
                kind, measurement, selection_provenance = self._select_measurement_kind(
                    state, node, candidate["question"], variable, step=1,
                    forced_kind=external_kind or (None if recovery.get("measurement_kind_reselection") else prior_skeleton.get("dependent_measurement", {}).get("measurement_kind")),
                    excluded_kinds=recovery.get("excluded_measurement_kinds"))
                if external_kind:
                    selection_provenance["kind"] = {
                        "structured": {"measurement_kind": external_kind},
                        "origin": "EXTERNAL_REASONING",
                        "decision_id": recovery.get("external_response_decision_id"),
                    }
                    active = state.get("active_external_continuation")
                    if active:
                        active["semantic_handler_result"] = deepcopy(measurement)
                        active.setdefault("events", []).append({"event": "STANDARD_MEASUREMENT_SEMANTICS_DERIVED",
                            "timestamp": now_iso(), "measurement_kind": external_kind,
                            "measurement": deepcopy(measurement)})
            except StructuredGenerationExhausted as exc:
                if recovery.get("measurement_kind_reselection"):
                    self._external_reasoning_required(
                        state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=bounded_measurement_kind_reselection")
                    return
                return self._fresh_skeleton_first_candidate(state, node, candidate, contract, recovery, str(exc))
        if kind == "uncertain" or measurement is None:
            return self._fresh_skeleton_first_candidate(state, node, candidate, contract, recovery, "measurement_kind_selection=uncertain")
        coherence = variable_measurement_coherence(variable, measurement)
        if coherence != "COHERENT":
            return self._fresh_skeleton_first_candidate(state, node, candidate, contract, recovery, "measurement coherence=" + coherence)
        skeleton = {"original_candidate": recovery.get("original_candidate") or prior_skeleton.get("original_candidate"),
                    "prior_clarified_candidate": candidate["question"], "independent_variable": variable,
                    "dependent_measurement": measurement, "variable_measurement_coherence": coherence,
                    "bounded_scope": True, "bounded_regime": prior_skeleton.get("bounded_regime") or {"scope_type": "BOUNDED_COMPUTATIONAL"},
                    "required_modality": "executable_computation", "automation_closure": "HIGH",
                    "computational_testability": "TESTABLE", "experimental_skeleton_status": "UNCERTAIN",
                    "artifact_type": "COMPUTATIONAL_EXPERIMENTAL_SKELETON", "not_full_experiment_contract": True,
                    "candidate_disposition": "RECOVERED", "field_provenance": {}, "created_at": now_iso()}
        skeleton["experiment_relation_coherence"] = experiment_relation_coherence(variable, measurement)
        finalize_skeleton_status(skeleton)
        state["computational_experimental_skeleton"] = deepcopy(skeleton)
        if recovery.get("measurement_kind_reselection"):
            state.setdefault("measurement_kind_attempt_history", []).append({
                "measurement_kind": kind, "observable": deepcopy(measurement.get("measurement_observable")),
                "measurement_source": measurement.get("measurement_source"),
                "informativeness": measurement.get("informativeness"),
                "relation_coherence": skeleton.get("experiment_relation_coherence"),
                "status": "ACCEPTED_REPLACEMENT" if skeleton["experimental_skeleton_status"] == "VALID" else "UNUSABLE_REPLACEMENT",
                "failure_reason": None if skeleton["experimental_skeleton_status"] == "VALID" else "; ".join(skeleton["validation_errors"]),
                "created_at": now_iso()})
        if skeleton["experimental_skeleton_status"] != "VALID":
            if recovery.get("measurement_kind_reselection"):
                active = state.get("active_external_continuation")
                if active:
                    active["status"] = "CONTINUATION_SEMANTIC_REJECTED"
                    active["continuation_validation_result"] = deepcopy(skeleton["validation_errors"])
                    active["updated_at"] = now_iso()
                    active.setdefault("events", []).append({"event": "CONTINUATION_VALIDATION_REJECTED",
                        "timestamp": now_iso(), "errors": deepcopy(skeleton["validation_errors"])})
                    attempt_id = recovery.get("continuation_attempt_id")
                    attempt = next((item for item in active.get("attempts", []) if item.get("attempt_id") == attempt_id), None)
                    if attempt:
                        attempt["status"] = "RETRY_SEMANTIC_REJECTED"; attempt["completed_at"] = now_iso()
                state.setdefault("retired_candidate_history", []).append({"candidate": deepcopy(candidate),
                    "status": "UNSUITABLE_OPERATIONALIZATION", "reason": "; ".join(skeleton["validation_errors"]),
                    "created_at": now_iso()})
                self._external_reasoning_required(
                    state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=neutral_computational_measurement",
                    computational_measurement_decision_contract(state, node["node_id"]))
                return
            return self._fresh_skeleton_first_candidate(
                state, node, candidate, contract, recovery, "; ".join(skeleton["validation_errors"]))
        return self._complete_recovered_computational_refinement(state, node, candidate, contract, skeleton,
                                                                {"measurement_selection": selection_provenance})

    def _fresh_skeleton_first_candidate(self, state, node, candidate, contract, recovery, retirement_reason):
        if state.get("fresh_skeleton_regeneration_count", 0) >= 1:
            self._external_reasoning_required(state, node, "NO_COHERENT_COMPUTATIONAL_CANDIDATE semantic_task=measurement_kind_selection")
            return
        state["fresh_skeleton_regeneration_count"] = 1
        state.setdefault("retired_candidate_history", []).append({"candidate": deepcopy(candidate),
            "status": "UNSUITABLE_OPERATIONALIZATION", "reason": retirement_reason, "created_at": now_iso()})
        recovered_value = recovery.get("recovered_historical_variable")
        if recovered_value:
            variable_text = recovered_value["normalized_display_text"]
            variable_response = {"structured": {"variable": variable_text},
                                 "semantic_source": "HISTORICAL_MODEL_OUTPUT_REEVALUATED",
                                 "reevaluation": deepcopy(recovered_value)}
        else:
            variable_prompt = self._atomic_prompt(
                objective="Identify one controllable input quantity for a bounded computational investigation of TOPIC.",
                context={"topic": state.get("topic")}, expected='Return exactly {"variable":"input size"}.',
                constraints=["Name a semantic control candidate.", "Do not answer a research question."])
            self._record_prompt_telemetry(state, node, 3, variable_prompt, "fresh_skeleton_variable")
            try:
                variable_response = self.gateway.generate_structured(state, LLMRequest(variable_prompt, stage=node["node_id"],
                    requested_model_class="CHEAP", task_class=node["llm_task_class"]), schema=VARIABLE_VALUE_SCHEMA,
                    semantic_validator=lambda data: ([] if semantic_value_record(data.get("variable"))["semantic_validation_status"] == "VALID"
                                                     else ["independent variable is not a semantic control candidate"]), estimated_cost=0.001)
            except StructuredGenerationExhausted:
                self._external_reasoning_required(state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=fresh_variable_generation")
                return
            variable_semantics = semantic_value_record(variable_response["structured"]["variable"])
            variable_text = variable_semantics["normalized_display_text"]
        if recovery.get("recovered_control_type"):
            control_response = {"structured": {"control_type": recovery["recovered_control_type"]},
                                "semantic_source": "PRESERVED_VALIDATED_FIELD"}
        else:
            control_prompt = self._atomic_prompt(objective="Classify how this input quantity would be controlled.",
                context={"topic": state.get("topic"), "variable": variable_text}, expected="Return only control_type.", constraints=["Classify only."])
            control_response = self.gateway.generate_structured(state, LLMRequest(control_prompt, stage=node["node_id"],
                requested_model_class="CHEAP", task_class=node["llm_task_class"]), schema=CONTROL_TYPE_SCHEMA, estimated_cost=0.001)
        variable = {"display_text": variable_text, "canonical_id": canonical_semantic_id(variable_text),
                    "raw_model_value": (recovered_value or {}).get("raw_model_value", variable_response["structured"]["variable"]),
                    "normalization_strategy": (recovered_value or normalize_semantic_value(variable_response["structured"]["variable"]))["normalization_strategy"],
                    "semantic_validation_status": "VALID",
                    "control_type": control_response["structured"]["control_type"]}
        variable["control_availability"] = control_availability_for_refinement(variable["control_type"])
        source_response = None
        if variable["control_type"] == "DERIVED_FROM_INPUT":
            source_prompt = self._atomic_prompt(
                objective="Identify the directly controlled input from which this proposed variable is derived.",
                context={"topic": state.get("topic"), "derived_variable": variable_text},
                expected="Return source_variable and its constrained control_type.",
                constraints=["Use readable semantic text, not an identifier.",
                             "The source control must be DIRECT_INPUT or ALGORITHM_PARAMETER.",
                             "Do not answer the research question."],)
            self._record_prompt_telemetry(state, node, 4, source_prompt, "fresh_skeleton_source_variable")
            try:
                source_response = self.gateway.generate_structured(
                    state, LLMRequest(source_prompt, stage=node["node_id"], requested_model_class="CHEAP",
                                      task_class=node["llm_task_class"]),
                    schema=SOURCE_VARIABLE_SCHEMA,
                    semantic_validator=lambda data: semantic_display_errors(data.get("source_variable"), "source variable")
                        + ([] if data.get("control_type") in {"DIRECT_INPUT", "ALGORITHM_PARAMETER"}
                           else ["derived source control_type must be directly controllable"]),
                    estimated_cost=0.001)
            except StructuredGenerationExhausted:
                self._external_reasoning_required(
                    state, node, "ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task=fresh_derived_source_variable")
                return
            source_data = source_response["structured"]
            variable["source_variable"] = {
                "display_text": source_data["source_variable"],
                "canonical_id": canonical_semantic_id(source_data["source_variable"]),
                "control_type": source_data["control_type"],
                "control_availability": control_availability_for_refinement(source_data["control_type"]),}
        kind, measurement, selection = self._select_measurement_kind(
            state, node, state.get("topic"), variable, step=5,
            forced_kind=recovery.get("recovered_measurement_kind"))
        if not measurement or variable_measurement_coherence(variable, measurement) != "COHERENT":
            self._external_reasoning_required(state, node, "NO_COHERENT_COMPUTATIONAL_CANDIDATE semantic_task=measurement_kind_selection")
            return
        skeleton = {"original_candidate": recovery.get("original_candidate"), "independent_variable": variable,
                    "dependent_measurement": measurement, "variable_measurement_coherence": "COHERENT",
                    "bounded_scope": True, "bounded_regime": {"scope_type": "BOUNDED_COMPUTATIONAL",
                        "reproducibility_requirement": "persisted generator parameters, bounded domain, and seeds where applicable"},
                    "required_modality": "executable_computation", "automation_closure": "HIGH",
                    "computational_testability": "TESTABLE", "artifact_type": "COMPUTATIONAL_EXPERIMENTAL_SKELETON",
                    "not_full_experiment_contract": True, "candidate_disposition": "REGENERATED_SKELETON_FIRST",
                    "field_provenance": {"variable": variable_response, "control_type": control_response,
                                         "source_variable": source_response}, "created_at": now_iso()}
        return self._complete_recovered_computational_refinement(state, node, candidate, contract, skeleton,
                                                                {"measurement_selection": selection})

    def _regenerate_candidate_from_evidence(self, state, node, trigger, reasons):
        maximum = int(os.environ.get("RESEARCH_CANDIDATE_REGENERATION_LIMIT", "2"))
        used = int(state.get("candidate_question_regeneration_count", 0))
        if used >= maximum:
            return False
        previous = deepcopy(state.get("candidate_questions", []))
        state.setdefault("candidate_question_history", []).append({
            "candidate_questions": previous,
            "validation": deepcopy(state.get("candidate_question_validation")),
            "trigger": trigger,
            "reasons": list(reasons),
            "created_at": now_iso(),
        })
        question, response, validation = self._generate_guided_candidate_question(
            state, node, state.get("literature_cache", []), step=used + 4
        )
        state["candidate_question_regeneration_count"] = used + 1
        state["candidate_question_validation"] = validation
        if not validation["substantive_question"]:
            return self._regenerate_candidate_from_evidence(state, node, "regenerated_candidate_invalid", validation["rejection_reasons"])
        state["candidate_questions"] = [{
            "question": question,
            "why_interesting": "Grounded in retrieved literature and suitable for later feasibility checks.",
            "falsifiability": "Observable evidence can contradict or constrain the proposed relationship.",
            "local_executability": "Initial evidence checks can use local/free resources.",
        }]
        state["candidate_question_field_provenance"] = [{
            "question": {"origin": "local_model", "source": "bounded_candidate_regeneration", "trigger": trigger, "model": response.get("model")},
            "why_interesting": {"origin": "deterministic_supervisor"},
            "falsifiability": {"origin": "deterministic_supervisor"},
            "local_executability": {"origin": "deterministic_supervisor"},
        }]
        state.setdefault("guided_agent_steps", []).append({
            "node_id": node["node_id"], "microstep": "candidate_regeneration",
            "trigger": trigger, "question": question, "candidate_question_validation": validation,
            "created_at": now_iso(),
        })
        return True

    def _assess_novelty_with_challenge(self, state, node, question):
        query_prompt = "\n".join([
            "OBJECTIVE", "Write one scholarly search phrase that would find prior work most similar to this research question.",
            "QUESTION", question,
            "CONSTRAINTS", "Search for existing work that could challenge novelty.", "Keep it short.", "Do not invent citations.",
            "EXPECTED", 'Return {"query":"..."}.',
        ])
        self._record_prompt_telemetry(state, node, 3, query_prompt, "novelty_challenge_query")
        raw_query = None
        query_origin = "local_model"
        try:
            response = self.gateway.generate_structured(
                state,
                LLMRequest(query_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
                schema=SEARCH_QUERY_SCHEMA,
                semantic_validator=lambda data: validate_novelty_search_query(data.get("query"), state.get("topic"), question),
                estimated_cost=0.001,
            )
            raw_query = response["structured"].get("query")
        except Exception as exc:
            state.setdefault("guided_agent_steps", []).append({
                "node_id": node["node_id"], "microstep": "novelty_challenge_query",
                "status": "MODEL_QUERY_FAILED", "error": str(exc), "created_at": now_iso(),
            })
        if validate_novelty_search_query(raw_query, state.get("topic"), question):
            raw_query = " ".join(sorted(lexical_terms(question))[:10])
            query_origin = "deterministic_candidate_terms"
        normalized_query = re.sub(r"_+", " ", _text(raw_query)).strip()
        controller = GuidedToolController(self.store, state["run_id"], self.work_root, self.literature_provider)
        tool_result = controller.execute({
            "tool": "literature_search",
            "arguments": {"query": normalized_query, "limit": int(os.environ.get("RESEARCH_LITERATURE_LIMIT", "5"))},
        })
        results = dedupe_records(tool_result["result"].get("records", []))
        relevance = literature_relevance_report(results, state.get("topic"), [{"question": question}])
        challenge = {
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "query_origin": query_origin,
            "results": results,
            "relevance": relevance,
            "retrieved_at": now_iso(),
            "provider": tool_result["result"].get("provider") or getattr(self.literature_provider, "provider_name", None),
            "tool_artifact": tool_result.get("artifact"),
        }
        state.setdefault("novelty_challenge_history", []).append(challenge)
        state["novelty_challenge"] = challenge
        cards = compact_literature_cards(results, limit=4)
        assessment_prompt = "\n".join([
            "OBJECTIVE", "Assess whether the candidate appears already well covered, has a plausible specific gap, or remains uncertain.",
            "QUESTION", question,
            "PRIOR-WORK CARDS", json.dumps(cards, sort_keys=True),
            "RULES", "Assess prior-work coverage only. Do not answer the research question.",
            "Use uncertain when these records cannot support a coverage judgment.",
            "EXPECTED", 'Return {"assessment":"plausible_gap|well_covered|uncertain","reason":"...","confidence":0..1}.',
        ])
        self._record_prompt_telemetry(state, node, 4, assessment_prompt, "novelty_assessment")
        response = self.gateway.generate_structured(
            state,
            LLMRequest(assessment_prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=NOVELTY_ASSESSMENT_SCHEMA,
            semantic_validator=lambda data: validate_novelty_assessment(data, state.get("topic")),
            estimated_cost=0.001,
        )
        data = response["structured"]
        score_map = {"plausible_gap": 0.6, "well_covered": 0.0, "uncertain": None}
        return {
            "score": score_map[data["assessment"]], "reason": data["reason"],
            "assessment": data["assessment"], "confidence": float(data["confidence"]),
            "origin": "local_model_with_novelty_challenge", "model": response.get("model"), "cards": cards,
        }

    def _regenerate_candidate_from_gap(self, state, node, question, cards):
        if int(state.get("gap_grounded_candidate_regeneration_count", 0)) >= 1:
            return False
        prompt = "\n".join([
            "OBJECTIVE", "Make this research question more specific using the supplied gap evidence.",
            "CURRENT QUESTION", question,
            "PRIOR-WORK OBSERVATIONS", json.dumps(cards, sort_keys=True),
            "CONSTRAINTS", "Keep it empirically testable.", "Do not invent a gap.",
            "Do not ask what the literature says.", "Produce one question only.",
            "EXPECTED", 'Return {"question":"..."}.',
        ])
        self._record_prompt_telemetry(state, node, 5, prompt, "gap_grounded_candidate_regeneration")
        response = self.gateway.generate_structured(
            state,
            LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP", task_class=node["llm_task_class"]),
            schema=ONE_QUESTION_SCHEMA,
            semantic_validator=lambda data: validate_single_question(data.get("question"), state.get("topic"), state.get("literature_cache", [])),
            estimated_cost=0.001,
        )
        refined = normalize_atomic_question(response["structured"].get("question"))
        if refined.lower() == question.lower():
            return False
        state.setdefault("candidate_question_history", []).append({
            "candidate_questions": deepcopy(state.get("candidate_questions", [])),
            "trigger": "novelty_well_covered", "novelty_challenge": deepcopy(state.get("novelty_challenge")),
            "created_at": now_iso(),
        })
        state["gap_grounded_candidate_regeneration_count"] = 1
        state["candidate_questions"][0]["question"] = refined
        state["candidate_question_validation"] = candidate_question_semantic_validation(refined, state.get("topic"), state.get("literature_cache", []))
        return True

    def _atomic_refinement_prompt(self, objective, question, dimension):
        rules = {
            "falsifiability": [
                "Ignore whether we currently possess enough literature.",
                "Do not answer the research question.",
                "Ask only whether conceivable measurements or observations could count against the proposed relationship.",
            ],
            "feasibility": [
                "Consider data, tools, computation, experiments, and local/free execution constraints.",
                "Do not assess novelty or answer the research question.",
            ],
        }[dimension]
        return "\n".join([
            "OBJECTIVE",
            objective,
            "QUESTION",
            question,
            "RULES",
            *rules,
            "EXPECTED",
            'Return {"score":0..1,"reason":"..."}.',
        ])

    def _question_refinement_limitations(self, state):
        limitations = []
        records = state.get("literature_cache", [])
        if records:
            limitations.append(f"Only {len(records)} retrieved metadata/abstract records have been checked at refinement time.")
        if any(not record.get("abstract") for record in records):
            limitations.append("Some retrieved records lack abstracts or full text in the current artifact set.")
        limitations.append("No primary measurements or executable experiments have been run yet.")
        return limitations

    def _node_feasibility_analysis(self, state, node):
        if local_guidance_high():
            return self._guided_feasibility_analysis(state, node)
        prompt = {
            "task": "Create a generic research specification and evidence contracts for the selected question. Return JSON only.",
            "selected_question": state.get("selected_question"),
            "literature": minimal_literature_context(state.get("literature_cache", [])),
            "allowed_methodologies": [
                "experiment",
                "simulation",
                "formal reasoning",
                "dataset analysis",
                "comparative study",
                "systematic literature analysis",
                "solver/tool execution",
            ],
            "constraints": [
                "Choose a methodology that can execute locally with current evidence.",
                "Do not require domain-specific built-ins.",
                "Do not include results before execution.",
            ],
            "required_schema": {
                "research_question": "string",
                "methodology": "string",
                "feasibility_verdict": "FEASIBLE|PARTIAL|BLOCKED",
                "evidence_requirements": [],
                "resource_constraints": [],
                "validation_plan": [],
                "hypotheses": [],
                "falsification_criteria": [],
                "required_claims": [],
                "completion_contracts": [],
                "replication_tolerance": "object",
            },
        }
        response = self.gateway.generate_structured(
            state,
            LLMRequest(json.dumps(prompt), stage=node["node_id"], requested_model_class="STANDARD", task_class=node["llm_task_class"]),
            required_keys=["research_question", "methodology", "hypotheses", "falsification_criteria", "required_claims", "completion_contracts"],
            schema=FEASIBILITY_SCHEMA,
            semantic_validator=validate_feasibility_semantics,
            estimated_cost=0.006,
        )
        spec = response["structured"]
        state["research_spec"] = spec
        self._persist_node_artifact(state, node, "feasibility.json", spec)

    def _guided_feasibility_analysis(self, state, node):
        question = state.get("selected_question")
        resume = state.pop("feasibility_resume_context", None) or {}
        snapshot = deepcopy(resume.get("input_snapshot")) or self._feasibility_input_snapshot(state)
        snapshot_artifact = self._persist_supporting_artifact(state, node, "feasibility_input_snapshot.json", snapshot)
        state["feasibility_input_snapshot"] = snapshot
        operational = resume.get("operationalization")
        if not operational:
            operational = self._atomic_feasibility_call(state, node, "\n".join([
                "OBJECTIVE", "State one observable test implied by this research question.", "QUESTION", question,
                "RULES", "Do not answer the question.", "Do not claim data exists.", "Do not invent measurements already performed.",
                "Describe what would have to be observed or compared.", "EXPECTED", 'Return {"observable_test":"..."}.',
            ]), FEASIBILITY_OPERATIONALIZATION_SCHEMA, "feasibility_operationalization",
                lambda data: validate_observable_test(data, state.get("topic")))["observable_test"]
        route = deepcopy(resume.get("route"))
        if not route:
            route = self._generate_feasibility_route(state, node, question, operational)
        routes = [self._evaluate_feasibility_route(state, node, question, operational, route, snapshot,
                                                    prefetched_fit=resume.get("scientific_fit"),
                                                    skip_optional=resume.get("skip_optional_requirements", False))]
        if routes[0]["empirical_evidence_path"] == "CONDITIONAL" and not resume.get("skip_alternative_route", False):
            try:
                alternative = self._generate_feasibility_route(state, node, question, operational, excluded={route["approach"]})
                routes.append(self._evaluate_feasibility_route(state, node, question, operational, alternative, snapshot))
            except (StructuredGenerationExhausted, NoEligibleLocalModel) as exc:
                state.setdefault("feasibility_optional_enrichment_failures", []).append({
                    "semantic_task": "feasibility_alternative_route", "failure": str(exc), "created_at": now_iso(),
                })
        rank = {"YES": 3, "CONDITIONAL": 2, "UNKNOWN": 1, "NO": 0}
        fit_rank = {"good": 3, "partial": 2, "uncertain": 1, "poor": 0}
        selected = max(routes, key=lambda item: (fit_rank[item["scientific_fit"]["fit"]], rank[item["empirical_evidence_path"]], -len(item["unresolved_requirements"])))
        for item in routes:
            item["selected"] = item is selected
        matches = selected["resource_matches"]
        unresolved = selected["unresolved_requirements"]
        empirical_path = selected["empirical_evidence_path"]
        fit = selected["scientific_fit"]
        requirements = selected["requirements"]
        route = selected["route"]
        modality_assessment = assess_automation_closure(
            ROUTE_MODALITIES.get(route["approach"], []), self._current_capability_semantics(state),
            state.get("verified_evidence_artifacts", []),
        )
        state["research_modality_plan"] = {
            **modality_assessment, "source": "selected_feasibility_route", "route": route["approach"],
        }
        if fit["fit"] == "poor" or empirical_path == "NO":
            verdict = "BLOCKED"
        elif fit["fit"] in {"uncertain", "partial"} or unresolved or empirical_path in {"CONDITIONAL", "UNKNOWN"}:
            verdict = "PARTIAL"
        else:
            verdict = "FEASIBLE"
        spec = {
            "research_question": question,
            "methodology": route["approach"],
            "feasibility_verdict": verdict,
            "evidence_requirements": [item["requirement"] for item in requirements],
            "resource_constraints": [f"{item['status']}: {item['requirement']}" for item in matches],
            "validation_plan": [f"Execute and validate: {operational}", "Verify all acquired resources before empirical analysis."],
            "hypotheses": [f"The observable relationship can be evaluated by: {operational}"],
            "falsification_criteria": [f"Observed results fail to support or constrain the proposed relationship under: {operational}"],
            "required_claims": ["Any reported relationship must derive from executed, provenance-bearing empirical artifacts."],
            "completion_contracts": ["Required resources are verified before execution.", "Raw outputs and validation provenance are persisted."],
            "external_dependencies": [item["requirement"] for item in unresolved],
            "replication_tolerance": {"policy": "defined after measurement scale and dataset are verified"},
        }
        schema_errors = validate_json_schema_subset(spec, FEASIBILITY_SCHEMA)
        semantic_errors = [] if schema_errors else validate_feasibility_semantics(spec)
        if schema_errors or semantic_errors:
            raise StructuredGenerationExhausted("SEMANTIC_VALIDATION_FAILURE supervisor feasibility assembly invalid", [{
                "stage": node["node_id"], "task_class": "candidate_question_generation", "status": "FAILED",
                "failure_type": "SEMANTIC_VALIDATION_FAILURE", "schema_errors": schema_errors,
                "semantic_errors": semantic_errors, "raw_response": json.dumps(spec), "actual_cost": 0.0,
            }])
        provenance = {
            "research_question": {"origin": "persisted_selected_question"},
            "methodology": {"origin": "local_model", "semantic_task": "feasibility_route_generation"},
            "feasibility_verdict": {"origin": "deterministic_supervisor", "inputs": ["scientific_fit", "resource_matches", "empirical_evidence_path"]},
            "evidence_requirements": {"origin": "deterministic_route_semantics_and_optional_local_model_proposals", "availability_assertions": False},
            "resource_constraints": {"origin": "deterministic_capability_matching"},
            "validation_plan": {"origin": "deterministic_supervisor"},
            "hypotheses": {"origin": "deterministic_supervisor_planning_statement"},
            "falsification_criteria": {"origin": "deterministic_supervisor"},
            "required_claims": {"origin": "deterministic_evidence_policy"},
            "completion_contracts": {"origin": "deterministic_evidence_policy"},
        }
        state["feasibility_operationalization"] = operational
        state["feasibility_routes"] = routes
        state["feasibility_resource_matches"] = matches
        state["feasibility_unresolved_requirements"] = unresolved
        state["empirical_evidence_path"] = empirical_path
        state["feasibility_field_provenance"] = provenance
        state["research_spec"] = spec
        self._persist_node_artifact(state, node, "feasibility.json", {
            **spec, "input_snapshot_artifact": snapshot_artifact["path"], "observable_test": operational,
            "routes": routes, "resource_matches": matches,
            "unresolved_requirements": unresolved, "empirical_evidence_path": empirical_path,
            "field_provenance": provenance,
        })

    def _generate_feasibility_route(self, state, node, question, operational, excluded=None):
        excluded = set(excluded or ())
        prompt = "\n".join([
            "OBJECTIVE", "Propose one credible investigation route for the research question.", "QUESTION", question,
            "OBSERVABLE TEST", operational, "EXCLUDED ROUTE TYPES", ", ".join(sorted(excluded)) or "none",
            "RULES", "Propose a route only; do not claim resources or data exist.", "Do not answer the research question.",
            "Use a materially different route type from every excluded type.",
            "EXPECTED", 'Return {"approach":"secondary_data_analysis|simulation|primary_measurement|controlled_experiment|systematic_evidence_analysis|other","reason":"..."}.',
        ])
        def validator(data):
            errors = validate_feasibility_route(data, state.get("topic"))
            if isinstance(data, dict) and data.get("approach") in excluded:
                errors.append("route type duplicates an already evaluated route")
            return errors
        return self._atomic_feasibility_call(state, node, prompt, FEASIBILITY_ROUTE_SCHEMA, "feasibility_route_generation", validator)

    def _evaluate_feasibility_route(self, state, node, question, operational, route, snapshot, prefetched_fit=None, skip_optional=False):
        inherent = self._route_inherent_requirements(route["approach"])
        additional = []
        prompt = "\n".join([
            "OBJECTIVE", "State one additional resource or capability needed to carry out this research route.",
            "QUESTION", question, "ROUTE", route["approach"], "OBSERVABLE TEST", operational,
            "RULES", "Do not claim the resource currently exists.", "Do not answer the research question.",
            "Do not claim measurements have already occurred.",
            "EXPECTED", 'Return {"requirement_type":"data|measurement|software|compute|external_service|apparatus|method|other","requirement":"..."}.',
        ])
        try:
            if skip_optional:
                raise StopIteration
            proposed = self._atomic_feasibility_call(state, node, prompt, FEASIBILITY_REQUIREMENT_SCHEMA,
                "feasibility_requirement_generation", validate_feasibility_requirement)
            proposed.update({"origin": "local_model_proposal", "importance": "optional"})
            additional.append(proposed)
        except StopIteration:
            pass
        except (StructuredGenerationExhausted, NoEligibleLocalModel) as exc:
            state.setdefault("feasibility_optional_enrichment_failures", []).append({
                "semantic_task": "feasibility_requirement_generation", "route": route["approach"],
                "failure": str(exc), "created_at": now_iso(),
            })
        requirements = inherent + additional
        matches = [self._match_feasibility_requirement(item, snapshot) for item in requirements]
        empirical_path = self._empirical_evidence_path(route["approach"], matches)
        fit = deepcopy(prefetched_fit)
        if not fit:
            fit = self._atomic_feasibility_call(state, node, "\n".join([
            "OBJECTIVE", "Judge whether this proposed route addresses the selected research question.",
            "QUESTION", question, "OBSERVABLE TEST", operational, "ROUTE TYPE", route["approach"],
            "RULES", "Judge scientific fit only.", "Do not answer the research question or invent results.",
            "Do not discuss JSON, schema, or instructions.",
            "EXPECTED", 'Return {"fit":"good|partial|poor|uncertain","reason":"..."}.',
            ]), FEASIBILITY_FIT_SCHEMA, "feasibility_scientific_fit",
                lambda data: validate_feasibility_fit(data, state.get("topic")))
        return {"route": deepcopy(route), "approach": route["approach"], "raw_reason": route.get("reason"),
                "inherent_requirements": inherent, "model_proposed_requirements": additional,
                "requirements": requirements, "resource_matches": matches,
                "unresolved_requirements": [item for item in matches if item["status"] != "AVAILABLE_VERIFIED"],
                "empirical_evidence_path": empirical_path, "scientific_fit": fit, "selected": False}

    def _atomic_feasibility_call(self, state, node, prompt, schema, semantic_task, validator):
        self._record_prompt_telemetry(state, node, len(state.get("feasibility_atomic_steps", [])) + 1, prompt, semantic_task)
        try:
            response = self.gateway.generate_structured(
                state,
                LLMRequest(prompt, stage=node["node_id"], requested_model_class="CHEAP",
                           task_class="candidate_question_generation", semantic_task=semantic_task),
                schema=schema, semantic_validator=validator, estimated_cost=0.001,
            )
        except StructuredGenerationExhausted as exc:
            raise StructuredGenerationExhausted(
                f"ATOMIC_LOCAL_REASONING_EXHAUSTED semantic_task={semantic_task}", exc.attempts
            ) from exc
        state.setdefault("feasibility_atomic_steps", []).append({
            "semantic_task": semantic_task, "structured": deepcopy(response["structured"]),
            "model": response.get("model"), "created_at": now_iso(),
        })
        return response["structured"]

    def _feasibility_input_snapshot(self, state):
        manifest = state.get("artifact_manifest", {}).get("artifacts", [])
        dataset_suffixes = {".csv", ".parquet", ".arrow", ".feather", ".sqlite", ".db"}
        datasets = [entry for entry in manifest if Path(entry.get("path", "")).suffix.lower() in dataset_suffixes]
        skills = [entry.get("skill", entry) for entry in state.get("skill_registry", [])]
        tools = [tool for tool in ("python3", "bash", "git") if shutil.which(tool)]
        missing = []
        if not datasets:
            missing.append("No verified measurement or analysis dataset artifact is currently registered.")
        return {
            "created_at": now_iso(), "selected_question": state.get("selected_question"), "topic": state.get("topic"),
            "relevant_literature_count": int(state.get("literature_relevance", {}).get("relevant_count", len(state.get("literature_cache", [])))),
            "evidence_types_available": ["scholarly_metadata", "abstract_text"] if state.get("literature_cache") else [],
            "actual_datasets": datasets, "registered_tools": tools,
            "registered_capabilities": [{"capability_id": skill.get("capability_id"), "skill_id": skill.get("skill_id"),
                                         "requirement_types": skill.get("requirement_types", [])}
                                        for skill in skills if isinstance(skill, dict)],
            "runtime_facts": {"logical_cores": os.cpu_count(), "local_llm_provider": os.environ.get("RESEARCH_LLM_PROVIDER") == "local"},
            "network_provider_availability": {"literature_provider": getattr(self.literature_provider, "provider_name", "unknown"), "web_search": bool(os.environ.get("RESEARCH_WEB_SEARCH_PROVIDER"))},
            "paid_fallback_allowed": os.environ.get("RESEARCH_ALLOW_PAID_FALLBACK", "0").lower() in {"1", "true", "yes"},
            "known_missing_resources": missing,
            "provenance": {"literature": "state.literature_cache", "datasets": "state.artifact_manifest", "capabilities": "state.skill_registry", "tools": "shutil.which", "constraints": "environment"},
        }

    def _current_capability_semantics(self, state):
        capabilities = deepcopy(state.get("capability_semantics", []))
        if shutil.which("python3"):
            capabilities.append({
                "capability_id": "local_python_execution", "capability_class": "code_execution",
                "status": "AVAILABLE_VERIFIED", "produces_modalities": ["executable_computation"],
                "verification_mechanisms": ["process_exit_status", "artifact_validation"],
                "provenance": "deterministic_runtime_discovery",
            })
        for entry in state.get("skill_registry", []):
            skill = entry.get("skill", entry)
            if skill.get("capability_status") == "AVAILABLE_VERIFIED":
                capabilities.append({
                    "capability_id": skill.get("capability_id"), "status": "AVAILABLE_VERIFIED",
                    "produces_modalities": skill.get("produces_modalities", []),
                    "verification_mechanisms": skill.get("verification_mechanisms", []),
                    "provenance": "verified_skill_registry",
                })
        return capabilities

    def _route_inherent_requirements(self, approach):
        contracts = {
            "primary_measurement": [
                ("method", "A capability to acquire observations for the specified comparison."),
                ("measurement", "An artifact containing the resulting observations for the comparison."),
            ],
            "secondary_data_analysis": [("data", "An acquired and verified dataset suitable for the specified analysis.")],
            "controlled_experiment": [
                ("method", "A capability to execute the specified controlled comparison."),
                ("measurement", "An artifact containing observations produced by the controlled comparison."),
            ],
            "simulation": [
                ("software", "An executable simulation or model capability."),
                ("data", "Verified inputs required to execute the simulation or model."),
            ],
            "systematic_evidence_analysis": [("data", "A verified literature or evidence corpus for systematic analysis.")],
            "other": [("method", "A verified capability to execute the proposed investigation route.")],
        }
        return [{"requirement_type": kind, "requirement": text, "importance": "required",
                 "origin": "deterministic_route_semantics"} for kind, text in contracts.get(approach, contracts["other"])]

    def _match_feasibility_requirement(self, requirement, snapshot):
        kind = requirement["requirement_type"]
        matched_capabilities = [item["capability_id"] for item in snapshot["registered_capabilities"]
                                if item.get("capability_id") and kind in item.get("requirement_types", [])]
        matched_tools = list(snapshot["registered_tools"]) if kind == "compute" else []
        matched_artifacts = [entry.get("path") for entry in snapshot["actual_datasets"]] if kind in {"data", "measurement"} else []
        if matched_capabilities or matched_tools or matched_artifacts:
            status = "AVAILABLE_VERIFIED"
        elif kind == "data" and (snapshot["network_provider_availability"].get("web_search") or any("discover" in str(item).lower() for item in snapshot["registered_capabilities"])):
            status = "DISCOVERABLE"
        elif kind in {"data", "measurement", "method", "apparatus"}:
            status = "MISSING"
        elif kind in {"software", "external_service", "other"}:
            status = "UNKNOWN"
        else:
            status = "MISSING"
        return {
            "requirement": requirement["requirement"], "requirement_type": kind,
            "origin": requirement.get("origin", "local_model_proposal"), "importance": requirement.get("importance", "required"),
            "status": status,
            "matched_capability_ids": matched_capabilities, "matched_tool_ids": matched_tools,
            "matched_artifact_ids": matched_artifacts,
            "evidence": {"snapshot_fields": ["registered_capabilities", "registered_tools", "actual_datasets"],
                         "matching_key": "requirement_type"},
            "availability_origin": "deterministic_supervisor",
        }

    def _empirical_evidence_path(self, approach, matches):
        actual_data = any(item["matched_artifact_ids"] for item in matches)
        if approach == "systematic_evidence_analysis":
            return "NO"
        if approach == "secondary_data_analysis":
            return "YES" if actual_data else "CONDITIONAL"
        if approach in {"primary_measurement", "controlled_experiment", "simulation"}:
            return "CONDITIONAL"
        return "UNKNOWN"

    def _node_capability_gap_analysis(self, state, node):
        unresolved = state.get("feasibility_unresolved_requirements", [])
        base_requirement = capability_requirement(
            "literature_metadata_analysis",
            "Analyze retrieved scholarly metadata and produce reproducible structured evidence for the selected research question.",
            required_inputs=["evidence/discovery.json"],
            required_outputs=["analysis/literature_metrics.json", "analysis/literature_records.csv"],
            validation_criteria=["outputs parse", "record counts match retrieval inventory"],
            required_tools=["python3"], resource_requirements={}, network_requirements="none", risk="low",
            expected_artifacts=["analysis/literature_metrics.json", "analysis/literature_records.csv"],
            produces_modalities=["literature_metadata"], verification_mechanisms=["retrieval_provenance_and_reexecution"],
        )
        base_requirement.update({"requirement_id": "substudy:literature_metadata_inventory",
                                 "objective_relation": "SUPPORTING_EVIDENCE", "evidence_modality": "scholarly_metadata"})
        state["capability_requirements"] = [base_requirement]
        modality_requirements = {
            "executable_computation": capability_requirement(
                "executable_computation_evidence",
                "Execute the persisted bounded experimental objective with correctness checks, raw measurements, validation, and re-execution.",
                required_inputs=["experiments/contract.json"],
                required_outputs=["experiments/contract.json", "experiments/manifest.json",
                                  "experiments/measurements.jsonl", "experiments/correctness.json",
                                  "experiments/summary.json", "experiments/counterexamples.json",
                                  "claims/candidates.json", "adversarial/falsification.json"],
                validation_criteria=["contract validates", "correctness passes before timing acceptance",
                                     "raw measurements independently recompute", "artifact hashes verify"],
                required_tools=["python3"], risk="medium",
                expected_artifacts=["experiments/contract.json", "experiments/manifest.json",
                                    "experiments/measurements.jsonl", "experiments/correctness.json",
                                    "experiments/summary.json", "experiments/counterexamples.json",
                                    "claims/candidates.json", "adversarial/falsification.json"],
                produces_modalities=["executable_computation"],
                verification_mechanisms=["reexecution_and_output_validation"],
                evidence_protocol={
                    "validation": {"command_path": "validator_path", "report_artifact": "validation/report.json"},
                    "replication": {"comparison_report_field": "scientific_signature"},
                    "adversarial": {"report_artifact": "adversarial/falsification.json"},
                    "claims": {"candidates_artifact": "claims/candidates.json"},
                }),
        }
        mandatory_modalities = objective_required_modalities(state)
        for modality in mandatory_modalities:
            requirement = modality_requirements.get(modality)
            if requirement:
                requirement.update({"requirement_id": f"objective:modality:{modality}",
                                    "objective_relation": "DIRECT_REQUIREMENT",
                                    "evidence_modality": modality, "mandatory": True,
                                    "origin": "deterministic_objective_modality_projection"})
                state["capability_requirements"].append(requirement)
        state["research_modality_plan"] = {"required_evidence_modalities": mandatory_modalities,
                                           "origin": "deterministic_objective_modality_projection"}
        if unresolved:
            requirements = []
            for index, item in enumerate(unresolved, 1):
                requirement = capability_requirement(
                    f"resolve_{'_'.join(sorted(lexical_terms(item['requirement']))[:5])}",
                    item["requirement"], required_inputs=[], required_outputs=[f"resources/requirement_{index}.json"],
                    validation_criteria=["resource existence and provenance verified"], required_tools=[],
                    resource_requirements={"feasibility_status": item["status"]},
                    network_requirements="bounded discovery" if item["status"] == "DISCOVERABLE" else "unknown",
                    risk="low", expected_artifacts=[f"resources/requirement_{index}.json"],
                    produces_modalities=REQUIREMENT_TYPE_MODALITIES.get(item.get("requirement_type"), []),
                    verification_mechanisms=["verified_artifact_postcondition"],
                )
                requirement.update({"requirement_id": f"parent_objective:requirement_{index}",
                                    "objective_relation": "DIRECT_REQUIREMENT", "evidence_modality": item.get("requirement_type")})
                requirements.append(requirement)
            state["capability_requirements"].extend(requirements)
        self._persist_node_artifact(state, node, "capabilities/requirements.json", {"requirements": state["capability_requirements"]})

    def _node_skill_discovery_creation(self, state, node):
        registry = SkillRegistry(self.store.run_root(state["run_id"]) / "skills")
        manager = SkillManager(registry)
        work_dir = self.work_root / state["run_id"] / node["node_id"]
        work_dir.mkdir(parents=True, exist_ok=True)
        generated = []
        for req in state.get("capability_requirements", []):
            skill, result = manager.resolve(state, req, str(work_dir), max_repairs=1)
            if not skill:
                block_node(state, node["node_id"], "WAITING_FOR_HUMAN", "skill requires human approval")
                return
            generated.append({"skill": skill, "validation": result})
        state["skill_registry"] = generated
        self._persist_node_artifact(state, node, "skills/registry.json", {"skills": generated})

    def _node_executable_artifact_dag(self, state, node):
        tasks = []
        for requirement in state.get("capability_requirements", []):
            tasks.append({
                "task_id": f"execute_{requirement['capability_id']}",
                "capability_id": requirement["capability_id"],
                "inputs": requirement.get("required_inputs", []),
                "outputs": requirement.get("required_outputs", []),
                "producer": "dynamic_skill",
                "objective_relation": requirement.get("objective_relation"),
                "evidence_modality": requirement.get("evidence_modality"),
                "evidence_contract_ids": [requirement.get("requirement_id")],
                "permitted_claim_relations": (["DIRECT_ANSWER", "DIRECT_CONSTRAINT"]
                    if requirement.get("objective_relation") == "DIRECT_REQUIREMENT"
                    else ["PROCESS_OR_METADATA", "SUPPORTING_EVIDENCE"]),
            })
        dag = {"tasks": tasks, "origin": "deterministic_capability_requirement_projection"}
        state["artifact_dag"] = dag
        self._persist_node_artifact(state, node, "artifact_dag.json", dag)

    def _node_research_execution(self, state, node):
        work_dir = self.work_root / state["run_id"] / node["node_id"]
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        discovery = self._read_artifact(state, "evidence/discovery.json")
        write_json(work_dir / "evidence/discovery.json", discovery)
        records = []
        expected = []
        for task in state.get("artifact_dag", {}).get("tasks", []):
            script = self._find_skill_script(state, task["capability_id"])
            record = self._run_script(state, node, script, work_dir, producer=f"dynamic_skill:{task['capability_id']}")
            records.append(record)
            if record["exit_status"] != 0:
                block_node(state, node["node_id"], "FAILED", f"dynamic skill execution failed: {task['capability_id']}")
                return
            expected.extend(task.get("outputs", []))
        write_json(work_dir / "execution/provenance.json", {"execution_records": records, "created_at": now_iso()})
        outputs = []
        for rel in list(dict.fromkeys(["execution/provenance.json"] + expected)):
            path = work_dir / rel
            if not path.exists():
                block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", f"skill did not produce {rel}")
                return
            outputs.append(self.store.put_artifact(state["run_id"], path, rel, "dynamic_skill"))
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].extend(outputs)
        complete_node(state, node["node_id"], outputs)

    def _node_independent_validation(self, state, node):
        executable = executable_evidence_skill(state)
        if executable:
            protocol_errors = validate_executable_evidence_protocol(executable)
            if protocol_errors:
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "; ".join(protocol_errors))
                return
            protocol = executable["evidence_protocol"]["validation"]
            validator = executable.get("implementation", {}).get(protocol["command_path"])
            execution_dir = self.work_root / state["run_id"] / "research_execution"
            if not validator:
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "validation command is not installed")
                return
            result = subprocess.run([validator], cwd=execution_dir, capture_output=True, text=True, timeout=120)
            report_path = execution_dir / protocol["report_artifact"]
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
                "validator": "missing", "status": "FAIL", "errors": [result.stderr or "validator emitted no report"]}
            report["execution"] = {"command": validator, "exit_code": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr}
            state.setdefault("validation_reports", []).append(report)
            if result.returncode or report.get("status") != "PASS":
                record_verification(state, node["node_id"], "VERIFICATION_FAILED", verifier=report.get("validator"), reason="; ".join(report.get("errors", [])))
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "; ".join(report.get("errors", [])))
                return
            requirement = next(item for item in state.get("capability_requirements", [])
                               if item.get("capability_id") == executable.get("capability_id"))
            state.setdefault("requirement_validations", []).append({
                "requirement_id": requirement.get("requirement_id"), "status": "PASS",
                "verified_artifacts": requirement.get("expected_artifacts", []),
                "validator": report.get("validator"), "created_at": now_iso()})
            self._persist_node_artifact(state, node, protocol["report_artifact"], report)
            record_verification(state, node["node_id"], "VERIFIED", [protocol["report_artifact"]], report.get("validator"))
            return
        discovery = self._read_artifact(state, "evidence/discovery.json")
        metrics = self._read_artifact(state, "analysis/literature_metrics.json")
        expected = len(dedupe_records([r for retr in discovery["retrievals"] for r in retr.get("records", [])]))
        errors = []
        if metrics.get("record_count") != expected:
            errors.append(f"record_count mismatch: {metrics.get('record_count')} != {expected}")
        report = {"validator": "deterministic_independent_validator", "status": "PASS" if not errors else "FAIL", "errors": errors, "created_at": now_iso()}
        state.setdefault("validation_reports", []).append(report)
        if errors:
            record_verification(state, node["node_id"], "VERIFICATION_FAILED", verifier=report["validator"], reason="; ".join(errors))
            block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "; ".join(errors))
            return
        self._persist_node_artifact(state, node, "validation/report.json", report)
        record_verification(state, node["node_id"], "VERIFIED", ["validation/report.json"], report["validator"])

    def _node_adversarial_falsification(self, state, node):
        findings = []
        executable = executable_evidence_skill(state)
        if executable:
            protocol_errors = validate_executable_evidence_protocol(executable)
            if protocol_errors:
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "; ".join(protocol_errors))
                return
            adversarial = self._read_artifact(state, executable["evidence_protocol"]["adversarial"]["report_artifact"])
            if adversarial.get("status") != "PASS" or not adversarial.get("attempted_falsifications"):
                findings.append({"role": "capability_falsification_gate", "severity": "fatal",
                                 "summary": "executable capability did not pass its declared adversarial checks", "blocks_paper": True})
            findings.extend(adversarial.get("findings", []))
            findings.extend(deterministic_adversarial_review(state.get("claim_evidence_ledger", {}).get("claims", []), self.store.load_manifest(state["run_id"])))
            state["adversarial_findings"] = findings
            if any(item.get("severity") == "fatal" for item in findings):
                state["unresolved_findings"] = findings
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "fatal adversarial findings")
                return
            self._persist_node_artifact(state, node, "adversarial/findings.json", {"findings": findings,
                "attempted_falsifications": adversarial["attempted_falsifications"]})
            return
        metrics = self._read_artifact(state, "analysis/literature_metrics.json")
        if metrics.get("record_count", 0) < int(os.environ.get("RESEARCH_MIN_LITERATURE_RECORDS", "3")):
            findings.append({
                "role": "methodology_critic",
                "severity": "fatal",
                "summary": "too few retrieved literature records to support the selected evidence contract",
                "blocks_paper": True,
                "suggested_experiment": "broaden search queries and rerun evidence discovery",
            })
        findings.extend(deterministic_adversarial_review(state.get("claim_evidence_ledger", {}).get("claims", []), self.store.load_manifest(state["run_id"])))
        state["adversarial_findings"] = findings
        if any(f.get("severity") == "fatal" for f in findings):
            state["unresolved_findings"] = findings
            block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "fatal adversarial findings")
            return
        self._persist_node_artifact(state, node, "adversarial/findings.json", {"findings": findings})

    def _node_replication(self, state, node):
        executable = executable_evidence_skill(state)
        if executable:
            protocol_errors = validate_executable_evidence_protocol(executable)
            if protocol_errors:
                block_node(state, node["node_id"], "BLOCKED_REPLICATION_FAILURE", "; ".join(protocol_errors))
                return
            work_dir = self.work_root / state["run_id"] / node["node_id"]
            if work_dir.exists(): shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            script = executable["implementation"]["script_path"]
            record = self._run_script(state, node, script, work_dir, producer="independent_replicator")
            validation_protocol = executable["evidence_protocol"]["validation"]
            replication_protocol = executable["evidence_protocol"]["replication"]
            validator = executable["implementation"].get(validation_protocol["command_path"])
            checked = subprocess.run([validator], cwd=work_dir, capture_output=True, text=True, timeout=120)
            report_path = validation_protocol["report_artifact"]
            original = self._read_artifact(state, report_path)
            validation = json.loads((work_dir / report_path).read_text())
            comparison_field = replication_protocol["comparison_report_field"]
            matched = original.get(comparison_field) == validation.get(comparison_field) and original.get(comparison_field) is not None
            verdict = "PASS" if record["exit_status"] == checked.returncode == 0 and validation.get("status") == "PASS" and matched else "FAIL"
            report = {"replicator": "fresh_workspace_reexecution_from_persisted_contract", "verdict": verdict,
                      "criteria_predeclared_in_capability_contract": True,
                      "comparison_report_field": comparison_field, "comparison_matched": matched,
                      "original_validator_report": original,
                      "validator_report": validation, "created_at": now_iso()}
            write_json(work_dir / "replication/report.json", report)
            outputs = []
            replicated_artifacts = replication_protocol.get("artifacts", executable.get("outputs", []))
            for rel in list(dict.fromkeys(["replication/report.json"] + replicated_artifacts + [report_path])):
                target = f"replication/{rel}" if not rel.startswith("replication/") else rel
                outputs.append(self.store.put_artifact(state["run_id"], work_dir / rel, target, "independent_replicator"))
            state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].extend(outputs)
            state.setdefault("replication_reports", []).append(report)
            state["replication_status"] = "PASSED" if verdict == "PASS" else "FAILED"
            if verdict != "PASS":
                block_node(state, node["node_id"], "BLOCKED_REPLICATION_FAILURE", "replication criteria failed")
                return
            complete_node(state, node["node_id"], outputs)
            return
        original = self._read_artifact(state, "analysis/literature_metrics.json")
        work_dir = self.work_root / state["run_id"] / node["node_id"]
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        discovery = self._read_artifact(state, "evidence/discovery.json")
        write_json(work_dir / "evidence/discovery.json", discovery)
        script = self._find_skill_script(state, "literature_metadata_analysis")
        record = self._run_script(state, node, script, work_dir, producer="replicator")
        replicated = json.loads((work_dir / "analysis/literature_metrics.json").read_text(encoding="utf-8")) if (work_dir / "analysis/literature_metrics.json").exists() else {}
        verdict = "PASS" if replicated == original and record["exit_status"] == 0 else "FAIL"
        report = {"replicator": "fresh_workspace_reexecution", "verdict": verdict, "original": original, "replicated": replicated, "created_at": now_iso()}
        write_json(work_dir / "replication/report.json", report)
        artifact = self.store.put_artifact(state["run_id"], work_dir / "replication/report.json", "replication/report.json", "replicator")
        state.setdefault("replication_reports", []).append(report)
        state["replication_status"] = "PASSED" if verdict == "PASS" else "FAILED"
        if verdict != "PASS":
            block_node(state, node["node_id"], "BLOCKED_REPLICATION_FAILURE", "replication output differed")
            return
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(artifact)
        complete_node(state, node["node_id"], [artifact])

    def _node_claim_adjudication(self, state, node):
        executable = executable_evidence_skill(state)
        if executable:
            protocol_errors = validate_executable_evidence_protocol(executable)
            if protocol_errors:
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "; ".join(protocol_errors))
                return
            candidate_path = executable["evidence_protocol"]["claims"]["candidates_artifact"]
            payload = self._read_artifact(state, candidate_path)
            available = {entry.get("path") for entry in self.store.load_manifest(state["run_id"]).get("artifacts", [])}
            adjudication = adjudicate_executable_claim_candidates(
                payload, executable, available,
                bool(state.get("validation_reports") and state["validation_reports"][-1].get("status") == "PASS"),
                state.get("replication_status") == "PASSED")
            if not adjudication["valid"]:
                block_node(state, node["node_id"], "BLOCKED_INVALID_METHOD", "; ".join(adjudication["errors"]))
                return
            claims = adjudication["claims"]
            contracts = adjudication.get("experiment_contracts", [])
            if contracts:
                state["computational_experiment_contracts"] = deepcopy(contracts)
            state["claim_evidence_ledger"] = {"claims": claims}
            self._persist_node_artifact(state, node, "claims/adjudication.json", {"claims": claims})
            return
        metrics = self._read_artifact(state, "analysis/literature_metrics.json")
        claims = [
            {
                "claim_id": "C001",
                "claim": f"The retrieved and deduplicated literature inventory contains {metrics['record_count']} verified metadata records relevant to the selected research question.",
                "status": "VERIFIED_TOOL_OUTPUT",
                "origin": "deterministic_adjudicator",
                "producer": "dynamic_skill",
                "validated_by": ["deterministic_independent_validator"],
                "artifacts": ["analysis/literature_metrics.json", "analysis/literature_records.csv"],
                "validator_artifacts": ["validation/report.json"],
                "counterevidence": [],
                "assumptions": ["Public metadata retrieval is sufficient for this small evidence contract."],
                "limitations": [lim for r in state.get("literature_cache", []) for lim in r.get("limitations", [])][:10],
                "replication_status": state.get("replication_status", "NOT_ATTEMPTED"),
                "allowed_paper_language": "The retrieved metadata inventory contains reproducible records; broader scholarly conclusions require additional evidence.",
                "paper_role": "supporting",
                "objective_relation": "PROCESS_OR_METADATA",
                "satisfies_requirement_ids": ["substudy:literature_metadata_inventory"],
                "evidence_modality": "scholarly_metadata",
            }
        ]
        state["claim_evidence_ledger"] = {"claims": claims}
        self._persist_node_artifact(state, node, "claims/adjudication.json", {"claims": claims})

    def _node_research_readiness(self, state, node):
        manifest = self.store.load_manifest(state["run_id"])
        coverage = selected_objective_coverage(state, manifest)
        state["selected_objective_coverage"] = coverage
        state["requirement_lifecycle"] = coverage["requirement_lifecycle"]
        report = {"ready": coverage["status"] == "SUFFICIENT",
                  "errors": coverage["reasons"], "selected_objective_coverage": coverage,
                  "contracts": state.get("research_spec", {}).get("completion_contracts", []), "created_at": now_iso()}
        self._persist_node_artifact(state, node, "readiness/report.json", report)
        if not report["ready"]:
            state["status"] = "PARTIAL_RESEARCH"
            return
        package, package_report = write_research_package(self.store, state)
        if not package:
            block_node(state, node["node_id"], "BLOCKED_MISSING_EVIDENCE", "; ".join(package_report.get("reasons", [])))

    def _persist_node_artifact(self, state, node, rel_path, data):
        run_dir = self.work_root / state["run_id"] / node["node_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / rel_path
        write_json(path, data)
        try:
            entry = self.store.put_artifact(state["run_id"], path, rel_path, node["node_id"])
        except ValueError as exc:
            if "immutable artifact already exists with different checksum" not in str(exc):
                raise
            versioned_rel_path = f"{Path(rel_path).parent}/{Path(rel_path).stem}-{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}{Path(rel_path).suffix}"
            versioned_rel_path = str(Path(versioned_rel_path))
            entry = self.store.put_artifact(state["run_id"], path, versioned_rel_path, node["node_id"])
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(entry)
        complete_node(state, node["node_id"], [entry])

    def _persist_supporting_artifact(self, state, node, rel_path, data):
        run_dir = self.work_root / state["run_id"] / node["node_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / rel_path
        write_json(path, data)
        try:
            entry = self.store.put_artifact(state["run_id"], path, rel_path, node["node_id"])
        except ValueError as exc:
            if "immutable artifact already exists with different checksum" not in str(exc):
                raise
            versioned = f"{Path(rel_path).stem}-{now_iso().replace(':', '').replace('.', '')}-{uuid4().hex[:8]}{Path(rel_path).suffix}"
            entry = self.store.put_artifact(state["run_id"], path, versioned, node["node_id"])
        state.setdefault("artifact_manifest", {"artifacts": []})["artifacts"].append(entry)
        return entry

    def _read_artifact(self, state, rel_path):
        path = Path(self.store.get_artifact_path(state["run_id"], rel_path))
        return json.loads(path.read_text(encoding="utf-8"))

    def _find_skill_script(self, state, capability_id):
        for entry in state.get("skill_registry", []):
            skill = entry.get("skill", {})
            if skill.get("capability_id") == capability_id:
                return skill.get("implementation", {}).get("script_path")
        raise FileNotFoundError(f"no skill for capability {capability_id}")

    def _run_script(self, state, node, script, work_dir, producer):
        import subprocess

        started = now_iso()
        result = subprocess.run([script], cwd=work_dir, capture_output=True, text=True, timeout=120)
        ended = now_iso()
        log_path = work_dir / "execution.log"
        log_path.write_text(f"returncode={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}", encoding="utf-8")
        log_entry = self.store.put_artifact(state["run_id"], log_path, f"logs/{node['node_id']}_{producer}.log", producer, immutable=False)
        record = {
            "node_id": node["node_id"],
            "producer": producer,
            "command": script,
            "working_directory": str(work_dir),
            "start_time": started,
            "end_time": ended,
            "exit_status": result.returncode,
            "stdout_artifact": log_entry["path"],
            "environment": {"python": os.sys.version.split()[0], "platform": os.name},
        }
        state.setdefault("execution_records", []).append(record)
        return record
