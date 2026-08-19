import re


VARIABLE_CONTROL_TYPES = {"DIRECT_INPUT", "ALGORITHM_PARAMETER", "DERIVED_FROM_INPUT", "UNCONTROLLED", "UNCERTAIN"}
MEASUREMENT_KINDS = {"runtime", "operation_count", "memory", "correctness", "output_value", "other"}
MEASUREMENT_NEUTRALITY = {"NEUTRAL_MEASUREMENT", "RESULT_EMBEDDED", "NOT_MEASURABLE", "UNCERTAIN"}
SEMANTIC_VALUE_VALIDATOR_VERSION = "semantic-value-v2"

STANDARD_MEASUREMENTS = {
    "runtime": ("execution time", "execution_time"),
    "operation_count": ("operation count", "operation_count"),
    "memory": ("memory usage", "memory_usage"),
    "correctness": ("correctness", "correctness"),
}

MEASUREMENT_KIND_METADATA = {
    "runtime": {"semantic_requirement": "SELF_DESCRIBING", "required_sources": ["EXECUTION_METRIC"]},
    "operation_count": {"semantic_requirement": "SELF_DESCRIBING", "required_sources": ["EXECUTION_METRIC"]},
    "memory": {"semantic_requirement": "SELF_DESCRIBING", "required_sources": ["EXECUTION_METRIC"]},
    "correctness": {"semantic_requirement": "REQUIRES_EXPERIMENT_CRITERION", "required_sources": ["VALIDATION_DERIVED"], "downstream_requirement": "correctness_oracle"},
    "output_value": {"semantic_requirement": "REQUIRES_OBSERVABLE_DEFINITION", "required_sources": ["ALGORITHM_OUTPUT"]},
    "other": {"semantic_requirement": "REQUIRES_OBSERVABLE_DEFINITION", "required_sources": ["EXECUTION_METRIC", "ALGORITHM_OUTPUT", "VALIDATION_DERIVED", "EXTERNAL_OBSERVATION", "OTHER"]},
    "uncertain": {"semantic_requirement": "UNRESOLVED"},
}

STANDARD_MEASUREMENT_SOURCES = {"runtime": "EXECUTION_METRIC", "operation_count": "EXECUTION_METRIC",
                                "memory": "EXECUTION_METRIC", "correctness": "VALIDATION_DERIVED"}


def canonical_semantic_id(display_text):
    words = re.findall(r"[a-z0-9]+", str(display_text or "").lower())
    return "_".join(words)


def normalize_semantic_value(raw_value):
    raw = str(raw_value or "")
    text = raw.strip()
    strategies = []
    unquoted = re.sub(r"^[`'‘’“”\"]+|[`'‘’“”\",]+$", "", text).strip()
    if unquoted != text:
        strategies.append("STRIP_SURROUNDING_QUOTES")
    text = unquoted
    if "_" in text or "-" in text:
        text = re.sub(r"[_-]+", " ", text)
        strategies.append("IDENTIFIER_TO_DISPLAY_TEXT")
    camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    if camel != text:
        strategies.append("CAMEL_CASE_TO_DISPLAY_TEXT")
        camel = camel.lower()
    text = camel
    if text and text == text.upper() and any(c.isalpha() for c in text):
        text = text.lower()
        strategies.append("ALL_CAPS_TO_DISPLAY_TEXT")
    normalized = re.sub(r"\s+", " ", text).strip()
    if normalized != text:
        strategies.append("COLLAPSE_WHITESPACE")
    return {"raw_model_value": raw, "normalized_display_text": normalized,
            "canonical_id": canonical_semantic_id(normalized),
            "normalization_strategy": strategies or ["NONE"]}


def semantic_field_role(raw_or_display):
    value = normalize_semantic_value(raw_or_display)
    text = value["normalized_display_text"].lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    placeholders = {"", "...", "???", "tbd", "todo", "variable", "parameter", "value", "string", "num", "count"}
    if text in placeholders or not any(c.isalpha() for c in text):
        return "NONSEMANTIC"
    measurement_aliases = {
        "runtime": {"runtime", "time", "duration", "latency"},
        "operation_count": {"operation", "operations", "instruction", "instructions", "steps"},
        "memory": {"memory", "space", "bytes", "storage"},
        "correctness": {"correctness", "accuracy", "error", "errors", "success"},}
    if any(tokens & aliases for aliases in measurement_aliases.values()):
        return "OBSERVABLE_OR_MEASUREMENT"
    if any(marker in text for marker in ("increase in", "decrease in", "improved", "reduced", "faster", "slower")):
        return "RESULT_OR_EFFECT"
    control_markers = {"input", "inputs", "limit", "size", "length", "number", "quantity", "configuration", "seed", "bound", "bounded", "threshold", "parameter"}
    if tokens & control_markers and len(tokens) >= 2:
        return "CONTROL_CANDIDATE"
    return "UNCERTAIN"


def semantic_value_record(raw_value, field_role="independent_variable"):
    result = normalize_semantic_value(raw_value)
    role = semantic_field_role(result["normalized_display_text"])
    result.update({"field_role": field_role, "semantic_role_classification": role,
                   "representation_classification": "PLACEHOLDER_OR_EMPTY" if role == "NONSEMANTIC"
                       else ("REPRESENTATION_NORMALIZABLE" if result["normalization_strategy"] != ["NONE"] else "NATIVE_DISPLAY"),
                   "semantic_validation_status": "VALID" if role == "CONTROL_CANDIDATE"
                       else ("ROLE_MISMATCH" if role in {"OBSERVABLE_OR_MEASUREMENT", "RESULT_OR_EFFECT"}
                             else "INVALID" if role == "NONSEMANTIC" else "UNCERTAIN"),
                   "validator_version": SEMANTIC_VALUE_VALIDATOR_VERSION})
    return result


def classify_measurement_source(display_text):
    normalized = normalize_semantic_value(display_text)["normalized_display_text"].lower()
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    role = semantic_field_role(normalized)
    if role == "CONTROL_CANDIDATE":
        return "INPUT_PROPERTY"
    if tokens & {"runtime", "time", "duration", "latency", "operation", "operations", "instruction", "instructions", "memory", "bytes", "space"}:
        return "EXECUTION_METRIC"
    if tokens & {"correctness", "accuracy", "error", "errors", "success", "validity"}:
        return "VALIDATION_DERIVED"
    if tokens & {"solution", "answer", "returned", "produced", "cardinality", "objective", "score"}:
        return "ALGORITHM_OUTPUT"
    if role == "NONSEMANTIC":
        return "UNCERTAIN"
    return "TOPIC_OR_DOMAIN_DESCRIPTION" if len(tokens) >= 3 else "UNCERTAIN"


def measurement_source_compatible(kind, source):
    return source in MEASUREMENT_KIND_METADATA.get(kind, {}).get("required_sources", [])


def experiment_relation_coherence(variable, measurement):
    if variable_measurement_coherence(variable, measurement) != "COHERENT":
        return "NO_OBSERVABLE_RELATION"
    source = measurement.get("measurement_source", "UNCERTAIN")
    if not measurement_source_compatible(measurement.get("measurement_kind"), source):
        return "SOURCE_MISMATCH"
    if source in {"INPUT_PROPERTY", "TOPIC_OR_DOMAIN_DESCRIPTION", "UNCERTAIN"}:
        return "NO_OBSERVABLE_RELATION" if source != "UNCERTAIN" else "UNCERTAIN"
    return "EXPERIMENT_RELATION_COHERENT"


def observable_specificity_errors(display_text, kind=None):
    text = normalize_semantic_value(display_text)["normalized_display_text"].lower()
    category_echoes = {"output", "output value", "algorithm output", "result", "value", "measurement", "observable"}
    errors = semantic_display_errors(display_text, "measurement observable")
    if text in category_echoes or (kind and text == kind.replace("_", " ")):
        errors.append("measurement observable repeats a generic category rather than defining a scientific observable")
    return errors


def measurement_from_kind(kind, other_display=None, observable_display=None, measurement_source=None):
    if kind == "uncertain":
        return None
    if kind in STANDARD_MEASUREMENTS:
        display, canonical = STANDARD_MEASUREMENTS[kind]
        raw_model_value = None
        normalization_strategy = ["DETERMINISTIC_STANDARD_MEASUREMENT"]
    elif kind in {"output_value", "other"} and (observable_display or other_display):
        semantic = normalize_semantic_value(observable_display or other_display)
        display = semantic["normalized_display_text"]
        canonical = semantic["canonical_id"]
        raw_model_value = semantic["raw_model_value"]
        normalization_strategy = semantic["normalization_strategy"]
    else:
        return None
    requirement = MEASUREMENT_KIND_METADATA[kind]
    observable = {"display_text": display, "canonical_id": canonical,
                  "specificity": "SELF_DESCRIBING" if requirement["semantic_requirement"] != "REQUIRES_OBSERVABLE_DEFINITION" else "SPECIFIC",
                  "normalization_strategy": normalization_strategy}
    if raw_model_value is not None:
        observable["raw_model_value"] = raw_model_value
    result = {"display_text": display, "canonical_id": canonical, "measurement_kind": kind,
              "measurement_observable": observable, "semantic_requirement": requirement["semantic_requirement"],
              "measurement_source": measurement_source or STANDARD_MEASUREMENT_SOURCES.get(kind)
                  or classify_measurement_source(display),
              "neutrality": "NEUTRAL_MEASUREMENT", "origin": "deterministic_standard_measurement_mapping" if kind in STANDARD_MEASUREMENTS else "atomic_parameterized_observable"}
    if requirement.get("downstream_requirement"):
        result["downstream_requirements"] = [requirement["downstream_requirement"]]
    return result


def control_availability_for_refinement(control_type):
    return "PROPOSED_NOT_VERIFIED" if control_type in {"DIRECT_INPUT", "ALGORITHM_PARAMETER", "DERIVED_FROM_INPUT"} else "NOT_AVAILABLE"


def variable_measurement_coherence(variable, measurement):
    if not variable or not measurement:
        return "UNCERTAIN"
    if variable.get("canonical_id") == measurement.get("canonical_id"):
        return "INCOHERENT"
    return "COHERENT"


def deterministic_bounded_question(variable_display, measurement_display):
    return f"How does varying {variable_display} affect {measurement_display} under a bounded computational input regime?"


def semantic_display_errors(display_text, field):
    normalized = normalize_semantic_value(display_text)
    text = normalized["normalized_display_text"]
    errors = []
    if not text:
        return [f"{field} display text is missing"]
    if text.lower() in {"...", "???", "tbd", "todo", "semantic display text", "readable semantic text", "variable", "parameter", "value", "measurement", "source variable", "string", "num", "count"}:
        errors.append(f"{field} is a schema placeholder rather than a semantic value")
    if not any(char.isalpha() for char in text):
        errors.append(f"{field} must name a semantic quantity")
    return errors


def measurement_direction_errors(measurement):
    text = str(measurement or "").lower()
    directional = (
        "increase in", "decrease in", "improvement", "improved", "reduction", "reduced",
        "faster", "slower", "becomes", "grows", "declines", "higher", "lower",
    )
    return (["dependent measurement embeds an expected result or direction"]
            if any(marker in text for marker in directional) else [])


def validate_variable_contract(data):
    errors = semantic_display_errors(data.get("variable"), "independent variable") if isinstance(data, dict) else ["variable contract missing"]
    if isinstance(data, dict):
        role = semantic_field_role(data.get("variable"))
        if role == "OBSERVABLE_OR_MEASUREMENT":
            errors.append("independent variable has measurement semantics rather than control-candidate semantics")
        elif role == "RESULT_OR_EFFECT":
            errors.append("independent variable embeds a result or effect")
        elif role in {"NONSEMANTIC", "UNCERTAIN"} and not errors:
            errors.append("independent variable semantic role is uncertain")
    control_type = data.get("control_type") if isinstance(data, dict) else None
    if control_type not in VARIABLE_CONTROL_TYPES:
        errors.append("variable control_type is invalid")
    return errors


def validate_measurement_contract(data):
    errors = semantic_display_errors(data.get("measurement"), "dependent measurement") if isinstance(data, dict) else ["measurement contract missing"]
    kind = data.get("measurement_kind") if isinstance(data, dict) else None
    if kind not in MEASUREMENT_KINDS:
        errors.append("measurement_kind is invalid")
    # Only self-describing kinds have a fixed vocabulary. Parameterized kinds
    # are validated through their explicit observable definition instead of
    # requiring the prose to echo the category label.
    kind_terms = {"runtime": ("runtime", "time", "duration"), "operation_count": ("operation", "instruction", "step"),
                  "memory": ("memory", "space", "byte"), "correctness": ("correct", "accuracy", "error", "success")}
    if kind in kind_terms and not any(term in str(data.get("measurement") or "").lower() for term in kind_terms[kind]):
        errors.append("measurement display text is incompatible with measurement_kind")
    errors.extend(measurement_direction_errors(data.get("measurement")))
    return errors


def experimental_skeleton_validation(skeleton):
    errors = []
    variable = skeleton.get("independent_variable", {})
    measurement = skeleton.get("dependent_measurement", {})
    errors.extend(validate_variable_contract({"variable": variable.get("display_text"), "control_type": variable.get("control_type")}))
    errors.extend(validate_measurement_contract({"measurement": measurement.get("display_text"),
                                                 "measurement_kind": measurement.get("measurement_kind")}))
    source = measurement.get("measurement_source")
    if not measurement_source_compatible(measurement.get("measurement_kind"), source):
        errors.append(f"measurement source {source or 'missing'} is incompatible with measurement_kind {measurement.get('measurement_kind')}")
    if variable.get("canonical_id") and variable.get("canonical_id") == measurement.get("canonical_id"):
        errors.append("dependent measurement must be distinct from the independent variable")
    control_type = variable.get("control_type")
    if control_type == "DERIVED_FROM_INPUT":
        source = variable.get("source_variable") or {}
        errors.extend(semantic_display_errors(source.get("display_text"), "derived-variable source"))
        if source.get("control_type") not in {"DIRECT_INPUT", "ALGORITHM_PARAMETER"}:
            errors.append("derived variable requires a directly controllable source variable")
    elif control_type not in {"DIRECT_INPUT", "ALGORITHM_PARAMETER"}:
        errors.append("independent variable is not controllable")
    if measurement.get("neutrality") != "NEUTRAL_MEASUREMENT":
        errors.append("dependent measurement is not confirmed neutral")
    observable = measurement.get("measurement_observable") or {}
    requirement = MEASUREMENT_KIND_METADATA.get(measurement.get("measurement_kind"), {}).get("semantic_requirement")
    if requirement == "REQUIRES_OBSERVABLE_DEFINITION":
        errors.extend(observable_specificity_errors(observable.get("display_text"), measurement.get("measurement_kind")))
    if observable.get("specificity") not in {"SELF_DESCRIBING", "SPECIFIC"}:
        errors.append("measurement observable is not sufficiently specified")
    if measurement.get("informativeness") != "INFORMATIVE":
        errors.append("measurement is not established as experimentally informative")
    relation = skeleton.get("experiment_relation_coherence") or experiment_relation_coherence(variable, measurement)
    if relation != "EXPERIMENT_RELATION_COHERENT":
        errors.append(f"experiment relation is not coherent: {relation}")
    if not skeleton.get("bounded_regime"):
        errors.append("bounded reproducible input regime is missing")
    if skeleton.get("required_modality") != "executable_computation":
        errors.append("required modality is incompatible with computational refinement")
    if skeleton.get("automation_closure") not in {"HIGH", "CONDITIONAL"}:
        errors.append("automation closure is incompatible")
    if skeleton.get("computational_testability") != "TESTABLE":
        errors.append("computational testability is not established")
    return errors


def finalize_skeleton_status(skeleton):
    errors = experimental_skeleton_validation(skeleton)
    uncertain_component = (skeleton.get("computational_testability") == "UNCERTAIN"
                           or skeleton.get("independent_variable", {}).get("control_type") == "UNCERTAIN"
                           or skeleton.get("dependent_measurement", {}).get("neutrality") == "UNCERTAIN")
    skeleton["experimental_skeleton_status"] = "VALID" if not errors else ("UNCERTAIN" if uncertain_component else "INVALID")
    skeleton["validation_errors"] = errors
    return skeleton


def clarification_effective(original_errors, rewritten_skeleton):
    post_errors = experimental_skeleton_validation(rewritten_skeleton)
    return {"effective": not post_errors, "original_rejection_reasons": list(original_errors),
            "post_rewrite_validation": {"status": "VALID" if not post_errors else "INVALID", "errors": post_errors}}
