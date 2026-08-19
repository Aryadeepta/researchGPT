REQUIRED_FIELDS = {
    "research_question",
    "hypotheses",
    "falsification_criteria",
    "artifact_producing_stages",
    "tool_requirements",
    "implementation_requirements",
    "experiment_requirements",
    "baselines",
    "controls",
    "validation_stages",
    "independent_validators",
    "adversarial_stages",
    "replication",
    "completion_contracts",
    "paper_gate",
    "failure_states",
    "evidence_provenance",
}


def validate_pipeline_meta_schema(pipeline):
    errors = []
    for field in sorted(REQUIRED_FIELDS):
        if field not in pipeline or pipeline[field] in (None, "", [], {}):
            errors.append(f"missing {field}")

    stages = pipeline.get("artifact_producing_stages", [])
    executable = [s for s in stages if s.get("requires_execution") or s.get("produces_raw_artifacts")]
    if not executable:
        errors.append("no executable artifact-producing stage")
    if stages and all(set(s.get("outputs", [])) and all(str(o).endswith((".md", ".txt", ".tex")) for o in s.get("outputs", [])) for s in stages):
        errors.append("primarily prose-producing stages")

    if not pipeline.get("independent_validators"):
        errors.append("no independent validator")
    if not pipeline.get("replication"):
        errors.append("no replication")
    if not pipeline.get("falsification_criteria"):
        errors.append("no falsification criteria")
    if not pipeline.get("evidence_provenance"):
        errors.append("no evidence provenance")
    if not pipeline.get("paper_gate"):
        errors.append("no paper-readiness gate")
    if not pipeline.get("failure_states"):
        errors.append("no failure state")

    for stage in stages:
        for key in ("expected_results", "measured_results", "statistical_significance"):
            if key in stage:
                errors.append(f"stage {stage.get('name', '<unnamed>')} hard-codes {key}")
        validators = stage.get("validators", [])
        if stage.get("producer") and stage.get("producer") in validators:
            errors.append(f"stage {stage.get('name', '<unnamed>')} validates its own evidence")

    for validator in pipeline.get("validation_stages", []):
        if validator.get("producer") and validator.get("producer") == validator.get("validates_producer"):
            errors.append(f"validator {validator.get('name', '<unnamed>')} is not independent")

    return {"valid": not errors, "errors": errors}
