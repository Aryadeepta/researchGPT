import json
import re
import os


ADVERSARIAL_ROLES = {
    "provenance_auditor": "checks artifact origin, checksums, producer/validator independence, and citation provenance",
    "scientific_falsification_reviewer": "looks for decisive falsification tests and unsupported causal claims",
    "methodology_critic": "checks whether methods, comparisons, controls, and conclusions match the research specification",
    "quantitative_evidence_reviewer": "checks whether numeric or structured claims are derived from raw artifacts when such claims exist",
    "literature_novelty_reviewer": "checks whether novelty claims are grounded in retrieved literature",
    "dynamic_domain_reviewer": "created from the research specification when a claim requires project-specific expertise",
    "reproduction_agent": "attempts or specifies independent reproduction from artifacts",
}


def structured_finding(role, severity, claim_id, summary, blocks_paper=False, suggested_experiment=None):
    return {
        "role": role,
        "severity": severity,
        "claim_id": claim_id,
        "summary": summary,
        "blocks_paper": bool(blocks_paper),
        "suggested_experiment": suggested_experiment,
    }


def deterministic_adversarial_review(claims, manifest):
    artifact_paths = {a.get("path") for a in manifest.get("artifacts", [])}
    findings = []
    for claim in claims:
        claim_id = claim.get("claim_id")
        producer = claim.get("producer")
        if producer and producer in claim.get("validated_by", []):
            findings.append(structured_finding("provenance_auditor", "fatal", claim_id, "producer validated its own evidence", True))
        if claim.get("status") == "VERIFIED_MEASUREMENT":
            missing = [a for a in claim.get("artifacts", []) if a not in artifact_paths]
            if missing:
                findings.append(structured_finding("quantitative_evidence_reviewer", "fatal", claim_id, f"measurement lacks raw artifacts: {missing}", True))
        if claim.get("status") == "SUPPORTED_BY_CITATION" and not claim.get("citation_ids"):
            findings.append(structured_finding("literature_novelty_reviewer", "fatal", claim_id, "citation-supported claim has no verified citation ids", True))
    return findings

class AdversarialBoard:
    def __init__(self, topic="general research"):
        from src.agent import ResearchAgent
        from src.config import SMART_QUEUE

        # Fully generic, topic-agnostic reviewers
        self.correctness_reviewer = ResearchAgent(f"You are a logical correctness and accuracy reviewer for {topic}. Focus on syntax errors, logical bugs, and semantic correctness.", model_queue=SMART_QUEUE)
        self.completeness_reviewer = ResearchAgent(f"You are a completeness and system-design reviewer for {topic}. Verify if the artifacts meet the specified goals and contain all necessary implementation details.", model_queue=SMART_QUEUE)
        self.methodology_reviewer = ResearchAgent(f"You are a research methodology reviewer for {topic}. Focus on whether methods, comparisons, controls, and conclusions are justified by the available artifacts.", model_queue=SMART_QUEUE)
        self.reproducibility_reviewer = ResearchAgent(f"You are a reproducibility reviewer for {topic}. Verify if the step can be exactly replicated and if the artifact provenance is valid.", model_queue=SMART_QUEUE)

    def review_claim(self, step_name, logs, artifacts, step_idx, project_dir):
        reviewers = [self.correctness_reviewer, self.completeness_reviewer, self.methodology_reviewer, self.reproducibility_reviewer]
        reviews = []
        
        # Read artifact contents
        artifact_contents = ""
        for art in artifacts:
            path = os.path.join(project_dir, art)
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        artifact_contents += f"\n--- {art} ---\n{f.read()}\n"
                except Exception as e:
                    artifact_contents += f"\n--- {art} (Could not read: {e}) ---\n"
        
        for r in reviewers:
            review_prompt = (f"Review the results for step '{step_name}' (index {step_idx}).\n"
                             f"Raw Logs:\n{logs}\n"
                             f"Artifact Contents:\n{artifact_contents}\n"
                             f"REQUIRED: Evaluate the validity of the step. Based on the artifacts and logs, decide the next action.\n"
                             f"Return ONLY a raw JSON object with these fields:\n"
                             f"{{\n  \"action\": \"RETRY\" | \"PIVOT\" | \"ADVANCE\",\n  \"next_step_index\": int,\n  \"reason\": \"...\"\n}}")
            
            report = r.chat(review_prompt)
            try:
                cleaned = re.sub(r'```(?:json)?', '', report, flags=re.IGNORECASE)
                start = cleaned.find('{')
                end = cleaned.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = cleaned[start:end]
                    review = json.loads(json_str)
                    reviews.append(review)
                else:
                    raise ValueError("No JSON found")
            except Exception as e:
                reviews.append({"action": "RETRY", "next_step_index": step_idx, "reason": f"Parse Error: {str(e)}"})
        return reviews
