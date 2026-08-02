from src.agent import ResearchAgent
from src.config import SMART_QUEUE

class AdversarialBoard:
    def __init__(self):
        # Independent reviewers
        self.security_reviewer = ResearchAgent("You are a cryptography/security reviewer. Focus on side-channel leakage, cryptographic hardness, and threat models.", model_queue=SMART_QUEUE)
        self.architecture_reviewer = ResearchAgent("You are a hardware architecture reviewer. Focus on RTL efficiency, area/power/timing metrics, and synthesis reports.", model_queue=SMART_QUEUE)
        self.methodology_reviewer = ResearchAgent("You are an experimental methodology reviewer. Focus on benchmark fairness, statistical significance, and artifact provenance.", model_queue=SMART_QUEUE)
        self.reproducibility_reviewer = ResearchAgent("You are a reproducibility reviewer. Can this research be replicated from the artifacts provided?", model_queue=SMART_QUEUE)

    def review_claim(self, step_name, logs, artifacts):
        # Aggregation logic
        reviewers = [self.security_reviewer, self.architecture_reviewer, self.methodology_reviewer, self.reproducibility_reviewer]
        reports = []
        for r in reviewers:
            reports.append(r.chat(f"Review the results for step '{step_name}'.\n"
                                  f"Raw Logs:\n{logs}\n"
                                  f"Artifacts generated:\n{', '.join(artifacts)}\n"
                                  f"REQUIRED: Validate that the generated artifacts support the claims made in the logs. If no artifacts were generated for a claim, reject the step."))
        return reports
