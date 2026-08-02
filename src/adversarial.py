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
    # Independent reviewers
    reviewers = [self.security_reviewer, self.architecture_reviewer, self.methodology_reviewer, self.reproducibility_reviewer]
    reviews = []
    for r in reviewers:
        review_prompt = (f"Review the results for step '{step_name}'.\n"
                         f"Raw Logs:\n{logs}\n"
                         f"Artifacts generated:\n{', '.join(artifacts)}\n"
                         f"REQUIRED: Evaluate the validity of the step. Based on the artifacts and logs, decide if the step is complete and verified. "
                         f"Return ONLY a raw JSON object with these fields (no markdown, no other text):\n"
                         f"{{\n  \"action\": \"RETRY\" | \"PIVOT\" | \"ADVANCE\",\n  \"reason\": \"...\"\n}}")

        report = r.chat(review_prompt)
        # Parse JSON with robust cleanup
        try:
            # Remove markdown if present, then extract the first { and last }
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
            print(f"DEBUG: Failed to parse reviewer output: {report[:100]}... Error: {e}", flush=True)
            reviews.append({"action": "RETRY", "reason": f"Failed to parse reviewer output: {str(e)}"})

    return reviews


