from src.agent import ResearchAgent
from src.config import SMART_QUEUE
import json
import re
import os

class AdversarialBoard:

    def __init__(self, topic="general research"):
        self.security_reviewer = ResearchAgent(f"You are a critical reviewer for {topic}. Focus on security, safety, and robustness.", model_queue=SMART_QUEUE)
        self.architecture_reviewer = ResearchAgent(f"You are an architectural reviewer for {topic}. Focus on efficiency, feasibility, and system design.", model_queue=SMART_QUEUE)
        self.methodology_reviewer = ResearchAgent(f"You are an experimental methodology reviewer for {topic}. Focus on benchmark fairness, statistical significance, and artifact provenance.", model_queue=SMART_QUEUE)
        self.reproducibility_reviewer = ResearchAgent(f"You are a reproducibility reviewer for {topic}. Can this research be replicated from the artifacts provided?", model_queue=SMART_QUEUE)


    def review_claim(self, step_name, logs, artifacts, step_idx, project_dir):
        # Independent reviewers - already generalized in __init__
        reviewers = [self.security_reviewer, self.architecture_reviewer, self.methodology_reviewer, self.reproducibility_reviewer]
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
