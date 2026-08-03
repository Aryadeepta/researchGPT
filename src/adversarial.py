from src.agent import ResearchAgent
from src.config import SMART_QUEUE
import json
import re
import os

class AdversarialBoard:
    def __init__(self, topic="general research"):
        # Fully generic, topic-agnostic reviewers
        self.correctness_reviewer = ResearchAgent(f"You are a logical correctness and accuracy reviewer for {topic}. Focus on syntax errors, logical bugs, and semantic correctness.", model_queue=SMART_QUEUE)
        self.completeness_reviewer = ResearchAgent(f"You are a completeness and system-design reviewer for {topic}. Verify if the artifacts meet the specified goals and contain all necessary implementation details.", model_queue=SMART_QUEUE)
        self.methodology_reviewer = ResearchAgent(f"You are an experimental methodology and scientific reviewer for {topic}. Focus on evaluation metrics, benchmark fairness, and statistical validity.", model_queue=SMART_QUEUE)
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
