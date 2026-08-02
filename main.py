import argparse
import os
import re
import json
import sys
import shutil
import traceback
import signal
from string import Template
from datetime import datetime
from src.agent import ResearchAgent, CodeExecutor
from src.ledger import EvidenceLedger
from src.adversarial import AdversarialBoard
from src.config import *
from src.rag import ProjectRAG, summarize_project

def prompt_from_template(template_str, mapping):
    return Template(template_str).safe_substitute(mapping)

class ResearchOrchestrator:
    def __init__(self, project_dir=None):
        self.state = {"steps": [], "idx": 0, "context": "", "topic": "", "proposal": "", "status": "PLANNED"}
        self.project_dir = project_dir
        if self.project_dir:
            self.state["project_dir"] = self.project_dir
        self.planner = ResearchAgent("You are a research planner. Output JSON workflows.", model_queue=SMART_QUEUE)
        self.coder = ResearchAgent("You are a Python coder. Output runnable code only. For every expected artifact, you MUST implement Python code to write it to disk using 'with open(filename, \"w\") as f: f.write(content)'. Do not omit this.", model_queue=FAST_QUEUE)
        self.adversary_board = AdversarialBoard()
        self.skills = {}
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)

    def _handle_stop_signal(self, signum, frame):
        print(f"Stop signal received ({signum}).")
        self.stop_requested = True

    def check_stop(self):
        if self.stop_requested:
            self.save_state()
            sys.exit(0)

    def load_skills(self):
        self.skills = {}
        if not os.path.exists("skills"): os.makedirs("skills")
        for filename in os.listdir("skills"):
            if filename.endswith(".md"):
                with open(os.path.join("skills", filename), "r") as f:
                    self.skills[filename] = f.read()
        return "\n".join([f"Skill: {f}\nContent: {c}\n" for f, c in self.skills.items()])

    def save_state(self):
        self.state["project_dir"] = self.project_dir
        with open(os.path.join(self.project_dir, "state.json"), "w") as f:
            json.dump(self.state, f)

    def load_state(self, project_dir):
        self.state = {"steps": [], "idx": 0, "context": "", "topic": "", "proposal": ""}
        state_path = os.path.join(project_dir, "state.json")
        with open(state_path, "r") as f:
            self.state.update(json.load(f))
        self.project_dir = project_dir
        self.load_skills()

    def _notify_completion(self):
        print(f"Research run '{self.state.get('topic')}' complete.")

    def draft_paper(self):
        print("Orchestrator: Implementing structured LaTeX drafting...")
        ledger = EvidenceLedger(self.project_dir)
        if not ledger.is_paper_ready():
            print("CRITICAL: Research not ready for paper drafting. Unverified claims exist.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        draft_dir = os.path.join(self.project_dir, f"paper_draft_{timestamp}")
        sections_dir = os.path.join(draft_dir, "sections")
        os.makedirs(sections_dir, exist_ok=True)
        
        evidence_ledger = json.dumps(ledger.data)
        
        style_guide_json = self.coder.chat(prompt_from_template(STYLE_GUIDE_PROMPT, {"topic": self.state['topic']}))
        style_data = json.loads(re.sub(r'```json|```', '', style_guide_json).strip())
        
        main_tex = self.coder.chat(prompt_from_template(MAIN_TEX_PROMPT, {"topic": self.state['topic'], "sections": str(style_data['sections']), "latex_class": style_data['latex_class']}))
        with open(os.path.join(draft_dir, "main.tex"), "w") as f: f.write(main_tex)
            
        safe_context = self.state['context'].replace("{", "{{").replace("}", "}}")
        for section in style_data['sections']:
            section_title = section.replace("_", " ").title()
    def run(self, field=None, resume=False):
        if not resume:
            short_topic = re.sub(r'[^a-zA-Z0-9]', '_', field)[:50]
            self.project_dir = os.path.join("results", f"multi_agent_{short_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(self.project_dir, exist_ok=True)
            self.state["project_dir"] = self.project_dir
            self.state["topic"] = field
            
            workflow_json = self.planner.chat(prompt_from_template(PLANNING_AND_CRITIQUE_PROMPT, {"topic": field, "skills_context": self.load_skills()}))
            workflow_data = json.loads(re.sub(r'```json|```', '', workflow_json).strip())
            
            if isinstance(workflow_data, dict):
                self.state["steps"] = workflow_data.get("steps", workflow_data)
            elif isinstance(workflow_data, list):
                self.state["steps"] = workflow_data
            else:
                self.state["steps"] = []
            
            self.state["proposal"] = self.planner.chat(prompt_from_template(PROPOSAL_GENERATION_PROMPT, {"topic": field}))
            self.state["status"] = "IMPLEMENTING"
            self.save_state()

        while self.state["idx"] < len(self.state["steps"]):
            self.check_stop()
            step = self.state["steps"][self.state["idx"]]
            print(f"Executing step {self.state['idx']}: {step['step']}")
            
            # PHASE 1: Implementation - Artifact by Artifact
            actual_artifacts = []
            for artifact_path in step.get("expected_artifacts", []):
                print(f"  Generating artifact: {artifact_path}")
                for attempt in range(3):
                    content = self.coder.chat(f"Task: {step['description']}. Goal: {step['goal']}.\nInstruction: {step.get('implementation_instruction', '')}\nProduce ONLY content for: {artifact_path}")
                    
                    full_path = os.path.join(self.project_dir, artifact_path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w") as f: f.write(content)
                    
                    reviews = self.adversary_board.review_claim(f"{step['step']} - {artifact_path}", "Artifact generated.", [artifact_path], self.state["idx"])
                    if all(r['action'] == "ADVANCE" for r in reviews):
                        actual_artifacts.append(artifact_path)
                        break
                    else:
                        print(f"  Attempt {attempt+1} rejected. Review: {reviews}")
                else:
                    self.state["status"] = "BLOCKED_RETRY_LIMIT_EXCEEDED"
                    self.save_state()
                    break
            else:
                # Only if PHASE 1 succeeded for all artifacts
                # PHASE 2: Adversarial Verification (Step-wide)
                logs = "All artifacts generated and verified."
                reviews = self.adversary_board.review_claim(step['step'], logs, actual_artifacts, self.state["idx"])
                if any(r['action'] == "PIVOT" for r in reviews):
                    self.state["status"] = "BLOCKED_INVALID_METHOD"
                    self.save_state()
                    sys.exit(1)
                
                self.state["idx"] += 1
                self.save_state()
        
        if self.state["idx"] >= len(self.state["steps"]):
            self.state["status"] = "RESEARCH_COMPLETE"
            self.save_state()
            self.draft_paper()
        
        self._notify_completion()
                    self.state["status"] = "BLOCKED_RETRY_LIMIT_EXCEEDED"
                    self.save_state()
                    break
                # Possibly navigate to specific node if requested
                requested_next = [r.get('next_step_index', self.state['idx']) for r in reviews if 'next_step_index' in r]
                if requested_next:
                    next_idx = requested_next[0]
            
            self.state["idx"] = next_idx
            self.save_state()
        
        if self.state["idx"] >= len(self.state["steps"]):
            self.state["status"] = "RESEARCH_COMPLETE"
            self.save_state()
            self.draft_paper()
        
        self._notify_completion()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", nargs='+')
    parser.add_argument("--resume")
    args = parser.parse_args()
    
    orch = ResearchOrchestrator()
    if args.resume:
        orch.load_state(args.resume)
        orch.run(resume=True)
    elif args.field:
        orch.run(field=" ".join(args.field), resume=False)
    else:
        print("Error: Provide --field or --resume")

if __name__ == "__main__":
    main()
