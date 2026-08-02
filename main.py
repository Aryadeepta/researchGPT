import argparse
import os
import re
import json
import sys
import shutil
import traceback
import signal
from datetime import datetime
from src.agent import ResearchAgent, CodeExecutor
from src.ledger import EvidenceLedger
from src.adversarial import AdversarialBoard
from src.config import *
from src.rag import ProjectRAG, summarize_project

class ResearchOrchestrator:
    def __init__(self, project_dir=None):
        self.state = {"steps": [], "idx": 0, "context": "", "topic": "", "proposal": ""}
        self.project_dir = project_dir
        if self.project_dir:
            self.state["project_dir"] = self.project_dir
        self.planner = ResearchAgent("You are a research planner. Output JSON workflows.", model_queue=SMART_QUEUE)
        self.coder = ResearchAgent("You are a Python coder. Output runnable code only.", model_queue=FAST_QUEUE)
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
        state_path = os.path.join(project_dir, "state.json")
        with open(state_path, "r") as f:
            self.state = json.load(f)
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
        
        style_guide_json = self.coder.chat(STYLE_GUIDE_PROMPT.format(topic=self.state['topic']))
        style_data = json.loads(re.sub(r'```json|```', '', style_guide_json).strip())
        
        main_tex = self.coder.chat(MAIN_TEX_PROMPT.format(topic=self.state['topic'], sections=style_data['sections'], latex_class=style_data['latex_class']))
        with open(os.path.join(draft_dir, "main.tex"), "w") as f: f.write(main_tex)
            
        safe_context = self.state['context'].replace("{", "{{").replace("}", "}}")
        for section in style_data['sections']:
            section_title = section.replace("_", " ").title()
            section_content = self.coder.chat(SECTION_DRAFTING_PROMPT.format(section_name=section, section_name_title=section_title, topic=self.state['topic'], evidence_ledger=evidence_ledger, research_context=safe_context))
            with open(os.path.join(sections_dir, f"{section}.tex"), "w") as f: f.write(section_content)
        
        print(f"LaTeX project generated in {draft_dir}")

    def run(self, field=None, resume=False):
        if not resume:
            short_topic = re.sub(r'[^a-zA-Z0-9]', '_', field)[:50]
            self.project_dir = os.path.join("results", f"multi_agent_{short_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(self.project_dir, exist_ok=True)
            self.state["project_dir"] = self.project_dir
            self.state["topic"] = field
            workflow_json = self.planner.chat(PLANNING_AND_CRITIQUE_PROMPT.format(topic=field, skills_context=self.load_skills()))
            self.state["steps"] = json.loads(re.sub(r'```json|```', '', workflow_json).strip())
            self.state["proposal"] = self.planner.chat(f"Create proposal for {field}")
            self.save_state()

        while self.state["idx"] < len(self.state["steps"]):
            self.check_stop()
            step = self.state["steps"][self.state["idx"]]
            print(f"Executing step {self.state['idx']}: {step['step']}")
            
            # Implementation
            skill_content = self.skills.get(step.get("skill"), "Perform task.")
            code = self.coder.chat(f"{skill_content}\nTask: {step['description']}. Goal: {step['goal']}. Context: {self.state['context']}")
            
            # Artifact execution
            python_block = re.search(r'```(?:python)?(.*?)```', code, re.DOTALL | re.IGNORECASE)
            result = {"stdout": "", "stderr": "", "artifacts": []}
            if python_block:
                code_content = python_block.group(1).strip()
                # Persist the artifact
                code_filename = f"step_{self.state['idx']}_{re.sub(r'[^a-zA-Z0-9]', '_', step['step']).lower()}.py"
                file_path = os.path.join(self.project_dir, code_filename)
                with open(file_path, "w") as f:
                    f.write(code_content)
                print(f"Artifact saved: {file_path}")
                
                result = CodeExecutor.execute_python(code_content, self.project_dir)
            
            logs = f"Stdout: {result['stdout']}\nStderr: {result['stderr']}"
            self.state["context"] += f"\nStep {step['step']} logs: {logs}"
            
            # Adversarial Check with Artifact Gating
            print(f"DEBUG: Running adversarial review for step {step['step']}", flush=True)
            critique = self.adversary_board.review_claim(step['step'], logs, result['artifacts'])
            self.state["context"] += f"\nCritique: {critique}"
            
            # HARD GATE: If any reviewer rejects, block advancement
            if any("REJECT" in r.upper() for r in critique):
                print(f"CRITICAL: Step '{step['step']}' failed adversarial review. Blocking.", flush=True)
                self.state["status"] = "BLOCKED_ADVERSARIAL_FAILURE"
                self.save_state()
                sys.exit(1)
            
            self.state["idx"] += 1
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
