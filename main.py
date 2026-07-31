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
from src.config import *
from src.rag import ProjectRAG, summarize_project

class ResearchOrchestrator:
    def __init__(self, project_dir=None):
        print("DEBUG: ResearchOrchestrator.__init__ started")
        self.project_dir = project_dir
        self.state = {"steps": [], "idx": 0, "context": "", "topic": "", "proposal": ""}
        print("DEBUG: Initializing agents")
        self.planner = ResearchAgent("You are a research planner. Output JSON workflows.", model_queue=SMART_QUEUE)
        self.coder = ResearchAgent("You are a Python coder. Output runnable code only.", model_queue=FAST_QUEUE)
        self.adversary = ResearchAgent("You are an adversarial reviewer. Critique results and check novelty.", model_queue=SMART_QUEUE)
        print("DEBUG: Agents initialized")
        self.skills = {}
        self.stop_requested = False
        print("DEBUG: ResearchOrchestrator.__init__ finished")
        
        signal.signal(signal.SIGINT, self._handle_stop_signal)
        signal.signal(signal.SIGTERM, self._handle_stop_signal)
        
    def _handle_stop_signal(self, signum, frame):
        print(f"Stop signal received ({signum}). Will checkpoint and stop soon.")
        self.stop_requested = True

    def check_stop(self):
        if self.stop_requested or (self.project_dir and os.path.exists(os.path.join(self.project_dir, ".stop_marker"))):
            print("Cooperative stop requested. Checkpointing...")
            self.save_state()
            print("State saved successfully. Exiting gracefully.")
            sys.exit(0)

    def load_skills(self):
        self.skills = {}
        if not os.path.exists("skills"):
            os.makedirs("skills")
        for filename in os.listdir("skills"):
            if filename.endswith(".md"):
                with open(os.path.join("skills", filename), "r") as f:
                    self.skills[filename] = f.read()
        return "\n".join([f"Skill File: {f}\nContent: {c}\n" for f, c in self.skills.items()])

    def save_state(self):
        self.state["project_dir"] = self.project_dir
        with open(os.path.join(self.project_dir, "state.json"), "w") as f:
            json.dump(self.state, f)

    def load_state(self, project_dir):
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project dir not found: {project_dir}")
        with open(os.path.join(project_dir, "state.json"), "r") as f:
            self.state = json.load(f)
            self.project_dir = project_dir
        self.load_skills()

    def _notify_completion(self):
        # Notify completion via GitHub CLI
        topic = self.state.get("topic", "Unknown Topic")
        project_dir = self.state.get("project_dir", "Unknown Directory")
        issue_num = self.state.get("issue_num")
        
        msg = f"Research run '{topic}' has completed successfully. Artifacts available in: `{project_dir}`."
        
        print(f"Orchestrator: Sending completion notification: {msg}")
        
        # If we have an issue context, post the comment
        if issue_num:
            try:
                # Use the new post-logs script
                subprocess.run(["scripts/post-logs", str(issue_num), msg], check=True)
                print("Orchestrator: Notification posted to issue.")
            except Exception as e:
                print(f"Orchestrator: Notification failed: {e}")
        else:
            print(f"### NOTIFICATION: {msg}")

    def run(self, field=None, resume=False, interactive=False):
        if resume:
            print(f"DEBUG: Resuming project in {self.project_dir}. Current state index: {self.state.get('idx')}")
            print(f"DEBUG: Total steps: {len(self.state.get('steps', []))}")
        else:
            # INTERACTIVE TOPIC SELECTION
            print("\n--- Topic Selection ---")
            print(f"Generating research topics for field: {field}...")
            topic_options_text = self.planner.chat(TOPIC_SELECTION_PROMPT.format(field=field))
            cleaned_json = re.sub(r'```json|```', '', topic_options_text).strip()
            data = json.loads(cleaned_json)
            topics = data.get("research_topics", [])
            print(f"\nGenerated Topic Candidates:")
            for t in topics:
                print(f"{t['number']}. {t['title']}: {t['description']}")
            
            if interactive:
                while True:
                    choice = input("\nSelect a topic number (1-5), or enter your own topic: ")
                    # Parsing 1-5
                    if choice.isdigit() and 1 <= int(choice) <= 5:
                        selected_topic = next((t for t in topics if t['number'] == int(choice)), None)
                        if selected_topic:
                            field = f"{selected_topic['title']}: {selected_topic['description']}"
                            break
                        else:
                            print("Topic number not found. Please try again.")
                    else:
                        field = choice
                        break
            else:
                # Auto-select the first topic in AFK mode
                if topics:
                    field = f"{topics[0]['title']}: {topics[0]['description']}"
                    print(f"\nAFK Mode: Auto-selecting topic: {field}")
                else:
                    print("\nAFK Mode: No topics generated, using original field.")
            
            print(f"Selected topic: {field}")

        # 1. Initialization/Planning
            print("\n--- Stage: Planning ---")
            print("Planner: Designing workflow (with iterative novelty/feasibility assessment)...")
            skills_context = self.load_skills()
            workflow_json = self.planner.chat(PLANNING_AND_CRITIQUE_PROMPT.format(topic=field, skills_context=skills_context))
            cleaned_json = re.sub(r'```json|```', '', workflow_json).strip()
            self.state["steps"] = json.loads(cleaned_json)
            
            print("\n--- Research Workflow Plan ---")
            for i, step in enumerate(self.state["steps"]):
                print(f"{i+1}. {step['step']} (Skill: {step.get('skill')})")
            print("------------------------------\n")
            
            self.state["topic"] = field
            # TRUNCATE TOPIC FOR PATH
            short_topic = re.sub(r'[^a-zA-Z0-9]', '_', field)[:50]
            self.project_dir = os.path.join("results", f"multi_agent_{short_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(self.project_dir, exist_ok=True)
            
            print("\n--- Stage: Proposal Generation ---")
            self.state["proposal"] = self.planner.chat(f"Create proposal for {field}")
            self.save_state()

        # 2. Execution Phase
        print("\n--- Stage: Execution ---")
        while self.state["idx"] < len(self.state["steps"]):
            self.check_stop()
            step = self.state["steps"][self.state["idx"]]
            goal = step.get("goal", "Complete the task.")
            goal_reached = False
            
            print(f"\n--- Starting Step: {step['step']} ---")
            
            while not goal_reached:
                self.check_stop()
                print("Coder: Executing...")
                skill_file = step.get("skill", "code_implementation.md")
                skill_path = os.path.join("skills", skill_file)

                if not os.path.exists(skill_path):
                    print(f"Orchestrator: Skill '{skill_file}' not found. Creating it dynamically...")
                    skill_content = self.planner.chat(f"Create a markdown skill template for: {skill_file}. Goal: {step['description']}")
                    with open(skill_path, "w") as f:
                        f.write(skill_content)
                    self.skills[skill_file] = skill_content
                
                skill_content = self.skills.get(skill_file, "Perform the task.")
                
                code = self.coder.chat(f"{skill_content}\nTask: {step['description']}. Goal: {goal}. Context: {self.state['context']}")
                
                python_block = re.search(r'```(?:python)?(.*?)```', code, re.DOTALL | re.IGNORECASE)
                bash_block = re.search(r'```(?:bash)?(.*?)```', code, re.DOTALL | re.IGNORECASE)
                
                logs = ""
                if python_block or bash_block:
                    if python_block:
                        code_content = python_block.group(1).strip()
                        
                        # Robustness: Check if there's a pip install line in the python block
                        if "pip install" in code_content:
                            lines = code_content.split('\n')
                            bash_cmd = ""
                            python_lines = []
                            for line in lines:
                                if "pip install" in line:
                                    bash_cmd += line.replace("pip install", "").strip() + " "
                                else:
                                    python_lines.append(line)

                            if bash_cmd:
                                print(f"DEBUG: Found pip command in python block, executing as shell: {bash_cmd}")
                                logs += CodeExecutor.execute_shell(f"pip install {bash_cmd}", self.project_dir)
                            code_content = '\n'.join(python_lines)
                        
                        if code_content.strip():
                            code_filename = f"step_{self.state['idx']}_{re.sub(r'[^a-zA-Z0-9]', '_', step['step']).lower()}.py"
                            file_path = os.path.join(self.project_dir, code_filename)
                            with open(file_path, "w") as f:
                                f.write(code_content)
                            
                            logs += CodeExecutor.execute_python(code_content, self.project_dir)
                        
                    elif bash_block:
                        logs += CodeExecutor.execute_shell(bash_block.group(1).strip(), self.project_dir)
                    
                    print(f"Coder Logs: {logs}")
                else:
                    print("Orchestrator: No code blocks found in response, treating as informational context.")
                    self.state["context"] += f"\nStep {step['step']} result: {code}"
                    goal_reached = True
                    continue
                
                # FEEDBACK: Append logs to context for the next iteration to see
                self.state["context"] += f"\nPrevious attempt logs: {logs}"
                
                if "Execution Error" in logs or "Stderr" in logs and "error" in logs.lower():
                    print("Orchestrator: Error detected, asking Adversary for fix...")
                    fix_suggestion = self.adversary.chat(f"The following execution failed with errors: {logs}. Please suggest a fix or explain the issue.")
                    self.state["context"] += f"\nFix suggestion from Adversary: {fix_suggestion}"
                
                critique = self.adversary.chat(f"Analyze logs: {logs}. Has the goal '{goal}' been reached? If yes, say 'GOAL_REACHED'. If no, suggest fixes.")
                self.state["context"] += f"\nCritique: {critique}"
                if "GOAL_REACHED" in critique:
                    goal_reached = True
                    print(f"Adversary: Step '{step['step']}' validated.")
                else:
                    print("Adversary: Step not reached, re-executing...")
            self.state["idx"] += 1
            self.save_state()
        # 3. Finalization & QA
        # After execution loop finishes:
        print("\n--- Research Complete ---")
        self._notify_completion()

        submission_dir = os.path.join(self.project_dir, "submission")
        os.makedirs(submission_dir, exist_ok=True)

        # Generate Paper
        print("\n--- Stage: Formal Paper Drafting (LaTeX) ---")
        try:
            # Sanitize the context to prevent .format() from misinterpreting braces
            safe_context = self.state['context'].replace("{", "{{").replace("}", "}}")
            
            paper_output = self.coder.chat(PAPER_DRAFTING_PROMPT.format(research_context=safe_context))
            
            draft_dir = os.path.join(self.project_dir, "paper_draft")
            os.makedirs(draft_dir, exist_ok=True)
            
            # Parse the output which has FILE: [name] markers
            files = re.split(r'FILE:\s*\[(.*?)\]', paper_output)
            for i in range(1, len(files), 2):
                filename = files[i]
                content = files[i+1].strip()
                with open(os.path.join(draft_dir, filename), "w") as f:
                    f.write(content)
            
            print(f"LaTeX paper draft successfully generated in: {draft_dir}")
        except Exception as e:
            print(f"CRITICAL ERROR in Paper Drafting: {e}")
            traceback.print_exc()
        
        print(f"\nFinal submission package ready in: {submission_dir}")
        
        # Copy artifacts to submission folder
        for f in os.listdir(self.project_dir):
            if f.endswith(('.py', '.md')):

                    shutil.copy2(os.path.join(self.project_dir, f), submission_dir)
        
        # Ensure the generated PDF and LaTeX are moved to the main project directory for top-level access
        
        # QA Loop
        if interactive:
            rag = ProjectRAG(self.project_dir)
            print("You can now ask questions about this research. Type 'exit' to quit.")
            while True:
                question = input("\nQuestion: ")
                if question.lower() == 'exit': break
                print("Answer:", rag.query(self.adversary, question))
        else:
            print("\nNon-interactive mode: Skipping QA loop.")
def main():
    print(f"DEBUG: Starting main.py with args: {sys.argv}")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--field", nargs='+', help="Research field(s)")
        parser.add_argument("--resume", help="Resume from project directory")
        parser.add_argument("--issue", help="Associated Issue Number")
        parser.add_argument("--interactive", action="store_true", help="Enable interactive topic selection")
        args = parser.parse_args()
        print(f"DEBUG: Parsed args: {args}")

        if args.resume:
            print("DEBUG: Instantiating ResearchOrchestrator for resume")
            orch = ResearchOrchestrator()
            print(f"DEBUG: Attempting to load state from: {args.resume}")
            orch.load_state(args.resume)
            print("DEBUG: State loaded, calling orch.run(resume=True)")
            orch.run(resume=True)
        elif args.field:
            field = " ".join(args.field)
            print("DEBUG: Instantiating ResearchOrchestrator for start")
            orch = ResearchOrchestrator()
            print("DEBUG: ResearchOrchestrator instantiated")
            orch.state["issue_num"] = args.issue
            print("DEBUG: Calling orch.run(resume=False)")
            orch.run(field=field, resume=False, interactive=args.interactive)
        else:
            print("Error: Provide --field or --resume")
    except Exception as e:
        print(f"CRITICAL ERROR in main: {e}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
