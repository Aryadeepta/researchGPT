import argparse
import os
import re
import json
import sys
import traceback
from datetime import datetime
from src.agent import ResearchAgent, CodeExecutor
from src.config import *
from src.rag import ProjectRAG, summarize_project

class ResearchOrchestrator:
    def __init__(self, project_dir=None):
        self.project_dir = project_dir
        self.state = {"steps": [], "idx": 0, "context": "", "topic": "", "proposal": ""}
        self.planner = ResearchAgent("You are a research planner. Output JSON workflows.")
        self.coder = ResearchAgent("You are a Python coder. Output runnable code only.")
        self.adversary = ResearchAgent("You are an adversarial reviewer. Critique results and check novelty.")
        self.skills = {}

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
        with open(os.path.join(self.project_dir, "state.json"), "w") as f:
            json.dump(self.state, f)

    def load_state(self, project_dir):
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project dir not found: {project_dir}")
        with open(os.path.join(project_dir, "state.json"), "r") as f:
            self.state = json.load(f)
            self.project_dir = project_dir
        self.load_skills()

    def run(self, field=None, resume=False, interactive=False):
        if resume:
            print(f"Resuming project in {self.project_dir}...")
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
            step = self.state["steps"][self.state["idx"]]
            goal = step.get("goal", "Complete the task.")
            goal_reached = False
            
            print(f"\n--- Starting Step: {step['step']} ---")
            
            while not goal_reached:
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
        print("\n--- Pipeline complete. Finalizing Submission Package ---")
        submission_dir = os.path.join(self.project_dir, "submission")
        os.makedirs(submission_dir, exist_ok=True)
        
        # Generate LaTeX and compile
        latex_content = self.coder.chat(f"Convert this research context to a full LaTeX paper: {self.state['context']}")
        latex_block = re.search(r'```latex(.*?)```', latex_content, re.DOTALL | re.IGNORECASE)
        
        if latex_block:
            with open(os.path.join(submission_dir, "paper.tex"), "w") as f:
                f.write(latex_block.group(1).strip())
            
            # Compile (requires pdflatex)
            compile_cmd = "pdflatex paper.tex && pdflatex paper.tex"
            logs = CodeExecutor.execute_shell(f"cd {submission_dir} && {compile_cmd}", self.project_dir)
            print(f"LaTeX Compilation: {logs}")
        
        # Copy artifacts to submission folder
        for f in os.listdir(self.project_dir):
            if f.endswith(('.py', '.md', '.pdf')):
                import shutil
                shutil.copy2(os.path.join(self.project_dir, f), submission_dir)
        
        # Ensure the generated PDF is moved from submission directory if it was compiled there
        if os.path.exists(os.path.join(submission_dir, "paper.pdf")):
             print("LaTeX PDF successfully generated in submission folder.")

        print(f"\nFinal submission package ready in: {submission_dir}")
        
        # QA Loop
        rag = ProjectRAG(self.project_dir)
        print("You can now ask questions about this research. Type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ")
            if question.lower() == 'exit': break
            print("Answer:", rag.query(self.adversary, question))
def main():
    print(f"DEBUG: Starting main.py with args: {sys.argv}")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--field", nargs='+', help="Research field(s)")
        parser.add_argument("--resume", help="Resume from project directory")
        parser.add_argument("--interactive", action="store_true", help="Enable interactive topic selection")
        args = parser.parse_args()
        print(f"DEBUG: Parsed args: {args}")

        if args.resume:
            orch = ResearchOrchestrator()
            print(f"DEBUG: Attempting to load state from: {args.resume}")
            orch.load_state(args.resume)
            orch.run(resume=True)
        elif args.field:
            field = " ".join(args.field)
            orch = ResearchOrchestrator()
            orch.run(field=field, resume=False, interactive=args.interactive)
        else:
            print("Error: Provide --field or --resume")
    except Exception as e:
        print(f"CRITICAL ERROR in main: {e}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
