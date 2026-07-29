import os
import subprocess
import json
import shutil
import sys
import tempfile

class CodeOrchestrator:
    def __init__(self, task_id, base_dir, worktree_dir, repo_dir, issue_num):
        self.task_id = task_id
        self.task_dir = os.path.join(base_dir, task_id)
        self.worktree_path = os.path.join(worktree_dir, task_id)
        self.repo_dir = repo_dir
        self.issue_num = issue_num
        self.branch_name = f"gemini/issue-{self.issue_num}"
        
        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)

    def load_state(self):
        state_file = os.path.join(self.task_dir, "task.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                return json.load(f)
        return {"issue_number": self.issue_num, "branch": self.branch_name, "status": "initialized"}

    def save_state(self, state):
        with open(os.path.join(self.task_dir, "task.json"), "w") as f:
            json.dump(state, f)

    def initialize_worktree(self):
        if os.path.exists(self.worktree_path):
            # If the directory exists but isn't a git repo, fix it
            if not os.path.exists(os.path.join(self.worktree_path, ".git")):
                shutil.rmtree(self.worktree_path)
            else:
                return

        # Check if branch exists
        res = subprocess.run(["git", "show-ref", "--verify", f"refs/heads/{self.branch_name}"], cwd=self.repo_dir, capture_output=True)
        if res.returncode != 0:
            # Create worktree with new branch
            subprocess.run(["git", "worktree", "add", self.worktree_path, "-b", self.branch_name], cwd=self.repo_dir, check=True)
        else:
            # Create worktree with existing branch
            subprocess.run(["git", "worktree", "add", self.worktree_path, self.branch_name], cwd=self.repo_dir, check=True)

    def _get_args(self, args_file):
        if args_file and os.path.exists(args_file):
            with open(args_file, "r") as f:
                return f.read().strip()
        return ""

    def run_gemini(self, instructions):
        # We write instructions to a file to pass to gemini safely
        prompt_file = os.path.join(self.task_dir, "prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(instructions)
        
        # Simulated or actual headless run
        print(f"Running Gemini headless with prompt file: {prompt_file}")
        
        # Invoke the actual agent headless command.
        env = os.environ.copy()
        
        with open(prompt_file, "r") as f:
            prompt_content = f.read()

        # Correct flags based on help output:
        # -p/--prompt for headless
        # --approval-mode=yolo for auto-approval
        # --skip-trust to bypass trust check
        try:
            subprocess.run(["gemini", "-p", prompt_content, "--approval-mode=yolo", "--skip-trust"], 
                           cwd=self.worktree_path, env=env, check=True)
            print("Agent run completed.")
        except FileNotFoundError:
            # Fallback if 'gemini' is not in PATH
            print("Warning: 'gemini' executable not found in PATH, simulating run.")
            pass

    def commit_and_push(self, commit_msg):
        # Only add changes within the worktree
        subprocess.run(["git", "add", "."], cwd=self.worktree_path, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=self.worktree_path, capture_output=True, text=True)
        if not status.stdout.strip():
            print("No changes to commit.")
            return False
            
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=self.worktree_path, check=True)
        subprocess.run(["git", "push", "--set-upstream", "origin", self.branch_name], cwd=self.worktree_path, check=True)
        return True

    def create_or_update_pr(self, title, body):
        # Try to create PR
        # using gh cli if available or simulated
        cmd = [
            "gh", "pr", "create", 
            "--head", self.branch_name, 
            "--base", "main", 
            "--title", title, 
            "--body", body,
            "--draft"
        ]
        res = subprocess.run(cmd, cwd=self.repo_dir, capture_output=True, text=True)
        if res.returncode == 0:
            print("Created Draft PR:", res.stdout.strip())
        else:
            if "already exists" in res.stderr:
                print("PR already exists, updated via push.")
            else:
                print("Failed to create PR:", res.stderr)

    def execute_command(self, cmd, args_file):
        state = self.load_state()
        args = self._get_args(args_file)
        
        if cmd == "run":
            print(f"Starting run for task {self.task_id}")
            self.initialize_worktree()
            state["status"] = "running"
            self.save_state(state)
            
            self.run_gemini(f"Please implement the requested feature for issue #{self.issue_num}. Details:\n{args}")
            
            if self.commit_and_push(f"Auto-generated implementation for issue #{self.issue_num}"):
                self.create_or_update_pr(f"Fix Issue #{self.issue_num}", f"Resolves #{self.issue_num}\n\nGenerated by Gemini.")
            
            state["status"] = "completed"
            self.save_state(state)
            
        elif cmd == "revise":
            print(f"Starting revision for task {self.task_id}")
            self.initialize_worktree()
            state["status"] = "revising"
            self.save_state(state)
            
            self.run_gemini(f"Please revise the implementation for issue #{self.issue_num}. Revisions requested:\n{args}")
            
            if self.commit_and_push(f"Revisions for issue #{self.issue_num}"):
                print("Pushed revisions to PR.")
                
            state["status"] = "completed"
            self.save_state(state)
            
        elif cmd == "test":
            print(f"Running tests for task {self.task_id}")
            self.initialize_worktree()
            # Run pytest or equivalent in worktree
            res = subprocess.run(["pytest"], cwd=self.worktree_path, capture_output=True, text=True)
            print("Test Output:\n", res.stdout)
            if res.stderr:
                print("Test Errors:\n", res.stderr)
            
        elif cmd == "status":
            print(f"Status for task {self.task_id}: {state.get('status')}")
            if os.path.exists(self.worktree_path):
                status_res = subprocess.run(["git", "status", "-s"], cwd=self.worktree_path, capture_output=True, text=True)
                print("Worktree changes:\n", status_res.stdout)
                
        elif cmd == "stop":
            print(f"Stopping task {self.task_id}")
            state["status"] = "stopped"
            self.save_state(state)
            # Send stop signal to any running gemini process (mocked via stop marker)
            with open(os.path.join(self.task_dir, ".stop_marker"), "w") as f:
                f.write("stop")
            print("Stop marker created.")
        else:
            print(f"Unknown command: {cmd}")
