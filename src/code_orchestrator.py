import os
import subprocess
import json
import shutil

class CodeOrchestrator:
    def __init__(self, task_id, base_dir, worktree_dir, repo_dir):
        self.task_id = task_id
        self.task_dir = os.path.join(base_dir, task_id)
        self.worktree_path = os.path.join(worktree_dir, task_id)
        self.repo_dir = repo_dir
        
        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)

    def initialize_worktree(self, issue_number):
        # 1. Create branch
        branch_name = f"gemini/issue-{issue_number}"
        
        # 2. Add worktree
        # Assumes repo_dir is a git repo
        subprocess.run(["git", "worktree", "add", self.worktree_path, "-b", branch_name], cwd=self.repo_dir, check=True)
        
        # 3. Initialize task state
        state = {"issue_number": issue_number, "branch": branch_name}
        with open(os.path.join(self.task_dir, "task.json"), "w") as f:
            json.dump(state, f)

    def run_gemini(self, instructions):
        # Prepare context for Gemini CLI
        # This will need to invoke the CLI tool in a headless manner
        pass
