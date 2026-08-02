from google import genai
import os
import subprocess
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError, ServerError

class ResearchAgent:
    def __init__(self, system_instruction, model_queue=None):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        self.client = genai.Client(api_key=api_key)

        self.system_instruction = system_instruction
        # 5-tier downgrade strategy starting from cheapest/most efficient
        if model_queue:
            self.model_queue = model_queue
        else:
            self.model_queue = [
                "models/gemini-3.1-flash-lite",
                "models/gemini-3.5-flash",
                "models/gemini-2.0-flash"
            ]
        self.current_model_idx = 0
    @retry(
        stop=stop_after_attempt(10),
        # Wait exponentially, adding 60s min, max 600s, with a random jitter
        wait=wait_exponential(multiplier=2, min=60, max=600),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def chat(self, message):
        # Try models in sequence until one succeeds
        for i in range(len(self.model_queue)):
            model_to_try = self.model_queue[(self.current_model_idx + i) % len(self.model_queue)]
            try:
                print(f"DEBUG: Calling model {model_to_try}", flush=True)
                response = self.client.models.generate_content(
                    model=model_to_try,
                    contents=message,
                    config={"system_instruction": self.system_instruction}
                )
                print(f"DEBUG: Model {model_to_try} response received", flush=True)

                # If successful, update the index for future calls
                self.current_model_idx = (self.current_model_idx + i) % len(self.model_queue)
                return response.text
            except (ClientError, ServerError) as e:
                # Catch 429/404 (Client) or 503 (Server)
                code = getattr(e, 'code', None)
                if code in [429, 404, 503]:
                    print(f"[!] Error {code} for {model_to_try}. Cycling to next model.", flush=True)
                    continue
                raise e

        # If we reach here, all models in queue failed
        print("[!] All models exhausted. Retrying global queue in 60 seconds...", flush=True)
        raise Exception("All fallback models exhausted.")
class CodeExecutor:
    @staticmethod
    def execute_python(code_str, project_dir):
        # 1. Identify and extract shell commands (e.g., #!pip install ... or lines starting with pip install)
        shell_commands = []
        python_lines = []
        
        for line in code_str.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("pip install"):
                shell_commands.append(line_stripped)
            elif line_stripped.startswith("#!") and "pip" in line_stripped:
                shell_commands.append(line_stripped.replace("#!", "").strip())
            else:
                python_lines.append(line)
        
        # 2. Execute shell commands first
        logs = ""
        for cmd in shell_commands:
            print(f"DEBUG: Executing shell dependency install: {cmd}", flush=True)
            res = CodeExecutor.execute_shell(cmd, project_dir)
            logs += res + "\n"
        
        # 3. Create temp file for Python execution
        temp_file_name = "temp_exec.py"
        temp_file_path = os.path.join(project_dir, temp_file_name)
        
        # Track existing files before execution
        before_files = set(os.listdir(project_dir))
        
        with open(temp_file_path, "w") as f:
            f.write('\n'.join(python_lines))
            
        try:
            # Run using the system python3
            result = subprocess.run(
                ["python3", temp_file_name], 
                capture_output=True, text=True, timeout=120,
                cwd=project_dir
            )
            # Track new files
            after_files = set(os.listdir(project_dir))
            new_artifacts = list(after_files - before_files)
            
            logs += f"Stdout: {result.stdout}\nStderr: {result.stderr}"
            return {
                "stdout": logs,
                "stderr": result.stderr,
                "artifacts": new_artifacts
            }
        except Exception as e:
            return {
                "stdout": logs,
                "stderr": str(e),
                "artifacts": []
            }

    @staticmethod
    def execute_shell(cmd_str, project_dir):
        try:
            # Specifically handle pip installs by running them with system pip
            if "pip install" in cmd_str:
                # Remove any potential existing path in the command string that might cause duplication
                clean_cmd = cmd_str.replace("pip install", "").strip()
                cmd_str = f"pip install {clean_cmd}"
            
            # Run in the project directory
            result = subprocess.run(
                cmd_str, shell=True,
                capture_output=True, text=True, timeout=120,
                cwd=project_dir
            )
            return f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        except Exception as e:
            return f"Execution Error: {str(e)}"
