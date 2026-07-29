from google import genai
import os
import subprocess
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError, ServerError

class ResearchAgent:
    def __init__(self, system_instruction):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        self.client = genai.Client(api_key=api_key)
        self.system_instruction = system_instruction
        # 5-tier downgrade strategy starting from cheapest/most efficient
        self.model_queue = [
            "gemini-3.1-flash-lite", 
            "gemini-2.0-flash-lite-preview",
            "gemini-2.0-flash-exp", 
            "gemini-2.0-flash",
            "gemini-3.5-flash"
        ]
        self.current_model_idx = 0

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=60, max=300),
        retry=retry_if_exception_type(Exception)
    )
    def chat(self, message):
        # Try models in sequence until one succeeds
        for i in range(len(self.model_queue)):
            model_to_try = self.model_queue[(self.current_model_idx + i) % len(self.model_queue)]
            try:
                print(f"Trying model: {model_to_try}...")
                response = self._call_model_without_retry(model_to_try, message)
                # If successful, update the index for future calls
                self.current_model_idx = (self.current_model_idx + i) % len(self.model_queue)
                return response
            except (ClientError, ServerError) as e:
                # Catch 429/404 (Client) or 503 (Server)
                code = getattr(e, 'code', None)
                if code in [429, 404, 503]:
                    print(f"[!] Error {code} for {model_to_try}. Cycling to next model.")
                    continue
                raise e
        
        # If we reach here, all models in queue failed
        print("[!] All models exhausted. Retrying global queue in 60 seconds...")
        raise Exception("All fallback models exhausted.")

    def _call_model_without_retry(self, model_name, message):
        # Prefix with 'models/' if not present
        model_id = model_name if model_name.startswith("models/") else f"models/{model_name}"
        # Ensure it doesn't have a double prefix if 'models/' was already there
        model_id = model_id.replace("models/models/", "models/")
        
        response = self.client.models.generate_content(
            model=model_id,
            contents=message,
            config={"system_instruction": self.system_instruction}
        )
        return response.text

class CodeExecutor:
    @staticmethod
    def execute_python(code_str, project_dir):
        # Create temp file inside the project directory
        temp_file_name = "temp_exec.py"
        temp_file_path = os.path.join(project_dir, temp_file_name)
        with open(temp_file_path, "w") as f:
            f.write(code_str)
        try:
            # Run using the system python3
            python_cmd = "python3"
            # Increase timeout for complex calculations
            # Run in the project directory using the relative filename
            result = subprocess.run(
                [python_cmd, temp_file_name], 
                capture_output=True, text=True, timeout=60,
                cwd=project_dir
            )
            return f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        except Exception as e:
            return f"Execution Error: {str(e)}"

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
