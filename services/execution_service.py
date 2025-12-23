import logging
import subprocess
import tempfile
import json
from textwrap import dedent
from typing import Dict, List, Tuple

from fastapi import HTTPException, status

from services.ibm_runtime_service import ibm_runtime_client
from services.llm_service import llm_service

logger = logging.getLogger("execution_service")


class ExecutionService:
    def __init__(self):
        self.ibm_client = ibm_runtime_client

    # ------------------------------------------------------------------
    # IBM cloud execution (existing behaviour – QASM only)
    # ------------------------------------------------------------------
    def submit_ibm_jobs(self, job_data: Dict) -> List[str]:
        try:
            code = job_data.get("code", "")
            language = job_data.get("language", "qasm")
            backend = job_data.get("backend")
            shots = int(job_data.get("shots", 0))
            jobs = int(job_data.get("jobs", 1))

            logger.info(f"Submitting job: backend={backend}, shots={shots}, jobs={jobs}, language={language}")

            if language not in {"qasm", "qiskit"}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only 'qasm' and 'qiskit' languages are supported.",
                )
            if not code.strip():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code cannot be empty.")
            if not backend:
                # auto-select an operational backend
                try:
                    live = self.ibm_client.list_backends()
                    operational = [b["name"] for b in live if b.get("operational")]
                    if operational:
                        backend = operational[0]
                        logger.info(f"Auto-selected backend: {backend}")
                    else:
                        # Fallback to any available backend
                        if live:
                            backend = live[0]["name"]
                            logger.info(f"Fallback selected backend: {backend}")
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="No IBM backends available. Please specify a backend.",
                            )
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Error selecting backend: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No IBM backends available: {str(e)}",
                    )
            if shots <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Shots must be positive.")
            if jobs <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Jobs must be positive.")
            if language == "qiskit":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Direct qiskit code submission is not supported for safety. Submit OpenQASM.",
                )

            job_ids: List[str] = []
            max_jobs = min(jobs, 5)  # prevent accidental floods
            for i in range(max_jobs):
                logger.info(f"Submitting job {i+1}/{max_jobs}")
                job_id = self.ibm_client.submit_qasm_job(code, backend, shots)
                job_ids.append(job_id)
                logger.info(f"Job {i+1} submitted successfully: {job_id}")
            
            logger.info(f"All jobs submitted successfully: {job_ids}")
            return job_ids
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error submitting jobs: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit job: {str(e)}",
            )

    def job_status(self, job_id: str) -> Dict:
        try:
            status_name, result = self.ibm_client.job_status_and_result(job_id)
            response = {
                "job_id": job_id,
                "status": status_name,
                "result": result,
            }
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting job status: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get job status: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Local / sandboxed multi-language execution for notebook & simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_language(code: str) -> str:
        """
        Best-effort detection of language from the source text.
        """
        snippet = code.lstrip()
        lower = snippet.lower()

        # OPENQASM header
        if lower.startswith("openqasm"):
            return "qasm"

        # C++ detection
        if "#include" in snippet or "int main(" in snippet or "std::" in snippet:
            return "cpp"

        # Qiskit-style Python
        if "from qiskit" in lower or "import qiskit" in lower:
            return "qiskit"

        # AWS Braket-style Python
        if "from braket" in lower or "import braket" in lower:
            return "braket"

        # Bash detection
        if snippet.startswith("#!") or lower.startswith("pip ") or lower.startswith("ls ") or lower.startswith("cd "):
            return "bash"

        # Fallback: treat as generic Python
        return "python"

    @staticmethod
    def _run_subprocess(cmd: List[str], timeout_s: int = 20) -> Tuple[int, str, str]:
        """
        Helper to run a subprocess safely with a timeout.
        Returns (exit_code, stdout, stderr).
        """
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
                text=True,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"Execution timed out after {timeout_s} seconds.",
            )
        except Exception as exc:  # pragma: no cover - very defensive
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start interpreter: {exc}",
            ) from exc

    def run_local_code(self, code: str, language: str | None = None) -> Dict:
        """
        Execute user code locally in a best-effort, sandboxed way.

        Supported languages for now:
        - 'qasm'   – run as OpenQASM 2.0 using qiskit + Aer simulator
        - 'qiskit' – Python with Qiskit available
        - 'braket' – Python with Braket SDK available
        - 'python' – generic Python

        The *same* API is used by:
        - Quantum Notebook cells
        - Simulation workspace cells (when they are not QASM / RL-transpiled)
        """
        if not code or not code.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Code payload cannot be empty.",
            )

        lang = (language or "").strip().lower() or self._auto_detect_language(code)

        if lang not in {"qasm", "qiskit", "braket", "python", "bash", "sh", "cpp"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported language. Supported: 'qasm', 'qiskit', 'braket', 'python', 'bash', 'sh', 'cpp'.",
            )

        # ------------------------------------------------------------------
        # QASM path – use qiskit + Aer exactly like the simulation backend
        # ------------------------------------------------------------------
        if lang == "qasm":
            try:
                from qiskit import QuantumCircuit
                try:
                    from qiskit_aer import Aer
                except ImportError:
                    from qiskit.providers.aer import Aer  # Fallback for older versions
            except Exception as exc:  # pragma: no cover
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"qiskit-aer is not installed: {exc}",
                ) from exc

            try:
                qc = QuantumCircuit.from_qasm_str(code)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to parse OpenQASM: {exc}",
                ) from exc

            try:
                backend = Aer.get_backend("aer_simulator")
                qc = qc.copy()
                # If the circuit doesn't have measurements, add them
                if not qc.cregs:
                    qc.measure_all()
                job = backend.run(qc, shots=1024)
                result = job.result()
                counts = result.get_counts()
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to execute QASM on simulator: {exc}",
                ) from exc

            return {
                "language": "qasm",
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "counts": counts,
            }

        # ------------------------------------------------------------------
        # C++ path
        # ------------------------------------------------------------------
        if lang == "cpp":
            with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            exe_path = tmp_path.replace(".cpp", ".exe")
            # Compile
            compile_cmd = ["g++", tmp_path, "-o", exe_path]
            c_code, c_out, c_err = self._run_subprocess(compile_cmd, timeout_s=15)
            
            if c_code != 0:
                return {
                    "language": "cpp",
                    "stdout": c_out,
                    "stderr": f"Compilation Error:\n{c_err}",
                    "exit_code": c_code,
                }
            
            # Run
            exit_code, stdout, stderr = self._run_subprocess([exe_path], timeout_s=20)
            return {
                "language": "cpp",
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }

        # ------------------------------------------------------------------
        # Bash / Shell path
        # ------------------------------------------------------------------
        if lang in {"bash", "sh"}:
            cmd = ["/bin/bash", "-c", code]
            exit_code, stdout, stderr = self._run_subprocess(cmd, timeout_s=40)
            return {
                "language": lang,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }

        # ------------------------------------------------------------------
        # Python-family path (python / qiskit / braket)
        # We simply invoke the system's `python3` interpreter. Your backend
        # environment should already have qiskit/braket installed so that
        # imports inside the user code succeed.
        #
        # We also handle Colab-style shell escapes (lines starting with '!')
        # ------------------------------------------------------------------
        processed_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("!"):
                shell_cmd = stripped[1:].replace('"', '\\"')
                processed_lines.append(f'import subprocess; subprocess.run("{shell_cmd}", shell=True)')
            else:
                processed_lines.append(line)
        
        py_code = dedent("\n".join(processed_lines))

        # We use a temporary file instead of `python -c` so that multi-line
        # notebooks / simulations behave naturally.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(py_code)
            tmp_path = tmp.name

        # Use python3 as it is more common on modern systems and specifically
        # exists in this environment while 'python' might not.
        cmd = ["python3", "-u", tmp_path]
        exit_code, stdout, stderr = self._run_subprocess(cmd, timeout_s=40)

        return {
            "language": lang,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }


    async def prepare_environment(self, code: str, language: str | None = None) -> Dict:
        """
        Agentic environment builder:
        1. Uses LLM to detect language and required libraries.
        2. Pre-installs missing libraries (Python) or verifies compiler (C++).
        """
        # Ask LLM to analyze the code for language and dependencies
        prompt = f"""Analyze this code snippet and return ONLY a valid JSON object with this exact structure:
{{
  "language": "python|cpp|qasm|braket|bash",
  "libraries": ["package1", "package2"]
}}

Rules:
- language: must be one of: python, cpp, qasm, braket, bash
- libraries: list of pip package names (for Python) or empty array (for C++/others)
- For Python: extract all import statements and map to pip packages
- For C++: return empty libraries array
- Return ONLY the JSON, no markdown, no explanation

CODE:
{code[:2000]}"""
        
        lang = None
        libraries = []
        
        try:
            llm_res = await llm_service.generate_completion(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                mode="code"
            )
            
            if "error" in llm_res:
                raise Exception(llm_res["error"])
            
            content = llm_res.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract JSON from potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                # Try to extract from any code block
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
            
            # Try to find JSON object in the content
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx+1]
            
            data = json.loads(content)
            lang = data.get("language", "").lower().strip()
            libraries = data.get("libraries", [])
            
            if not lang or lang not in {"python", "cpp", "qasm", "braket", "bash", "qiskit"}:
                raise ValueError(f"Invalid language detected: {lang}")
                
            logger.info(f"LLM detected language: {lang}, libraries: {libraries}")
        except Exception as e:
            logger.warning(f"LLM env detection failed: {e}. Falling back to heuristics.")
            lang = (language or "").strip().lower() or self._auto_detect_language(code)
            libraries = []

        if not lang:
            lang = (language or "").strip().lower() or self._auto_detect_language(code)
        
        if lang in {"python", "qiskit", "braket"}:
            if not libraries:
                # Heuristic fallback for libraries if LLM failed
                import re
                imports = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)
                libraries = list(set(imports))  # Remove duplicates
            
            # Mapping common imports to pip packages
            lib_map = {
                "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
                "scipy": "scipy", "qiskit": "qiskit", "qiskit_aer": "qiskit-aer",
                "braket": "amazon-braket-sdk", "seaborn": "seaborn", 
                "sklearn": "scikit-learn", "torch": "torch",
                "tensorflow": "tensorflow", "cv2": "opencv-python", 
                "PIL": "Pillow", "requests": "requests"
            }
            
            to_install = []
            for lib in libraries:
                lib_base = lib.split('.')[0].strip()
                package = lib_map.get(lib_base, lib_base)
                
                # Check if already installed
                try:
                    __import__(lib_base)
                    logger.debug(f"Package {lib_base} already installed, skipping")
                except ImportError:
                    if package not in to_install:
                        to_install.append(package)
            
            if to_install:
                # Flatten packages that might have spaces (like "qiskit qiskit-aer")
                flat_packages = []
                for pkg in to_install:
                    flat_packages.extend(pkg.split())
                to_install = list(set(flat_packages))
                
                install_cmd = ["python3", "-m", "pip", "install", "--quiet"] + to_install
                logger.info(f"Agentic Env Builder: Auto-installing {to_install}")
                
                exit_code, out, err = self._run_subprocess(install_cmd, timeout_s=120)
                if exit_code == 0:
                    return {"status": "success", "language": lang, "installed": to_install}
                else:
                    return {"status": "error", "language": lang, "error": err[:500]}  # Limit error length
            
            return {"status": "ready", "language": lang, "msg": "All required packages are already installed."}
                
        elif lang == "cpp":
            exit_code, out, err = self._run_subprocess(["g++", "--version"], timeout_s=5)
            if exit_code != 0:
                return {"status": "error", "language": "cpp", "error": "g++ compiler not found. Please install build-essential."}
            return {"status": "ready", "language": "cpp", "msg": "C++ compiler (g++) is available."}
            
        return {"status": "skipped", "reason": f"Language '{lang}' doesn't require auto-prep or not supported", "detected": lang}

execution_service = ExecutionService()
