from typing import Dict, List, Optional, Tuple
from fastapi import HTTPException, status
import logging
from config import settings


class IBMRuntimeClient:
    """
    Thin wrapper around QiskitRuntimeService with lazy initialization.
    Keeps imports local to avoid hard failures when dependency is missing at startup.
    """

    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.channel = settings.IBM_CHANNEL
        self.instance = settings.IBM_INSTANCE_CRN
        self._service = None
        self.allowed_backends = []  # Will be populated dynamically from available backends

    def _require_api_key(self):
        if not self.api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="IBM_API_KEY is not configured on the server.",
            )

    def _get_service(self):
        self._require_api_key()
        if self._service is None:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
            except Exception as exc:  # pragma: no cover - dependency issues
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"qiskit-ibm-runtime is not available: {exc}",
                ) from exc

            kwargs = {"channel": self.channel, "token": self.api_key}
            if self.instance:
                kwargs["instance"] = self.instance
            self._service = QiskitRuntimeService(**kwargs)
            logging.getLogger("ibm_runtime").info(
                "Initialized QiskitRuntimeService channel=%s instance=%s", self.channel, self.instance
            )
        return self._service

    def list_backends(self) -> List[Dict]:
        service = self._get_service()
        backends = []
        try:
            for backend in service.backends():
                try:
                    status_obj = backend.status()
                    status_msg = status_obj.status_msg
                    operational = status_obj.operational
                    pending_jobs = status_obj.pending_jobs
                except Exception:
                    status_msg = "unknown"
                    operational = False
                    pending_jobs = None

                backend_info = {
                    "name": backend.name,
                    "num_qubits": getattr(backend, "num_qubits", None),
                    "simulator": getattr(backend, "simulator", False),
                    "operational": operational,
                    "status": status_msg,
                    "pending_jobs": pending_jobs,
                }
                backends.append(backend_info)
                
                # Update allowed_backends list with all available backends
                if backend.name not in self.allowed_backends:
                    self.allowed_backends.append(backend.name)
        except Exception as exc:
            logging.getLogger("ibm_runtime").error(f"Error listing backends: {exc}")
        return backends

    def get_coupling_map(self, backend_name: str):
        service = self._get_service()
        backend = service.backend(backend_name)
        return getattr(backend, "coupling_map", None)

    def submit_qasm_job(self, qasm: str, backend: str, shots: int) -> str:
        service = self._get_service()
        logger = logging.getLogger("ibm_runtime")
        
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_ibm_runtime import SamplerV2 as Sampler
        except Exception as exc:
            logger.error(f"Failed to import qiskit libraries: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to import qiskit: {exc}",
            ) from exc

        try:
            logger.info(f"Parsing QASM code (length: {len(qasm)})")
            qc = QuantumCircuit.from_qasm_str(qasm)
            logger.info(f"Circuit created: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
        except Exception as exc:
            logger.error(f"Failed to parse QASM: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid OpenQASM payload: {exc}",
            ) from exc

        try:
            logger.info(f"Getting backend: {backend}")
            backend_obj = service.backend(backend)
            logger.info(f"Backend retrieved: {backend_obj.name}, simulator={getattr(backend_obj, 'simulator', False)}")
            
            # Try transpiling for hardware backends (not simulators)
            # Match Colab code - they don't transpile, but we'll try it for hardware
            circuit_to_run = qc
            if not getattr(backend_obj, 'simulator', False):
                try:
                    logger.info("Transpiling circuit for hardware backend...")
                    circuit_to_run = transpile(qc, backend=backend_obj)
                    logger.info(f"Circuit transpiled successfully: {circuit_to_run.num_qubits} qubits")
                except Exception as transpile_exc:
                    logger.warning(f"Transpilation failed, using original circuit: {transpile_exc}")
                    circuit_to_run = qc  # Fallback to original circuit
            
            # Use SamplerV2 (new API) - matching Colab code exactly
            logger.info(f"Creating SamplerV2 with backend {backend_obj.name}")
            sampler = Sampler(backend_obj)
            
            logger.info(f"Running circuit with {shots} shots...")
            # Match Colab: sampler.run([qc], shots=shots)
            job = sampler.run([circuit_to_run], shots=shots)
            job_id = job.job_id()
            logger.info(f"Job submitted successfully: {job_id}")
        except Exception as exc:
            logger.error(f"Failed to submit job: {exc}", exc_info=True)
            error_detail = str(exc)
            # Provide more helpful error messages
            if "authentication" in error_detail.lower() or "token" in error_detail.lower():
                error_detail = f"Authentication failed. Please check your IBM API key. Original error: {error_detail}"
            elif "backend" in error_detail.lower() and "not found" in error_detail.lower():
                error_detail = f"Backend '{backend}' not found or not available. Original error: {error_detail}"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to submit job to IBM backend '{backend}': {error_detail}",
            ) from exc

        return job_id

    def job_status_and_result(self, job_id: str, expected_shots: Optional[int] = None) -> Tuple[str, Optional[Dict]]:
        service = self._get_service()
        logger = logging.getLogger("ibm_runtime")
        
        try:
            logger.info(f"Fetching job: {job_id}")
            job = service.job(job_id)
        except Exception as exc:
            logger.error(f"Job not found: {job_id}, error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found: {exc}",
            ) from exc

        try:
            status_obj = job.status()
            # Convert status to string (might be enum)
            status_name = str(status_obj.name) if hasattr(status_obj, 'name') else str(status_obj)
            logger.info(f"Job {job_id} status: {status_name}")
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            status_name = "UNKNOWN"

        result = None
        if status_name in {"DONE", "COMPLETED"}:
            try:
                logger.info(f"Job {job_id} is done, extracting results...")
                # Use SamplerV2 result format: result[0].data.c.get_counts()
                qiskit_result = job.result()
                logger.info(f"Result type: {type(qiskit_result)}")
                
                # Check if this is SamplerV2 result format (list-like)
                if hasattr(qiskit_result, '__getitem__') and hasattr(qiskit_result, '__len__'):
                    if len(qiskit_result) > 0:
                        # SamplerV2 format: result is a list of PubResult objects
                        pub_result = qiskit_result[0]
                        logger.info(f"PubResult type: {type(pub_result)}")
                        
                        if hasattr(pub_result, 'data'):
                            logger.info(f"PubResult.data type: {type(pub_result.data)}")
                            # Try the SamplerV2 format: pub_result.data.c.get_counts()
                            if hasattr(pub_result.data, 'c'):
                                try:
                                    counts = pub_result.data.c.get_counts()
                                    logger.info(f"Got counts: {counts}")
                                    # Convert BitArray or similar to regular dict
                                    if counts:
                                        result = {str(k): int(v) for k, v in dict(counts).items()}
                                        logger.info(f"Converted result: {result}")
                                except Exception as e:
                                    logger.error(f"Error getting counts from c: {e}")
                                    # Fallback: try to access counts differently
                                    try:
                                        if hasattr(pub_result.data.c, '__dict__'):
                                            result = {str(k): int(v) for k, v in pub_result.data.c.__dict__.items() if not k.startswith('_')}
                                    except Exception as e2:
                                        logger.error(f"Fallback also failed: {e2}")
                            
                            # Fallback: try to get counts from data directly
                            if result is None and hasattr(pub_result.data, 'get_counts'):
                                try:
                                    counts = pub_result.data.get_counts()
                                    result = {str(k): int(v) for k, v in dict(counts).items()} if counts else None
                                except Exception as e:
                                    logger.error(f"Error with get_counts: {e}")
                
                # Fallback to old Sampler API format
                if result is None and hasattr(qiskit_result, "get_counts"):
                    try:
                        counts = qiskit_result.get_counts()
                        result = {str(k): int(v) for k, v in dict(counts).items()} if counts else None
                    except Exception as e:
                        logger.error(f"Error with old API get_counts: {e}")
                
                # Fallback to quasi-probability distribution format
                if result is None and hasattr(qiskit_result, "quasi_dists"):
                    try:
                        quasi = qiskit_result.quasi_dists[0]
                        shots = expected_shots or 1024
                        result = {str(bit): int(round(prob * shots)) for bit, prob in quasi.items()}
                    except Exception as e:
                        logger.error(f"Error with quasi_dists: {e}")
                
                if result is None:
                    logger.warning(f"Could not extract result for job {job_id}, result object: {qiskit_result}")
                    
            except Exception as exc:
                logger.error(f"Error extracting result: {exc}", exc_info=True)
                result = None

        return status_name, result


ibm_runtime_client = IBMRuntimeClient(api_key=settings.IBM_API_KEY)

