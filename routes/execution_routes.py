import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.execution_service import execution_service

logger = logging.getLogger("execution_routes")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

router = APIRouter()


class IBMRunRequest(BaseModel):
    code: str
    language: str = Field(default="qasm", description="Accepted: 'qasm' or 'qiskit'")
    backend: str
    shots: int = Field(gt=0, default=1024)
    jobs: int = Field(gt=0, default=1)


@router.post("/ibm/run")
def submit_ibm_job(payload: IBMRunRequest):
    try:
        logger.info(
            "Received job submission request: backend=%s, shots=%s, jobs=%s",
            payload.backend,
            payload.shots,
            payload.jobs,
        )
        job_ids = execution_service.submit_ibm_jobs(payload.model_dump())
        logger.info("Submitted IBM jobs successfully: %s", job_ids)
        return {"msg": "Job submitted", "job_ids": job_ids}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to submit IBM job")
        error_msg = str(exc)
        logger.error("Error details: %s", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/status/{job_id}")
def job_status(job_id: str):
    try:
        status = execution_service.job_status(job_id)
        return status
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to fetch job status for %s", job_id)
        raise HTTPException(status_code=500, detail=str(exc))


class CodeRunRequest(BaseModel):
    code: str
    language: str | None = Field(
        default=None,
        description=(
            "Optional explicit language hint. "
            "Supported: 'qasm', 'qiskit', 'braket', 'python'. "
            "If omitted, the backend will auto-detect."
        ),
    )


@router.post("/run-code")
def run_code(payload: CodeRunRequest):
    """
    Generic multi-language execution endpoint used by:
    - Quantum Notebook cells
    - Simulation workspace cells (for non-QASM code)
    """
    try:
        logger.info("Executing code via /execution/run-code language=%s", payload.language)
        result = execution_service.run_local_code(
            code=payload.code,
            language=payload.language,
        )
        return {"msg": "Execution complete", "result": result}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Local code execution failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/prepare-env")
async def prepare_env(payload: CodeRunRequest):
    """
    Trigger the agentic environment builder for a given code snippet.
    Uses LLM to detect language and required libraries, then auto-installs them.
    """
    try:
        result = await execution_service.prepare_environment(
            code=payload.code,
            language=payload.language
        )
        return result
    except Exception as exc:
        logger.exception("Environment preparation failed")
        raise HTTPException(status_code=500, detail=str(exc))
