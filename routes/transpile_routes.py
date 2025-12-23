import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from services.transpile_service import transpile_service

logger = logging.getLogger("transpile_routes")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class TranspileRequest(BaseModel):
    qasm: str
    mode: str = "safe-rl"  # "static" | "parallel" | "safe-rl"
    timeout_s: float | None = None  # optional per-request timeout

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str):
        allowed = {"static", "parallel", "safe-rl"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

router = APIRouter()

@router.post("/run")
def run_transpile(payload: TranspileRequest):
    try:
        logger.info("Received transpile request mode=%s", payload.mode)
        result = transpile_service.transpile_circuit(
            qasm=payload.qasm,
            mode=payload.mode,
            timeout_s=payload.timeout_s,
        )
        return {"msg": "Transpilation complete", "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transpilation failed")
        raise HTTPException(status_code=500, detail=str(exc))


class SimulateRequest(TranspileRequest):
    shots: int = 1024
    noise_enabled: bool = False
    noise_strength: float | None = None
    # Names of metrics the UI is specifically interested in;
    # the service may compute a superset and filter.
    noise_metrics: list[str] | None = None


@router.post("/simulation/run")
def run_simulation(payload: SimulateRequest):
    try:
        logger.info("Received simulation request mode=%s shots=%s", payload.mode, payload.shots)
        result = transpile_service.transpile_and_execute(
            qasm=payload.qasm,
            mode=payload.mode,
            shots=payload.shots,
            timeout_s=payload.timeout_s,
            noise_enabled=payload.noise_enabled,
            noise_strength=payload.noise_strength,
            noise_metrics=payload.noise_metrics,
        )
        return {"msg": "Simulation complete", "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(exc))
