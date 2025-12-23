from fastapi import APIRouter
from services.hardware_service import hardware_service

router = APIRouter()

@router.get("/list")
def list_backends():
    return {"backends": hardware_service.list_backends()}

@router.get("/coupling_map/{backend}")
def get_coupling_map(backend: str):
    return {"backend": backend, "coupling_map": hardware_service.fetch_coupling_map(backend)}
