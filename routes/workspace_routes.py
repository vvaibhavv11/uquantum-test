from fastapi import APIRouter
from services.workspace_service import workspace_service

router = APIRouter()

@router.post("/create")
def create_workspace(workspace: dict):
    result = workspace_service.create_workspace(workspace.get("user_id"), workspace)
    return {"msg": "Workspace created", "result": result}

@router.get("/list/{user_id}")
def list_workspaces(user_id: str):
    result = workspace_service.list_workspaces(user_id)
    return {"workspaces": result}

@router.post("/save/{workspace_id}")
def save_workspace(workspace_id: str, workspace: dict):
    result = workspace_service.save_workspace(workspace_id, workspace)
    return {"msg": "Workspace saved", "result": result}

@router.get("/get/{workspace_id}")
def get_workspace(workspace_id: str):
    result = workspace_service.get_workspace(workspace_id)
    return {"workspace": result}
