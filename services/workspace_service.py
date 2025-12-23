from config import settings

class WorkspaceService:
    def __init__(self):
        self.api_key = settings.INSTANTDB_KEY

    def create_workspace(self, user_id: str, data: dict):
        # TODO: Create new workspace
        pass

    def list_workspaces(self, user_id: str):
        # TODO: Return list of user's workspaces
        pass

    def save_workspace(self, workspace_id: str, data: dict):
        # TODO: Save workspace
        pass

    def get_workspace(self, workspace_id: str):
        # TODO: Get workspace by id
        pass

workspace_service = WorkspaceService()
