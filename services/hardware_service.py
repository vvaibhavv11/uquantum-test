from typing import List, Dict
from services.ibm_runtime_service import ibm_runtime_client


class HardwareService:
    def __init__(self):
        self.ibm_client = ibm_runtime_client

    def _fallback_backends(self) -> List[Dict]:
        # Basic static info when live listing is unavailable.
        # Common IBM backends as fallback
        fallback_names = ["ibm_brisbane", "ibm_lagos", "ibm_perth", "ibm_fez", "ibm_torino"]
        return [
            {"name": name, "num_qubits": None, "simulator": False, "operational": True, "status": "online"}
            for name in fallback_names
        ]

    def list_backends(self):
        try:
            live = self.ibm_client.list_backends()
            # Return all available backends, not filtered
            return live if live else self._fallback_backends()
        except Exception:
            return self._fallback_backends()

    def fetch_coupling_map(self, backend_name: str):
        try:
            return self.ibm_client.get_coupling_map(backend_name)
        except Exception:
            return None


hardware_service = HardwareService()
