from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # Database Configuration
    INSTANTDB_KEY: str = ""
    
    # IBM Quantum Configuration
    IBM_API_KEY: Optional[str] = None
    IBM_CHANNEL: str = "ibm_cloud"
    IBM_INSTANCE_CRN: Optional[str] = None
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # LLM/API Configuration
    GROQ_API_KEY: str = ""
    
    # Frontend Configuration
    FRONTEND_ORIGIN: str = "http://localhost:3000,http://localhost:8080"
    
    # Transpiler Configuration
    TRANSPILER_BACKEND_QUBITS: int = 27
    TRANSPILER_MODEL_PATH: Optional[str] = None
    TRANSPILER_LOG_DIR: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
