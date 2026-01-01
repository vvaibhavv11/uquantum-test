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
    # Default to the production front-end origin. Override via `.env` for local development.
    FRONTEND_ORIGIN: str = "https://uquantum.vercel.app"
    
    # Transpiler Configuration
    TRANSPILER_BACKEND_QUBITS: int = 27
    TRANSPILER_MODEL_PATH: Optional[str] = None
    TRANSPILER_LOG_DIR: str = "./logs"

    # Cookie settings (session cookie behavior)
    # PRODUCTION defaults: allow cross-site cookies for production front-ends.
    # Use COOKIE_SAMESITE='none' and COOKIE_SECURE=True for production (requires HTTPS).
    # For local development, override these in `.env` (e.g., COOKIE_SAMESITE=lax, COOKIE_SECURE=false).
    COOKIE_SAMESITE: str = "none"
    COOKIE_SECURE: bool = True  # Production default: require secure cookies (HTTPS)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
