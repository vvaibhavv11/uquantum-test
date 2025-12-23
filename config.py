import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    INSTANTDB_KEY: str = os.getenv("INSTANTDB_KEY", "")
    IBM_API_KEY: Optional[str] = os.getenv(
        "IBM_API_KEY",
        # Default provided by user for development; override in production via env.
        "scb2YbKsUUhIFRp_opfgdwL5_5gDiz--7KTMx2CW2XQC",
    )
    IBM_CHANNEL: str = os.getenv("IBM_CHANNEL", "ibm_cloud")
    IBM_INSTANCE_CRN: Optional[str] = os.getenv("IBM_INSTANCE_CRN")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

settings = Settings()
