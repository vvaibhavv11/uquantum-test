from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth_routes, workspace_routes, llm_routes, transpile_routes, execution_routes, hardware_routes
from settings import settings
from dotenv import load_dotenv, dotenv_values
import os
import logging

app = FastAPI(title="Uniq Quantum Hub Backend MVP")
load_dotenv() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization'],
)

# Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").info(f"CORS allowed origins: {settings.FRONTEND_ORIGIN}, allow_credentials=True")

# Include Routers
app.include_router(auth_routes.router, prefix="/auth")
app.include_router(workspace_routes.router, prefix="/workspace")
app.include_router(llm_routes.router, prefix="/llm")
app.include_router(transpile_routes.router, prefix="/transpile")
app.include_router(execution_routes.router, prefix="/execution")
app.include_router(hardware_routes.router, prefix="/hardware")

@app.get("/")
def root():
    return {"msg": "Uniq Quantum Hub Backend running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
