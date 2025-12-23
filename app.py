import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth_routes, workspace_routes, llm_routes, transpile_routes, execution_routes, hardware_routes
import logging

app = FastAPI(title="Uniq Quantum Hub Backend MVP")

# CORS
frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000,http://localhost:8080")
origins = [o.strip() for o in frontend_origin_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Accept both ports for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

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
