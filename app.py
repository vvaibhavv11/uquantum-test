from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth_routes, workspace_routes, llm_routes, transpile_routes, execution_routes, hardware_routes
from settings import settings
import logging

app = FastAPI(title="Uniq Quantum Hub Backend MVP")

# CORS
_origins_raw = (settings.FRONTEND_ORIGIN or "").strip()
# If FRONTEND_ORIGIN is set to '*' or is empty, allow any origin but do not allow credentials
if _origins_raw == "*" or not _origins_raw:
    origins = ["*"]
    allow_credentials = False  # browsers disallow credentials with wildcard origin
else:
    origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]
    allow_credentials = True  # explicit origins are safe for credentials

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn").info(f"CORS allowed origins: {origins}, allow_credentials={allow_credentials}")

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
