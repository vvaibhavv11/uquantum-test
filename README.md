# Uniq Quantum Hub Backend

## Setup

1. Python 3.9+ recommended
2. `pip install -r requirements.txt`
3. Set environment variables (or copy `.env.example` to `.env` and edit values):
   - `INSTANTDB_KEY`
   - `IBM_API_KEY`
   - `REDIS_URL`
   - `GROQ_API_KEY`
   - `FRONTEND_ORIGIN` (comma-separated list of allowed origins, e.g. `http://localhost:3000,https://uquantum.vercel.app` — use `*` to allow all origins but note credentials will be disabled)
4. Run backend:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Structure
- `app.py` – FastAPI main app
- `settings.py` – environment configuration (reads `.env` via pydantic)
- `services/` – business logic modules
- `routes/` – API endpoints
- `transpiler/` – quantum transpilation logic
- `workers/` – job queue worker stub
- `schemas/` – Pydantic models (request/response schemas)
- `utils/` – reusable utilities (error handling, etc.)

## Test
To be added in `tests/`.

---
This backend powers the Uniq Quantum Hub MVP – see code for individual service and route details.
