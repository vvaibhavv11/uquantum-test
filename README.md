# Uniq Quantum Hub Backend

## Setup

1. Python 3.9+ recommended
2. `pip install -r requirements.txt`
3. Set environment variables (or copy `.env.example` to `.env` and edit values):
   - `INSTANTDB_KEY`
   - `IBM_API_KEY`
   - `REDIS_URL`
   - `GROQ_API_KEY`
   - `FRONTEND_ORIGIN` (default points to the production front-end `https://uquantum.vercel.app`; override in `.env` for local development with something like `http://localhost:3000`). For credentialed cookies, set an explicit origin (wildcard `*` is not allowed with credentials).

   Production-first defaults:
   - `FRONTEND_ORIGIN=https://uquantum.vercel.app` (default)
   - `COOKIE_SAMESITE=none` and `COOKIE_SECURE=true` (cookies are cross-site and require HTTPS in production)

   Local development:
   - If you're running locally over HTTP, override `.env` with:
     - `FRONTEND_ORIGIN=http://localhost:3000`
     - `COOKIE_SAMESITE=lax`
     - `COOKIE_SECURE=false`
   - Client must send credentials: `fetch(url, { credentials: 'include' })` or Axios: `{ withCredentials: true }`.
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
This backend powers the Uniq Quantum Hub MVP – see code for individual service and route details.\n\n## License\n\nThis project is licensed under the GNU General Public License v2.0 (GPL-2.0). See LICENSE-GPL-2.0.txt in the root of the repository for the full license text.\n
