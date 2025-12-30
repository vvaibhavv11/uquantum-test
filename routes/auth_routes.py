from fastapi import APIRouter, HTTPException, Response, Request
from pydantic import BaseModel
from services.auth_service import auth_service
from settings import settings
import logging

router = APIRouter()

SESSION_COOKIE = "session_token"
_logger = logging.getLogger("auth")

def _cookie_settings():
    samesite = (getattr(settings, "COOKIE_SAMESITE", "none") or "none").lower()
    secure = bool(getattr(settings, "COOKIE_SECURE", True))

    # Browsers require Secure=True when SameSite=None for cross-site cookies.
    if samesite == "none" and not secure:
        _logger.warning("COOKIE_SAMESITE='none' but COOKIE_SECURE=False — this combination will be rejected by browsers. Consider enabling COOKIE_SECURE (HTTPS) for production.")

    return {"httponly": True, "samesite": samesite, "secure": secure}


class LoginPayload(BaseModel):
  username: str
  password: str


class GoogleLoginPayload(BaseModel):
  credential: str


@router.post("/login")
def login(user: LoginPayload, response: Response):
    result = auth_service.login_user(user.username, user.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    cookie_info = _cookie_settings()
    response.set_cookie(SESSION_COOKIE, result["token"], **cookie_info)
    # Debug header to help verify browser/server behavior during development
    response.headers["X-Session-Cookie"] = f"set; samesite={cookie_info.get('samesite')}; secure={cookie_info.get('secure')}; httponly={cookie_info.get('httponly')}"
    return {"msg": "Logged in", "user": result["user"]}


@router.post("/google")
def login_with_google(google_login: GoogleLoginPayload, response: Response):
    # Log receipt of login attempt (do NOT log full token in production)
    _logger.info("Google login attempt received; credential length=%d", len(google_login.credential or ""))
    result = auth_service.login_with_google(google_login.credential)
    if not result:
        _logger.warning("Google login failed for credential (length=%d)", len(google_login.credential or ""))
        raise HTTPException(status_code=401, detail="Invalid Google credential")
    _logger.info("Google login succeeded for user=%s", result["user"].get("email"))
    cookie_info = _cookie_settings()
    response.set_cookie(SESSION_COOKIE, result["token"], **cookie_info)
    # Debug header to help verify browser/server behavior during development
    response.headers["X-Session-Cookie"] = f"set; samesite={cookie_info.get('samesite')}; secure={cookie_info.get('secure')}; httponly={cookie_info.get('httponly')}"
    return {"msg": "Logged in with Google", "user": result["user"]}


@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        auth_service.logout(token)
    response.delete_cookie(SESSION_COOKIE)
    return {"msg": "Logged out"}


@router.get("/me")
def me(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = auth_service.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    return user
