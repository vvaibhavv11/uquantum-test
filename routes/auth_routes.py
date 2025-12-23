from fastapi import APIRouter, HTTPException, Response, Request
from pydantic import BaseModel
from services.auth_service import auth_service

router = APIRouter()

SESSION_COOKIE = "session_token"
COOKIE_SETTINGS = {"httponly": True, "samesite": "lax"}


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
    response.set_cookie(SESSION_COOKIE, result["token"], **COOKIE_SETTINGS)
    return {"msg": "Logged in", "user": result["user"]}


@router.post("/google")
def login_with_google(google_login: GoogleLoginPayload, response: Response):
    result = auth_service.login_with_google(google_login.credential)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid Google credential")
    response.set_cookie(SESSION_COOKIE, result["token"], **COOKIE_SETTINGS)
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
