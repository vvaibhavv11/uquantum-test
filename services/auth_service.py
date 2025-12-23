import secrets
import time
import requests
from typing import Dict, Optional
from config import settings


class AuthService:
    """
    Simple user/session service (MVP).
    Uses email as unique key; creates a user doc with empty arrays on first login.
    Data is cached in-memory after retrieval for this MVP.
    """

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.users: Dict[str, Dict] = {}  # keyed by email for quick access
        self.google_client_id = "859859771398-erv8s8o5kdvib9k0n1cu8eau2u9l6u4b.apps.googleusercontent.com"

    def _now_iso(self):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _get_or_create_user(self, email: str, provider: str = "password", name: Optional[str] = None):
        if email in self.users:
            return self.users[email]
        user_doc = {
            "id": secrets.token_hex(12),
            "email": email,
            "name": name or email.split("@")[0],
            "created_at": self._now_iso(),
            "recentWorks": [],
            "projects": [],
            "shared": [],
            "starred": [],
            "provider": provider,
        }
        self.users[email] = user_doc
        return user_doc

    def login_user(self, username: str, password: str, provider: str = "password"):
        if not username:
            return None
        user_doc = self._get_or_create_user(username, provider)
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {"email": user_doc["email"], "provider": provider}
        return {"token": token, "user": user_doc}

    def verify_google_token(self, credential: str) -> Optional[Dict]:
        """
        Verify Google ID token and extract user information.
        Returns user info if valid, None otherwise.
        """
        try:
            # Verify the token with Google's tokeninfo endpoint
            response = requests.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": credential},
                timeout=10
            )
            
            if response.status_code != 200:
                return None
            
            token_info = response.json()
            
            # Verify the audience matches our client ID
            if token_info.get("aud") != self.google_client_id:
                return None
            
            # Extract user information
            email = token_info.get("email")
            name = token_info.get("name")
            email_verified = token_info.get("email_verified", False)
            
            if not email or not email_verified:
                return None
            
            return {
                "email": email,
                "name": name,
                "picture": token_info.get("picture"),
            }
        except Exception as e:
            print(f"Error verifying Google token: {e}")
            return None

    def login_with_google(self, credential: str) -> Optional[Dict]:
        """
        Login with Google credential token.
        Returns session token and user info if successful.
        """
        user_info = self.verify_google_token(credential)
        if not user_info:
            return None
        
        user_doc = self._get_or_create_user(
            user_info["email"],
            provider="google",
            name=user_info.get("name")
        )
        
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {"email": user_doc["email"], "provider": "google"}
        return {"token": token, "user": user_doc}

    def get_user_by_token(self, token: str) -> Optional[Dict]:
        session = self.sessions.get(token)
        if not session:
            return None
        email = session["email"]
        return self.users.get(email)

    def logout(self, token: str):
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False


auth_service = AuthService()
