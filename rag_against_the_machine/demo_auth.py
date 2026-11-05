import os

class AuthManager:
    """Manages user authentication tokens."""
    
    def __init__(self, secret: str):
        self.secret = secret

    async def verify_token(self, token: str) -> dict:
        if not token:
            raise ValueError("Empty token")
        return {"status": "authenticated"}

def global_helper():
    return os.getpid()