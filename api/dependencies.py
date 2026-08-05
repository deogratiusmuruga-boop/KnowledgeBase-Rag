from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.user import UserProfile


def require_user(db: Session, user_id: int) -> UserProfile:
    """Return an existing user or raise a consistent client-facing error."""
    user = db.get(UserProfile, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
