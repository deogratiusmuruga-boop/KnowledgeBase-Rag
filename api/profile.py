from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models.user import UserProfile


router = APIRouter(
    prefix="/profile",
    tags=["User Profile"]
)


# ================================
# Request Schema
# ================================

from pydantic import BaseModel, Field
from typing import Optional, List


class ProfileCreate(BaseModel):

    age: Optional[int] = None

    location: Optional[str] = None

    chronic_conditions: List[str] = Field(default_factory=list)

    medications: List[str] = Field(default_factory=list)

    preferred_language: str = "en"

    speech_speed: str = "normal"



# ================================
# Create Profile
# ================================

@router.post("/")
def create_profile(
    profile: ProfileCreate,
    db: Session = Depends(get_db)
):

    user = UserProfile(

        age=profile.age,

        location=profile.location,

        chronic_conditions=", ".join(
            profile.chronic_conditions
        ),

        medications=", ".join(
            profile.medications
        ),

        preferred_language=profile.preferred_language,

        speech_speed=profile.speech_speed
    )


    db.add(user)

    db.commit()

    db.refresh(user)


    return {

        "message": "Profile created successfully",

        "user_id": user.id,

        "profile": {

            "age": user.age,

            "location": user.location,

            "chronic_conditions": user.chronic_conditions,

            "medications": user.medications

        }

    }
