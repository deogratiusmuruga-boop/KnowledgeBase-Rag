"""
CareBuddy FastAPI Service

API layer for:

- Frontend integration
- STT/TTS integration
- User profile integration

Endpoint:

POST /ask

Input:
{
    "question": "...",
    "user_profile": {...},
    "conversation_history": [...]
}

Output:
{
    "answer": "...",
    "sources": [],
    "reliability": {},
    "decision": {
        "decision": "ACCEPT",
        "score": 0.93,
        "reason": "..."
    }
}
"""

import sys
import os

from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models.user import UserProfile as UserProfileDB



# Add project root

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)



# Import CareBuddy Service

from api.profile import router as profile_router
from api.medication import router as medication_router
from api.appointment import router as appointment_router
from api.reminder import router as reminder_router



# FastAPI Application

app = FastAPI(

    title="CareBuddy API",

    description="""
    RAG-based elderly health and welfare assistant.

    Features:

    - Hybrid Retrieval
    - Evidence grounded generation
    - Source citation
    - Reliability evaluation
    - Adaptive decision control
    - STT/TTS
    - user profile personalization
    """,

    version="1.0.0"

)


# CORS

app.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"

    ],

    allow_credentials=True,

    allow_methods=[

        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS"

    ],

    allow_headers=["*"]

)
app.include_router(profile_router)
app.include_router(medication_router)
app.include_router(appointment_router)
app.include_router(reminder_router)


# User Profile Schema

class UserProfile(BaseModel):

    age: Optional[int] = None

    location: Optional[str] = None

    chronic_conditions: List[str] = Field(default_factory=list)

    medications: List[str] = Field(default_factory=list)

    preferred_language: str = "en"

    speech_speed: str = "normal"


def normalize_response_language(preferred_language: Optional[str]) -> str:
    """Restrict generated responses to the application's supported languages."""
    if isinstance(preferred_language, str):
        language = preferred_language.strip().lower()
        if language in {"ko", "en"}:
            return language
    return "en"



# Conversation History Schema

class ConversationMessage(BaseModel):

    role: str

    content: str



# Request Schema

class ChatMessage(BaseModel):
    role: str
    content: str
class QuestionRequest(BaseModel):

    question: str
    user_id: Optional[int] = None

    user_profile: Optional[UserProfile] = None

    conversation_history: List[ConversationMessage] = Field(default_factory=list)



# Decision Schema

class DecisionResponse(BaseModel):

    decision: str

    score: float

    reason: str


# Response Schema

class QuestionResponse(BaseModel):

    answer: str

    sources: List[str]

    reliability: dict

    decision: DecisionResponse



# Health Check

@app.get("/")
def root():

    return {

        "service": "CareBuddy API",

        "status": "running"

    }



# Ask Endpoint

@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):

    try:
                # Load user profile from database
        if request.user_id is not None:

            db_user = db.query(UserProfileDB).filter(
                UserProfileDB.id == request.user_id
            ).first()

            if db_user is None:
                raise HTTPException(status_code=404, detail="User not found")

            request.user_profile = UserProfile(

                    age=db_user.age,

                    location=db_user.location,

                    chronic_conditions=[
                        value.strip() for value in (db_user.chronic_conditions or "").split(",")
                        if value.strip()
                    ],

                    medications=[
                        value.strip() for value in (db_user.medications or "").split(",")
                        if value.strip()
                    ],

                    preferred_language=db_user.preferred_language,

                    speech_speed=db_user.speech_speed

            )

        # The RAG stack loads sizeable local/remote models.  Import it only when
        # /ask is called so health and CRUD endpoints remain available if the
        # model has not yet been downloaded.
        from scripts.carebuddy_service import answer_question

        response_language = normalize_response_language(
            request.user_profile.preferred_language if request.user_profile else None
        )

        result = answer_question(

            question=request.question,

            user_profile=request.user_profile,

            conversation_history=request.conversation_history,

            response_language=response_language

        )

        return result

    except HTTPException:
        raise
    except Exception:

        raise HTTPException(

            status_code=503,

            detail="The question-answering service is temporarily unavailable."

        )
