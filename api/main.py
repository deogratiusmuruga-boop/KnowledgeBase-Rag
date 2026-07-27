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
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.user import UserProfile as UserProfileDB


# ============================================================
# Add project root
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


# ============================================================
# Import CareBuddy Service
# ============================================================

from scripts.carebuddy_service import answer_question
from api.profile import router as profile_router


# ============================================================
# FastAPI Application
# ============================================================

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
    - Future STT/TTS integration
    - Future user profile personalization
    """,

    version="1.0.0"

)


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# User Profile Schema
# ============================================================

class UserProfile(BaseModel):

    age: Optional[int] = None

    location: Optional[str] = None

    chronic_conditions: List[str] = []

    medications: List[str] = []

    preferred_language: str = "en"

    speech_speed: str = "normal"


# ============================================================
# Conversation History Schema
# ============================================================

class ConversationMessage(BaseModel):

    role: str

    content: str


# ============================================================
# Request Schema
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str
class QuestionRequest(BaseModel):

    question: str
    user_id: Optional[int] = None

    user_profile: Optional[UserProfile] = None

    conversation_history: List[ConversationMessage] = []


# ============================================================
# Decision Schema
# ============================================================

class DecisionResponse(BaseModel):

    decision: str

    score: float

    reason: str


# ============================================================
# Response Schema
# ============================================================

class QuestionResponse(BaseModel):

    answer: str

    sources: List[str]

    reliability: dict

    decision: DecisionResponse


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():

    return {

        "service": "CareBuddy API",

        "status": "running"

    }


# ============================================================
# Ask Endpoint
# ============================================================

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
        if request.user_id:

            db_user = db.query(UserProfileDB).filter(
                UserProfileDB.id == request.user_id
            ).first()

            if db_user:

                request.user_profile = UserProfile(

                    age=db_user.age,

                    location=db_user.location,

                    chronic_conditions=db_user.chronic_conditions.split(","),

                    medications=db_user.medications.split(","),

                    preferred_language=db_user.preferred_language,

                    speech_speed=db_user.speech_speed

                )

        result = answer_question(

            question=request.question,

            user_profile=request.user_profile,

            conversation_history=request.conversation_history

        )

        return result

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )