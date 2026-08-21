
"""
CareBuddy FastAPI Service

API layer for:

- Frontend integration
- STT/TTS integration
- User profile integration
- ElderDocAI adaptive care context

Endpoint:

POST /ask

Input:
{
    "question": "...",
    "user_id": 5,
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
    },
    "care_context": {
        "available": true,
        "care_state": {},
        "transition": {},
        "adaptive_assistance": {}
    },
    "profile_used": true
}
"""

import os
import sys
import traceback

from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


# ============================================================
# DATABASE
# ============================================================

from database import get_db
from models.user import UserProfile as UserProfileDB


# ============================================================
# API ROUTERS
# ============================================================

from api.profile import router as profile_router
from api.medication import router as medication_router
from api.appointment import router as appointment_router
from api.reminder import router as reminder_router


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CareBuddy API",
    description="""
    RAG-based elderly health and welfare assistant.

    Features:

    - Hybrid Retrieval
    - Evidence grounded generation
    - Reliability evaluation
    - Adaptive decision control
    - ElderDocAI adaptive care context
    - STT/TTS
    - User profile personalization
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


# ============================================================
# ROUTERS
# ============================================================

app.include_router(profile_router)
app.include_router(medication_router)
app.include_router(appointment_router)
app.include_router(reminder_router)


# ============================================================
# USER PROFILE SCHEMA
# ============================================================

class UserProfile(BaseModel):
    patient_id: Optional[str] = None

    age: Optional[int] = None

    location: Optional[str] = None

    chronic_conditions: List[str] = Field(
        default_factory=list
    )

    medications: List[str] = Field(
        default_factory=list
    )

    preferred_language: str = "en"

    speech_speed: str = "normal"


# ============================================================
# LANGUAGE NORMALIZATION
# ============================================================

def normalize_response_language(
    preferred_language: Optional[str]
) -> str:
    """
    Restrict generated responses to the application's
    supported languages.

    Supported:
    - ko
    - en

    Fallback:
    - en
    """

    if isinstance(preferred_language, str):

        language = preferred_language.strip().lower()

        if language in {"ko", "en"}:
            return language

    return "en"


# ============================================================
# CONVERSATION HISTORY
# ============================================================

class ConversationMessage(BaseModel):

    role: str

    content: str


# ============================================================
# REQUEST SCHEMA
# ============================================================

class QuestionRequest(BaseModel):

    question: str

    user_id: Optional[int] = None

    user_profile: Optional[UserProfile] = None

    conversation_history: List[ConversationMessage] = Field(
        default_factory=list
    )


# ============================================================
# DECISION RESPONSE
# ============================================================

class DecisionResponse(BaseModel):

    decision: str

    score: float

    reason: str


# ============================================================
# CARE CONTEXT RESPONSE
# ============================================================

class CareContextResponse(BaseModel):

    available: bool

    context_status: Optional[str] = None

    window_start: Optional[str] = None

    window_end: Optional[str] = None

    care_state: dict = Field(
        default_factory=dict
    )

    transition: dict = Field(
        default_factory=dict
    )

    adaptive_assistance: dict = Field(
        default_factory=dict
    )

    assistance_plan: dict = Field(
        default_factory=dict
    )

    interpretation: Optional[str] = None


# ============================================================
# QUESTION RESPONSE
# ============================================================

class QuestionResponse(BaseModel):

    answer: str

    sources: List[str]

    reliability: dict

    decision: DecisionResponse

    care_context: CareContextResponse

    profile_used: bool


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "service": "CareBuddy API",
        "status": "running"
    }


# ============================================================
# ASK ENDPOINT
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

        # ----------------------------------------------------
        # LOAD USER PROFILE FROM DATABASE
        # ----------------------------------------------------

        if request.user_id is not None:

            db_user = (
                db.query(UserProfileDB)
                .filter(
                    UserProfileDB.id == request.user_id
                )
                .first()
            )

            if db_user is None:

                raise HTTPException(
                    status_code=404,
                    detail="User not found"
                )

            request.user_profile = UserProfile(
                patient_id=str(db_user.id),

                age=db_user.age,

                location=db_user.location,

                chronic_conditions=[
                    value.strip()
                    for value in (
                        db_user.chronic_conditions or ""
                    ).split(",")
                    if value.strip()
                ],

                medications=[
                    value.strip()
                    for value in (
                        db_user.medications or ""
                    ).split(",")
                    if value.strip()
                ],

                preferred_language=(
                    db_user.preferred_language
                    or "en"
                ),

                speech_speed=(
                    db_user.speech_speed
                    or "normal"
                )
            )

        # ----------------------------------------------------
        # IMPORT RAG SERVICE
        #
        # Importing here keeps the basic API endpoints
        # available even if the RAG models are unavailable.
        # ----------------------------------------------------

        from scripts.carebuddy_service import answer_question

        # ----------------------------------------------------
        # DETERMINE RESPONSE LANGUAGE
        # ----------------------------------------------------

        response_language = normalize_response_language(

            request.user_profile.preferred_language
            if request.user_profile
            else None

        )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        result = answer_question(

            question=request.question,

            user_profile=request.user_profile,

            conversation_history=request.conversation_history,

            response_language=response_language

        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return result

    # --------------------------------------------------------
    # PRESERVE INTENTIONAL HTTP ERRORS
    # --------------------------------------------------------

    except HTTPException:
        raise

    # --------------------------------------------------------
    # SHOW REAL RAG ERROR DURING DEVELOPMENT
    # --------------------------------------------------------

    except Exception as e:

        print("\n" + "=" * 70)
        print("CARE BUDDY /ASK ERROR")
        print("=" * 70)

        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        traceback.print_exc()

        print("=" * 70 + "\n")

        raise HTTPException(
            status_code=503,
            detail=(
                f"Question-answering service error: "
                f"{str(e)}"
            )
        )

