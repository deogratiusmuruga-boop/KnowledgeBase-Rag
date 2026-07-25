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
    "question": "..."
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


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel



# ============================================================
# Add project root
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.append(
    BASE_DIR
)



# ============================================================
# Import CareBuddy Service
# ============================================================

from scripts.carebuddy_service import answer_question



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

    allow_headers=[

        "*"

    ]

)



# ============================================================
# Request Schema
# ============================================================

class QuestionRequest(BaseModel):

    question: str



# ============================================================
# Reliability Schema
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

    sources: list[str]

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
    request: QuestionRequest
):

    try:

        result = answer_question(
            request.question
        )


        return result



    except Exception as e:


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )