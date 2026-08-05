"""
CareBuddy Service Layer

Responsible for:
- Connecting RAG generation with API layer
- Reliability evaluation
- Adaptive decision control
- User profile preparation
- Conversation history support

Used by:
- FastAPI backend
- Frontend application
- STT/TTS integration
"""

from scripts.rag_chat import generate_answer
from scripts.reliability_evaluation import evaluate_reliability
from scripts.adaptive_decision_controller import make_reliability_decision


# ============================================================
# Conversation History Context Builder
# ============================================================

def build_conversation_context(conversation_history):
    """
    Convert previous conversation into text
    for the RAG prompt.
    """

    if not conversation_history:
        return ""

    context = """

Previous Conversation:

"""

    for message in conversation_history:

        if isinstance(message, dict):

            role = message.get("role", "user")
            content = message.get("content", "")

        else:

            role = message.role
            content = message.content

        context += f"{role.capitalize()}: {content}\n"

    context += "\n"

    return context


# ============================================================
# Main CareBuddy Function
# ============================================================

def answer_question(
    question,
    user_profile=None,
    conversation_history=None,
    response_language="en"
):
    """
    Main service function.

    Returns:

    {
        answer,
        sources,
        reliability,
        decision,
        profile_used
    }
    """

    conversation_context = build_conversation_context(
        conversation_history
    )

    # --------------------------------------------------------
    # Generate RAG answer
    # --------------------------------------------------------

    answer, evidence = generate_answer(
        question,
        user_profile=user_profile,
        conversation_context=conversation_context,
        response_language=response_language,
        return_evidence=True
    )

    # --------------------------------------------------------
    # Reliability Evaluation
    # --------------------------------------------------------

    reliability_report = evaluate_reliability(
        query=question,
        evidence_items=evidence
    )

    # --------------------------------------------------------
    # Adaptive Decision
    # --------------------------------------------------------

    decision_result = make_reliability_decision(
        reliability_report
    )

    # --------------------------------------------------------
    # Extract Sources
    # --------------------------------------------------------

    sources = []

    for item in evidence:

        source = item.get(
            "source_document",
            "Unknown"
        )

        if source not in sources:
            sources.append(source)

    # --------------------------------------------------------
    # Final API Response
    # --------------------------------------------------------

    return {

        "answer": answer,

        "sources": sources,

        "reliability": reliability_report,

        "decision": decision_result,

        "profile_used": user_profile is not None

    }
