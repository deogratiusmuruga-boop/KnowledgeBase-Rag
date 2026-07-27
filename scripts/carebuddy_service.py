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
# Build Personalized Query
# ============================================================

def build_personalized_query(
    question,
    user_profile=None
):
    """
    Enrich the user's question with profile information
    to improve retrieval.
    """

    if user_profile is None:
        return question

    profile_parts = []

    if getattr(user_profile, "age", None):
        profile_parts.append(
            f"Age: {user_profile.age}"
        )

    if getattr(user_profile, "location", None):
        profile_parts.append(
            f"Location: {user_profile.location}"
        )

    if getattr(user_profile, "chronic_conditions", None):
        profile_parts.append(
            "Conditions: "
            + ", ".join(user_profile.chronic_conditions)
        )

    if getattr(user_profile, "medications", None):
        profile_parts.append(
            "Medications: "
            + ", ".join(user_profile.medications)
        )

    if not profile_parts:
        return question

    return (
        question
        + "\n\nUser Profile:\n"
        + "\n".join(profile_parts)
    )


# ============================================================
# User Profile Context Builder
# ============================================================

def build_profile_context(user_profile):
    """
    Convert user profile information
    into optional RAG context.
    """

    if not user_profile:
        return ""

    context = """

User Context:

"""

    for key, value in user_profile.model_dump().items():

        if value is None:
            continue

        if value == []:
            continue

        context += f"- {key}: {value}\n"

    context += """

Use profile information only when relevant.
Do not invent information.

"""

    return context


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
    conversation_history=None
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

    print("\n========== USER PROFILE DEBUG ==========")

    if user_profile:
        print(user_profile.model_dump())
    else:
        print("No user profile received")

    print("========================================\n")


    # --------------------------------------------------------
    # Build personalized question
    # --------------------------------------------------------
    

    # --------------------------------------------------------
    # Build personalized question
    # --------------------------------------------------------

    enhanced_question = build_personalized_query(
        question,
        user_profile
    )

    # --------------------------------------------------------
    # Build contexts
    # --------------------------------------------------------

    profile_context = build_profile_context(
        user_profile
    )

    conversation_context = build_conversation_context(
        conversation_history
    )

    # --------------------------------------------------------
    # Build final prompt
    # --------------------------------------------------------

    final_prompt = ""

    if profile_context:
        final_prompt += profile_context

    if conversation_context:
        final_prompt += conversation_context

    final_prompt += (
        "\nCurrent User Question:\n"
        + enhanced_question
    )

    # --------------------------------------------------------
    # Generate RAG answer
    # --------------------------------------------------------

    answer, evidence = generate_answer(
        final_prompt,
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