"""
CareBuddy Service Layer

Responsible for:
- Connecting RAG generation with API layer
- Reliability evaluation
- Adaptive decision control
- User profile preparation
- Conversation history support
- Exposing ElderDocAI adaptive care context

Used by:
- FastAPI backend
- Frontend application
- STT/TTS integration
"""

from scripts.rag_chat import (
    generate_answer,
    get_adaptive_context,
    get_assistance_plan,
    prepare_assistance_plan,
    prepare_adaptive_context,
    extract_patient_id
)

from scripts.reliability_evaluation import (
    evaluate_reliability
)

from scripts.adaptive_decision_controller import (
    make_reliability_decision
)


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

            role = message.get(
                "role",
                "user"
            )

            content = message.get(
                "content",
                ""
            )

        else:

            role = message.role

            content = message.content

        context += (
            f"{role.capitalize()}: "
            f"{content}\n"
        )

    context += "\n"

    return context


# ============================================================
# Adaptive Context Preparation
# ============================================================

def prepare_care_context(adaptive_context):
    """
    Convert the internal ElderDocAI adaptive-context record
    into a clean API response object.

    Adaptive context describes documented care activity
    and temporal changes.

    It is NOT a diagnosis and does NOT represent
    medical risk prediction.
    """

    if not adaptive_context:

        return {
            "available": False
        }

    care_state = (
        adaptive_context.get(
            "care_state"
        )
        or {}
    )

    transition = (
        adaptive_context.get(
            "transition"
        )
        or {}
    )

    assistance = (
        adaptive_context.get(
            "adaptive_assistance"
        )
        or {}
    )

    return {

        "available": True,

        "context_status":
            adaptive_context.get(
                "context_status"
            ),

        "window_start":
            adaptive_context.get(
                "window_start"
            ),

        "window_end":
            adaptive_context.get(
                "window_end"
            ),

        "care_state": {

            "state":
                care_state.get(
                    "state"
                ),

            "overall_score":
                care_state.get(
                    "overall_score"
                ),

            "has_documented_activity":
                care_state.get(
                    "has_documented_activity"
                ),

            "event_summary":
                care_state.get(
                    "event_summary"
                )
        },

        "transition": {

            "type":
                transition.get(
                    "type"
                ),

            "direction":
                transition.get(
                    "direction"
                ),

            "magnitude":
                transition.get(
                    "magnitude"
                ),

            "score_delta":
                transition.get(
                    "score_delta"
                ),

            "previous_state":
                transition.get(
                    "previous_state"
                ),

            "current_state":
                transition.get(
                    "current_state"
                )
        },

        "adaptive_assistance": {

            "mode":
                assistance.get(
                    "mode"
                ),

            "priority":
                assistance.get(
                    "priority"
                ),

            "reason_codes":
                assistance.get(
                    "reason_codes",
                    []
                ),

            "reasons":
                assistance.get(
                    "reasons",
                    []
                )
        },

        "interpretation":
            "Adaptive assistance is selected from documented "
            "care-state and transition information. It does "
            "not represent a diagnosis or medical risk prediction."
    }


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
    Main CareBuddy service function.

    Returns:

    {
        answer,
        sources,
        reliability,
        decision,
        care_context,
        profile_used
    }
    """

    # ========================================================
    # Conversation Context
    # ========================================================

    conversation_context = (
        build_conversation_context(
            conversation_history
        )
    )


    # ========================================================
    # Generate RAG Answer
    # ========================================================

    answer, evidence = generate_answer(

        question,

        user_profile=user_profile,

        conversation_context=conversation_context,

        response_language=response_language,

        return_evidence=True
    )


    # ========================================================
    # Reliability Evaluation
    # ========================================================

    reliability_report = (
        evaluate_reliability(

            query=question,

            evidence_items=evidence

        )
    )


    # ========================================================
    # Adaptive Decision
    # ========================================================

    decision_result = (
        make_reliability_decision(

            reliability_report

        )
    )


    # ========================================================
    # Extract Sources
    # ========================================================

    sources = []


    for item in evidence:

        source = item.get(
            "source_document",
            "Unknown"
        )


        if source not in sources:

            sources.append(
                source
            )


    # ========================================================
    # Retrieve Patient Adaptive Context
    # ========================================================

    patient_id = extract_patient_id(
        user_profile
    )


    adaptive_context = (
        get_adaptive_context(

            patient_id=patient_id

        )
    )


    care_context = (
        prepare_care_context(
            adaptive_context
        )
    )

    # --------------------------------------------------------
    # Assistance plan for the same adaptive window
    # --------------------------------------------------------
    assistance_plan_for_api = None

    if adaptive_context and isinstance(care_context, dict):

        plan_record = get_assistance_plan(
            patient_id=patient_id,
            window_start=adaptive_context.get("window_start"),
            window_end=adaptive_context.get("window_end")
        )

        assistance_plan_for_api = prepare_assistance_plan(
            plan_record
        )

        if assistance_plan_for_api:

            care_context["assistance_plan"] = assistance_plan_for_api


    # ========================================================
    # Final API Response
    # ========================================================

    return {

        "answer":
            answer,

        "sources":
            sources,

        "reliability":
            reliability_report,

        "decision":
            decision_result,

        "care_context":
            care_context,

        "profile_used":
            user_profile is not None

    }