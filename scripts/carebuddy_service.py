"""
CareBuddy Service Layer

Responsible for:
- Connecting RAG generation with API layer
- Reliability evaluation
- Adaptive decision control
- User profile preparation

Used by:
- FastAPI backend
- Frontend application
- STT/TTS integration
"""


from scripts.rag_chat import generate_answer

from scripts.reliability_evaluation import (
    evaluate_reliability
)

from scripts.adaptive_decision_controller import (
    make_reliability_decision
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

User Profile Information:

"""


    for key, value in user_profile.items():

        context += (
            f"- {key}: {value}\n"
        )



    context += """

Use profile information only when relevant.
Do not invent information.

"""


    return context



# ============================================================
# Main CareBuddy Function
# ============================================================

def answer_question(
    question,
    user_profile=None
):

    """
    Main service function.

    Parameters:

        question:
            User question

        user_profile:
            Optional user information


    Returns:

        {
            answer,
            sources,
            reliability,
            decision,
            profile_used
        }

    """



    # --------------------------------------------------------
    # Add user profile context
    # --------------------------------------------------------

    profile_context = build_profile_context(
        user_profile
    )



    if profile_context:

        enhanced_question = (

            profile_context

            +

            "\nUser Question:\n"

            +

            question

        )


    else:

        enhanced_question = question



    # --------------------------------------------------------
    # Generate RAG Answer
    # --------------------------------------------------------

    answer, evidence = generate_answer(

        enhanced_question,

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

            sources.append(

                source

            )



    # --------------------------------------------------------
    # Final API Response
    # --------------------------------------------------------

    result = {


        "answer": answer,


        "sources": sources,


        "reliability": reliability_report,


        "decision": decision_result,


        "profile_used":

            True

            if user_profile

            else False

    }



    return result