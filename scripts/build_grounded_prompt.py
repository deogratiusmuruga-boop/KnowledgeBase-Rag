"""
Evidence-Grounded Prompt Builder

Strict citation-first prompt for CareBuddy.
Forces the LLM to generate only directly supported answers.
"""


def build_grounded_prompt(
    query,
    evidence_items,
    reliability,
    decision,
    user_profile=None,
    conversation_context="",
    response_language="en"
):

    evidence_text = ""


    for i, item in enumerate(
        evidence_items,
        start=1
    ):

        evidence_text += (
            "\n"
            "=====================================================\n"
            f"SOURCE {i}\n"
            "=====================================================\n"
            f"Document: {item['source_document']}\n"
            f"Content:\n"
            f"{item['text']}\n"
            "\n"
        )



    profile_context = format_user_profile_context(user_profile)
    language_instruction = build_language_instruction(response_language)

    prompt = f"""

You are CareBuddy, an elderly-care RAG assistant.

Your task is to answer the user's question using ONLY
the provided SOURCE information.

You do NOT have permission to use your own medical knowledge.

=====================================================
STRICT FAITHFULNESS RULES
=====================================================

RULE 1:
Every sentence in your answer MUST be directly supported
by the SOURCE information.

RULE 2:
Do not add explanations, recommendations, warnings,
or examples that are not explicitly written in SOURCE.

RULE 3:
Do not combine information from different sources to
create a new conclusion.

RULE 4:
Do not generalize.

Bad:
"Exercise improves quality of life."

Good:
"Older adults should include balance training."

RULE 5:
If information is missing, say:

"I couldn't find that information in the knowledge base."

RULE 6:
Keep answers short.
Only include facts required to answer the question.

RULE 7:
Do not mention:
- SOURCE numbers
- retrieval
- evidence
- documents inside the answer

Only provide Sources at the end.

RULE 8:
User profile information is only for personalization and response adaptation. It is not evidence and must not be used as a source for medical claims.
User profile information must never override the SOURCE information.
When relevant, adapt language and response pacing to the profile, without adding facts not supported by SOURCE.

{language_instruction}


=====================================================
RELIABILITY INFORMATION
=====================================================

Authority:
{reliability['authority']:.2f}

Relevance:
{reliability['relevance']:.2f}

Support:
{reliability['support']:.2f}

Coverage:
{reliability['coverage']:.2f}

Consistency:
{reliability['consistency']:.2f}


=====================================================
SOURCE INFORMATION
=====================================================

{evidence_text}


{profile_context}


{conversation_context}


=====================================================
USER QUESTION
=====================================================

{query}


=====================================================
ANSWER FORMAT
=====================================================

Answer:

(short answer using only supported facts)

Sources:
- document_name.pdf


Remember:

No outside knowledge.
No assumptions.
No extra advice.
No medical interpretation.

"""


    return prompt


def format_user_profile_context(user_profile):
    """Format optional profile data as generation-only, non-evidence context."""
    if user_profile is None:
        return ""

    def get_value(name):
        if isinstance(user_profile, dict):
            return user_profile.get(name)
        return getattr(user_profile, name, None)

    def format_value(value):
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if str(item).strip())
        return str(value).strip() if value is not None else ""

    fields = (
        ("Age", "age"),
        ("Location", "location"),
        ("Chronic conditions", "chronic_conditions"),
        ("Medications", "medications"),
        ("Preferred language", "preferred_language"),
        ("Speech speed", "speech_speed"),
    )
    lines = [
        f"- {label}: {formatted}"
        for label, name in fields
        if (formatted := format_value(get_value(name)))
    ]

    if not lines:
        return ""

    return (
        "=====================================================\n"
        "USER PROFILE CONTEXT\n"
        "=====================================================\n\n"
        + "\n".join(lines)
        + "\n\nUser profile information is only for personalization and response adaptation. "
        "It is not evidence and must not be used as a source for medical claims.\n"
    )


def build_language_instruction(response_language):
    """Return an explicit generation instruction for the supported languages."""
    if response_language == "ko":
        return (
            "LANGUAGE INSTRUCTION:\n"
            "Generate the answer body naturally in Korean.\n"
            "Use polite Korean suitable for elderly users.\n"
            "Keep the final heading exactly as: Sources:"
        )

    return (
        "LANGUAGE INSTRUCTION:\n"
        "Generate the answer body naturally in English.\n"
        "Use clear and simple language suitable for elderly users.\n"
        "Keep the final heading exactly as: Sources:"
    )
