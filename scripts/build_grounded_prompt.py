"""
Evidence-Grounded Prompt Builder

Strict citation-first prompt for CareBuddy.
Forces the LLM to generate only directly supported answers.

Sources are retained internally as evidence metadata but are
not displayed in the user-facing answer.
"""


def build_grounded_prompt(
    query,
    evidence_items,
    reliability,
    decision,
    user_profile=None,
    conversation_context="",
    response_language="en",
    assistance_plan=None
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
    assistance_plan_context = format_assistance_plan_context(assistance_plan)

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
- documents
- source names
- citations

in the answer.

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

Overall Reliability:
{reliability['overall_reliability']:.2f}

Decision:
{decision.get('decision', 'UNKNOWN')}

Decision Reason:
{decision.get('reason', '')}


=====================================================
SOURCE INFORMATION
=====================================================

{evidence_text}


{profile_context}


{conversation_context}


{assistance_plan_context}


=====================================================
USER QUESTION
=====================================================

{query}


=====================================================
ANSWER FORMAT
=====================================================

Provide only the short answer.

Do not add:
- "Answer:"
- "Sources:"
- source names
- document names
- citations
- retrieval information
- reliability information

The response should contain only the natural-language answer
to the user's question.

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

        return getattr(
            user_profile,
            name,
            None
        )

    def format_value(value):
        if isinstance(value, (list, tuple, set)):
            return ", ".join(
                str(item)
                for item in value
                if str(item).strip()
            )

        return (
            str(value).strip()
            if value is not None
            else ""
        )

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
        if (
            formatted := format_value(
                get_value(name)
            )
        )
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
            "Generate the answer naturally in Korean.\n"
            "Use polite Korean suitable for elderly users.\n"
            "Do not add headings such as 'Answer:' or 'Sources:'."
        )

    return (
        "LANGUAGE INSTRUCTION:\n"
        "Generate the answer naturally in English.\n"
        "Use clear and simple language suitable for elderly users.\n"
        "Do not add headings such as 'Answer:' or 'Sources:'."
    )


def format_assistance_plan_context(assistance_plan):
    """
    Render the validated assistance plan (strategy, actions, safety
    constraints) as an internal prompt section.

    The assistance plan is a personalization / response-adaptation signal
    only. It is never evidence and must not be turned into a diagnosis,
    risk claim, or clinical conclusion.
    """
    if not assistance_plan:
        return ""

    strategy = assistance_plan.get("assistance_strategy")
    priority = assistance_plan.get("priority")
    actions = assistance_plan.get("actions") or []
    safety = assistance_plan.get("safety_constraints") or []

    if not strategy and not actions:
        return ""

    lines = [
        "=====================================================",
        "ASSISTANCE PLAN (response adaptation, NOT medical evidence)",
        "=====================================================",
        f"Strategy: {strategy}",
        f"Priority: {priority}" if priority else "Priority: ",
    ]

    if actions:
        lines.append("Assistance actions (system behavior to follow):")
        for action_item in actions:
            action_name = None
            action_reason = ""
            if isinstance(action_item, dict):
                action_name = action_item.get("action")
                action_reason = action_item.get("reason", "")
            else:
                action_name = str(action_item)
            if action_name:
                line_text = f"- {action_name}"
                if action_reason:
                    line_text += f": {action_reason}"
                lines.append(line_text)

    if safety:
        lines.append("Safety constraints (must never be violated):")
        for constraint in safety:
            lines.append(f"- {constraint}")

    lines.append(
        "These actions and constraints describe how to adapt the structure, "
        "tone, and pacing of the response and what to avoid. They are NOT "
        "clinical evidence, diagnoses, risk predictions, or instructions "
        "to claim the patient is stable, improved, or deteriorating."
    )
    lines.append("")

    return "\n".join(lines)