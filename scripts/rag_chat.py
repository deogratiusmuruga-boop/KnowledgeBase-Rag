"""
ElderDocAI / CareBuddy RAG Generation Module

Pipeline:

    User Query
        ↓
    Adaptive Context
        ↓
    Hybrid Retrieval
        ↓
    Evidence Preparation
        ↓
    Reliability Evaluation
        ↓
    Adaptive Decision
        ↓
    Grounded Prompt
        ↓
    Llama 3.2
        ↓
    Evidence-Grounded Response

Adaptive context contains:
    - patient profile
    - current care state
    - temporal transition
    - changed dimensions
    - adaptive assistance

Important:
    Adaptive context describes documented care activity and temporal
    changes. It is not a diagnosis and does not predict medical risk.
"""

import json
from pathlib import Path

import ollama

from scripts.hybrid_retriever import hybrid_search
from scripts.build_grounded_prompt import build_grounded_prompt
from scripts.reliability_evaluation import evaluate_reliability
from scripts.adaptive_decision_controller import make_reliability_decision


# ============================================================
# Configuration
# ============================================================

LLM_MODEL = "llama3.2:latest"


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ADAPTIVE_CONTEXT_FILE = (
    BASE_DIR
    / "datasets"
    / "synthea"
    / "elderdocai"
    / "processed"
    / "adaptive_context.json"
)

ASSISTANCE_PLAN_FILE = (
    BASE_DIR
    / "datasets"
    / "synthea"
    / "elderdocai"
    / "processed"
    / "assistance_plans.json"
)

ASSISTANCE_DECISION_FILE = (
    BASE_DIR
    / "datasets"
    / "synthea"
    / "elderdocai"
    / "processed"
    / "assistance_decisions.json"
)


# ============================================================
# Adaptive Context Loading
# ============================================================

print("Loading ElderDocAI adaptive context...")

try:

    with open(
        ADAPTIVE_CONTEXT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        adaptive_context_records = json.load(f)

except FileNotFoundError:

    print(
        "WARNING: adaptive_context.json was not found."
    )

    adaptive_context_records = []


print(
    f"Adaptive context records loaded: "
    f"{len(adaptive_context_records)}"
)


# ============================================================
# Build Patient Index
# ============================================================

adaptive_context_by_patient = {}


for record in adaptive_context_records:

    patient_id = record.get("patient_id")

    if not patient_id:
        continue

    adaptive_context_by_patient.setdefault(
        str(patient_id),
        []
    ).append(record)


# Keep chronological ordering
for patient_id in adaptive_context_by_patient:

    adaptive_context_by_patient[patient_id].sort(
        key=lambda x: (
            x.get("window_start", ""),
            x.get("window_end", "")
        )
    )


print(
    f"Patients with adaptive context: "
    f"{len(adaptive_context_by_patient)}"
)


# ============================================================
# Assistance Plan + Decision Loading
# ============================================================
# The validated assistance-plan and assistance-decision layers describe
# the response behavior (strategy, actions, safety constraints). They are
# loaded alongside adaptive context and joined by
# (patient_id, window_start, window_end) so the RAG layer honors them.

print("Loading ElderDocAI assistance decisions and plans...")

try:
    with open(
        ASSISTANCE_DECISION_FILE,
        "r",
        encoding="utf-8"
    ) as f_dec:

        assistance_decision_records = json.load(f_dec)

except FileNotFoundError:

    print(
        "WARNING: assistance_decisions.json was not found."
    )

    assistance_decision_records = []

try:
    with open(
        ASSISTANCE_PLAN_FILE,
        "r",
        encoding="utf-8"
    ) as f_plan:

        assistance_plan_records = json.load(f_plan)

except FileNotFoundError:

    print(
        "WARNING: assistance_plans.json was not found."
    )

    assistance_plan_records = []

# ------------------------------------------------------------------
# Window-keyed indexes for plans and decisions
# ------------------------------------------------------------------
assistance_plan_by_window = {}     # (pid, window_start, window_end) -> plan
assistance_decision_by_window = {} # (pid, window_start, window_end) -> decision

for record in assistance_plan_records:

    patient_id = record.get("patient_id")
    window_start = record.get("window_start")
    window_end = record.get("window_end")

    if not patient_id or window_start is None or window_end is None:
        continue

    assistance_plan_by_window[
        (str(patient_id), window_start, window_end)
    ] = record

for record in assistance_decision_records:

    patient_id = record.get("patient_id")
    window_start = record.get("window_start")
    window_end = record.get("window_end")

    if not patient_id or window_start is None or window_end is None:
        continue

    assistance_decision_by_window[
        (str(patient_id), window_start, window_end)
    ] = record

print(
    f"Assistance plans loaded: "
    f"{len(assistance_plan_by_window)}"
)

print(
    f"Assistance decisions loaded: "
    f"{len(assistance_decision_by_window)}"
)


# ============================================================
# Adaptive Context Lookup
# ============================================================

def get_adaptive_context(
    patient_id=None,
    context_date=None
):
    """
    Retrieve the appropriate adaptive-context window.

    Selection priority:

    1. Exact window containing context_date
    2. Latest non-DATA_GAP window
    3. Latest available window
    """

    if not patient_id:
        return None

    records = adaptive_context_by_patient.get(
        str(patient_id),
        []
    )

    if not records:
        return None

    # --------------------------------------------------------
    # Exact date-based lookup
    # --------------------------------------------------------

    if context_date:

        context_date = str(context_date)

        for record in records:

            start = record.get(
                "window_start",
                ""
            )

            end = record.get(
                "window_end",
                ""
            )

            if start <= context_date <= end:
                return record

    # --------------------------------------------------------
    # Otherwise use latest window containing data
    # --------------------------------------------------------

    usable = [
        record
        for record in records
        if record.get("context_status") != "DATA_GAP"
    ]

    if usable:
        return usable[-1]

    # --------------------------------------------------------
    # Last resort
    # --------------------------------------------------------

    return records[-1]


# ============================================================
# Assistance Plan + Decision Lookup
# ============================================================

def get_assistance_plan(
    patient_id=None,
    window_start=None,
    window_end=None
):
    """
    Retrieve the validated assistance-plan record for an exact
    (patient_id, window_start, window_end) window.

    Returns None when no matching plan exists.
    """
    if not patient_id or window_start is None or window_end is None:
        return None

    return assistance_plan_by_window.get(
        (str(patient_id), window_start, window_end)
    )


def get_assistance_decision(
    patient_id=None,
    window_start=None,
    window_end=None
):
    """
    Retrieve the validated assistance-decision record for an exact
    (patient_id, window_start, window_end) window.

    Returns None when no matching decision exists.
    """
    if not patient_id or window_start is None or window_end is None:
        return None

    return assistance_decision_by_window.get(
        (str(patient_id), window_start, window_end)
    )


def prepare_assistance_plan(plan):
    """
    Convert an internal assistance-plan record into the compact shape
    consumed by the grounded-prompt builder.

    Returns an empty dict when no plan is available so the prompt
    builder can fall back gracefully.
    """
    if not plan:
        return {}

    return {
        "assistance_strategy": plan.get(
            "assistance_strategy"
        ),
        "priority": plan.get(
            "priority"
        ),
        "actions": plan.get(
            "actions",
            []
        ),
        "safety_constraints": plan.get(
            "safety_constraints",
            []
        ),
    }


# ============================================================
# Extract Patient ID
# ============================================================

def extract_patient_id(user_profile):
    """
    Extract patient identifier from either:

    - a dictionary
    - a Pydantic model
    - another object exposing the relevant attribute

    This is important because the FastAPI layer passes a
    Pydantic UserProfile object, while direct Python tests
    may pass a dictionary.
    """

    if not user_profile:
        return None

    possible_fields = [
        "patient_id",
        "user_id",
        "id"
    ]

    for field in possible_fields:

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(user_profile, dict):

            value = user_profile.get(field)

        # ----------------------------------------------------
        # Pydantic model / object
        # ----------------------------------------------------

        else:

            value = getattr(
                user_profile,
                field,
                None
            )

        if value is not None:
            return str(value)

    return None


# ============================================================
# Prepare Adaptive Context
# ============================================================

def prepare_adaptive_context(context):

    if not context:

        return {
            "available": False
        }

    care_state = (
        context.get("care_state")
        or {}
    )

    transition = (
        context.get("transition")
        or {}
    )

    assistance = (
        context.get("adaptive_assistance")
        or {}
    )

    return {

        "available": True,

        "context_status":
            context.get(
                "context_status"
            ),

        "window_start":
            context.get(
                "window_start"
            ),

        "window_end":
            context.get(
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

        "changed_dimensions":
            context.get(
                "changed_dimensions",
                []
            ),

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
                ),

            "supporting_evidence":
                transition.get(
                    "supporting_evidence"
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
            context.get(
                "interpretation",
                (
                    "Adaptive assistance is selected from "
                    "documented care-state and transition "
                    "information. It does not represent a "
                    "diagnosis or medical risk prediction."
                )
            )

    }


# ============================================================
# Evidence Preparation
# ============================================================

def prepare_evidence(chunks):
    """
    Convert hybrid-retrieval results into the evidence format
    required by the reliability evaluator and prompt builder.
    """

    evidence_items = []

    for chunk in chunks:

        evidence_items.append(

            {
                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "source_document":
                    chunk.get(
                        "source_document",
                        "Unknown"
                    ),

                "document_category":
                    chunk.get(
                        "category",
                        chunk.get(
                            "document_category",
                            "Unknown"
                        )
                    ),

                "authority_score":
                    chunk.get(
                        "authority_score",
                        1.0
                    ),

                "similarity_score":
                    chunk.get(
                        "rerank_score",
                        chunk.get(
                            "dense_score",
                            chunk.get(
                                "similarity_score",
                                0.0
                            )
                        )
                    ),

                "text":
                    chunk.get(
                        "text",
                        ""
                    )
            }
        )

    return evidence_items


# ============================================================
# Reliability Report Printer
# ============================================================

def print_reliability_report(
    reliability,
    decision
):

    print("\n")
    print("-" * 70)
    print("RELIABILITY REPORT")
    print("-" * 70)

    print(
        f"  Authority:           "
        f"{reliability['authority']:.2f}"
    )

    print(
        f"  Relevance:           "
        f"{reliability['relevance']:.2f}"
    )

    print(
        f"  Support:             "
        f"{reliability['support']:.2f}"
    )

    print(
        f"  Coverage:            "
        f"{reliability['coverage']:.2f}"
    )

    print(
        f"  Consistency:         "
        f"{reliability['consistency']:.2f}"
    )

    print(
        f"  Overall Reliability: "
        f"{reliability['overall_reliability']:.2f}"
    )

    print(
        f"  Decision:            "
        f"{decision['decision']}"
    )

    print(
        f"  Reason:              "
        f"{decision['reason']}"
    )

    print("-" * 70)


# ============================================================
# Generate Answer
# ============================================================

def generate_answer(
    query,
    user_profile=None,
    conversation_context="",
    response_language="en",
    return_evidence=False
):

    print("\n" + "=" * 70)
    print("ELDERDOCAI / CAREBUDDY RAG")
    print("=" * 70)

    # ========================================================
    # Patient / Adaptive Context
    # ========================================================

    patient_id = extract_patient_id(
        user_profile
    )

    adaptive_context = get_adaptive_context(
        patient_id=patient_id
    )

    adaptive_context_for_prompt = (
        prepare_adaptive_context(
            adaptive_context
        )
    )

    # --------------------------------------------------------
    # Assistance plan + decision lookup for the same window
    # --------------------------------------------------------
    assistance_plan = None
    assistance_plan_for_prompt = None

    if adaptive_context:

        assistance_plan = get_assistance_plan(
            patient_id=patient_id,
            window_start=adaptive_context.get("window_start"),
            window_end=adaptive_context.get("window_end")
        )

        assistance_plan_for_prompt = (
            prepare_assistance_plan(
                assistance_plan
            )
        )

    print("\nAdaptive Context:")

    if adaptive_context:

        print(
            f"  patient_id: "
            f"{patient_id}"
        )

        print(
            f"  window: "
            f"{adaptive_context.get('window_start')} "
            f"to "
            f"{adaptive_context.get('window_end')}"
        )

        print(
            f"  context_status: "
            f"{adaptive_context.get('context_status')}"
        )

        care_state = (
            adaptive_context.get(
                "care_state"
            )
            or {}
        )

        print(
            f"  care_state: "
            f"{care_state.get('state')}"
        )

        print(
            f"  score: "
            f"{care_state.get('overall_score')}"
        )

        assistance = (
            adaptive_context.get(
                "adaptive_assistance"
            )
            or {}
        )

        print(
            f"  assistance: "
            f"{assistance.get('mode')}"
        )

        print(
            f"  priority: "
            f"{assistance.get('priority')}"
        )

        print(
            f"  assistance_strategy: "
            f"{assistance_plan_for_prompt.get('assistance_strategy')}"
        )

        print(
            f"  assistance actions: "
            f"{[a.get('action') for a in assistance_plan_for_prompt.get('actions', [])]}"
        )

    else:

        print(
            "  No adaptive context available."
        )

    # ========================================================
    # Hybrid Retrieval
    # ========================================================

    print("\nRetrieving evidence...\n")

    retrieved_chunks = hybrid_search(
        query
    )

    # ========================================================
    # No Evidence
    # ========================================================

    if not retrieved_chunks:

        answer = (
            "I couldn't find that information "
            "in the knowledge base."
        )

        if return_evidence:

            return (
                answer,
                []
            )

        return answer

    # ========================================================
    # Evidence Preparation
    # ========================================================

    evidence_items = prepare_evidence(
        retrieved_chunks
    )

    # ========================================================
    # Reliability Evaluation
    # ========================================================

    reliability = evaluate_reliability(
        query=query,
        evidence_items=evidence_items
    )

    # ========================================================
    # Adaptive Decision
    # ========================================================

    decision = make_reliability_decision(
        reliability
    )

    # ========================================================
    # Reliability Report
    # ========================================================

    print_reliability_report(
        reliability,
        decision
    )

    # ========================================================
    # Build Grounded Prompt
    # ========================================================

    prompt = build_grounded_prompt(

        query=query,

        evidence_items=evidence_items,

        reliability=reliability,

        decision=decision,

        user_profile=user_profile,

        conversation_context=conversation_context,

        response_language=response_language,

        assistance_plan=assistance_plan_for_prompt
    )

    # ========================================================
    # LLM Generation
    # ========================================================

    response = ollama.chat(

        model=LLM_MODEL,

        options={

            "temperature": 0,

            "top_p": 0.1,

            "top_k": 10

        },

        messages=[

            {
                "role": "system",

                "content": """
You are CareBuddy,
an evidence-grounded elderly-care assistant.

You have two information sources:

1. ADAPTIVE CONTEXT
   - Describes documented care activity,
     temporal changes, and selected assistance strategy.
   - It does NOT diagnose disease.
   - It does NOT predict medical risk.
   - It must not be treated as a medical diagnosis.

2. RETRIEVED KNOWLEDGE
   - Provides evidence from the CareBuddy knowledge base.
   - Answers must be grounded in this evidence.

STRICT RULES:

1. Use only the provided adaptive context and retrieved evidence.

2. Do not use outside medical knowledge.

3. Do not guess.

4. Do not invent facts about the patient.

5. Do not diagnose.

6. Do not predict disease progression,
   medical risk, hospitalization, or mortality.

7. If the retrieved evidence is insufficient
   to answer the user's question, say:

   I couldn't find that information in the knowledge base.

8. Use the adaptive context to personalize
   the structure and relevance of the response,
   but never turn care-state labels into diagnoses.

9. Keep answers short, clear, and factual.

10. When adaptive assistance is available,
    follow its assistance mode and priority
    while remaining grounded in retrieved evidence.

11. Do not include a Sources section in the
    user-facing answer.

12. Do not mention retrieval, evidence scores,
    reliability scores, internal decisions,
    care-state computation, or system architecture
    unless explicitly asked.

Return only the answer itself.
"""
            },

            {
                "role": "user",

                "content": prompt
            }

        ]

    )

    # ========================================================
    # Extract Answer
    # ========================================================

    answer = (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )

    # ========================================================
    # Remove Accidental Sources Section
    # ========================================================

    if "\nSources:" in answer:

        answer = answer.split(
            "\nSources:",
            1
        )[0].strip()

    elif answer.startswith("Sources:"):

        answer = ""

    # ========================================================
    # Remove Accidental Answer Heading
    # ========================================================

    if answer.startswith("Answer:"):

        answer = answer[
            len("Answer:"):
        ].strip()

    # ========================================================
    # Debug
    # ========================================================

    print("\nLLM RESPONSE OBJECT:")

    print(response)

    # ========================================================
    # Return
    # ========================================================

    if return_evidence:

        return (
            answer,
            evidence_items
        )

    return answer