"""
ElderDocAI - Assistance Planner / Action Layer

Converts validated assistance-strategy decisions into a small,
deterministic set of concrete, machine-readable assistance actions.

Pipeline position:

    Assistance Decision
            ↓
    ASSISTANCE PLANNER
            ↓
    Concrete Assistance Actions
            ↓
    Later: RAG / Evidence-Grounded Response

Design:
    - One plan record per assistance-decision window.
    - Primary input: assistance_decisions.json.
    - Upstream fields are reused, never recomputed.
    - Deterministic strategy-to-action mapping.
    - No ML.
    - No LLM.
    - Actions describe system behavior, not medical status.

Safety scope:
    - No disease diagnosis.
    - No medical-risk prediction.
    - No deterioration prediction.
    - No mortality prediction.
    - No hospitalization prediction.
    - NO_DATA does not mean STABLE.
    - Actions are based only on documented care activity and temporal context.
"""

import csv
import json
from collections import Counter
from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "elderdocai" / "processed"

INPUT_JSON = PROCESSED / "assistance_decisions.json"

OUTPUT_JSON = PROCESSED / "assistance_plans.json"
OUTPUT_CSV = PROCESSED / "assistance_plans.csv"


# ============================================================================
# ACTION VOCABULARY
# ============================================================================
#
# These actions describe what the SYSTEM should do.
# They do not describe a medical condition.
#
# ============================================================================

ALL_ACTIONS = [
    "REQUEST_CHECK_IN",
    "REQUEST_DATA_UPDATE",
    "PROVIDE_CONTEXTUAL_INFORMATION",
    "PROVIDE_TARGETED_INFORMATION",
    "PROVIDE_GENERAL_GUIDANCE",
    "REVIEW_RECENT_CARE_CONTEXT",
    "CONTINUE_MONITORING",
    "ENCOURAGE_FUTURE_CHECK_IN",
    "ENCOURAGE_APPROPRIATE_FOLLOW_UP",
    "REDUCE_INTERVENTION_INTENSITY",
    "AVOID_STRONG_PERSONALIZATION",
    "ESTABLISH_INITIAL_CONTEXT",
]


# ============================================================================
# STRATEGY → ACTION RULES
# ============================================================================
#
# Each assistance strategy has a deterministic set of actions.
#
# The planner does NOT recalculate the strategy.
# It simply translates the already validated strategy into actions.
#
# ============================================================================

STRATEGY_ACTIONS = {

    # ------------------------------------------------------------------------
    # DATA GAP
    # ------------------------------------------------------------------------
    "DATA_COLLECTION_SUPPORT": [
        (
            "REQUEST_CHECK_IN",
            "Current documented information is missing; invite a check-in "
            "to gather an update.",
        ),
        (
            "REQUEST_DATA_UPDATE",
            "Current documented information is missing; ask the user to "
            "provide or confirm data.",
        ),
        (
            "AVOID_STRONG_PERSONALIZATION",
            "Information is incomplete; avoid strong personalization based "
            "on missing data.",
        ),
    ],

    # ------------------------------------------------------------------------
    # INITIAL STATE
    # ------------------------------------------------------------------------
    "ONBOARDING_SUPPORT": [
        (
            "ESTABLISH_INITIAL_CONTEXT",
            "First observed care-state window; establish initial assistance "
            "context without medical assumptions.",
        ),
        (
            "REQUEST_CHECK_IN",
            "First observed window; encourage an initial check-in.",
        ),
        (
            "PROVIDE_GENERAL_GUIDANCE",
            "First observed window; introduce general assistance options.",
        ),
    ],

    # ------------------------------------------------------------------------
    # SAME-STATE INCREASING ACTIVITY
    # ------------------------------------------------------------------------
    "MONITORING_SUPPORT": [
        (
            "CONTINUE_MONITORING",
            "Documented care activity increased within the same state; "
            "continue observing future changes.",
        ),
        (
            "REVIEW_RECENT_CARE_CONTEXT",
            "Review the recent documented care context to remain aware "
            "of changes.",
        ),
        (
            "ENCOURAGE_FUTURE_CHECK_IN",
            "Increased documented activity observed; encourage a future "
            "check-in.",
        ),
    ],

    # ------------------------------------------------------------------------
    # SAME-STATE DECREASING ACTIVITY
    # ------------------------------------------------------------------------
    "FOLLOW_UP_SUPPORT": [
        (
            "REVIEW_RECENT_CARE_CONTEXT",
            "Documented care activity decreased within the same state; "
            "review recent context.",
        ),
        (
            "ENCOURAGE_APPROPRIATE_FOLLOW_UP",
            "Decreased documented activity; encourage appropriate "
            "follow-up without inferring improvement.",
        ),
        (
            "ENCOURAGE_FUTURE_CHECK_IN",
            "Decreased documented activity; encourage a future check-in.",
        ),
    ],

    # ------------------------------------------------------------------------
    # STATE ESCALATION
    # ------------------------------------------------------------------------
    "ENHANCED_CONTEXT_SUPPORT": [
        (
            "PROVIDE_TARGETED_INFORMATION",
            "Documented care activity moved to a higher state; provide "
            "targeted information about the documented change.",
        ),
        (
            "REVIEW_RECENT_CARE_CONTEXT",
            "Higher documented activity state; review recent care context.",
        ),
        (
            "REQUEST_CHECK_IN",
            "Higher documented activity state; invite a check-in.",
        ),
        (
            "ENCOURAGE_APPROPRIATE_FOLLOW_UP",
            "Higher documented activity state; encourage appropriate "
            "follow-up.",
        ),
    ],

    # ------------------------------------------------------------------------
    # STATE DEESCALATION
    # ------------------------------------------------------------------------
    "ADAPTIVE_DEESCALATION_SUPPORT": [
        (
            "REDUCE_INTERVENTION_INTENSITY",
            "Documented care activity moved to a lower state; reduce "
            "unnecessary intervention intensity.",
        ),
        (
            "PROVIDE_GENERAL_GUIDANCE",
            "Lower documented activity state; provide general guidance.",
        ),
        (
            "CONTINUE_MONITORING",
            "Lower documented activity state; continue monitoring future "
            "check-ins.",
        ),
    ],

    # ------------------------------------------------------------------------
    # LOW ACTIVITY / NO CHANGE
    # ------------------------------------------------------------------------
    "LIGHT_SUPPORT": [
        (
            "PROVIDE_GENERAL_GUIDANCE",
            "Documented care activity remained low; provide general "
            "guidance.",
        ),
        (
            "ENCOURAGE_FUTURE_CHECK_IN",
            "Low documented activity state; encourage a future check-in.",
        ),
    ],

    # ------------------------------------------------------------------------
    # MODERATE ACTIVITY / NO CHANGE
    # ------------------------------------------------------------------------
    "CONTEXTUAL_SUPPORT": [
        (
            "PROVIDE_CONTEXTUAL_INFORMATION",
            "Documented care activity remained moderate; provide "
            "contextual information.",
        ),
        (
            "ENCOURAGE_FUTURE_CHECK_IN",
            "Moderate documented activity state; encourage a future "
            "check-in.",
        ),
    ],

    # ------------------------------------------------------------------------
    # HIGH ACTIVITY / NO CHANGE
    # ------------------------------------------------------------------------
    "ENHANCED_SUPPORT": [
        (
            "PROVIDE_TARGETED_INFORMATION",
            "Documented care activity remained high; provide targeted "
            "information.",
        ),
        (
            "REVIEW_RECENT_CARE_CONTEXT",
            "High documented activity state; review recent care context.",
        ),
        (
            "ENCOURAGE_FUTURE_CHECK_IN",
            "High documented activity state; encourage a future "
            "check-in.",
        ),
    ],
}


# ============================================================================
# SAFETY CONSTRAINTS
# ============================================================================

BASE_SAFETY = [
    "NO_DIAGNOSIS",
    "NO_MEDICAL_RISK_PREDICTION",
]


INTERPRETATION = (
    "Rule-based assistance plan derived from documented care activity and "
    "temporal changes. This output does not constitute a diagnosis and does "
    "not predict medical risk, deterioration, mortality, or hospitalization."
)


# ============================================================================
# LOAD JSON
# ============================================================================

def _load_json(path):
    """Load a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ============================================================================
# BUILD ONE PLAN RECORD
# ============================================================================

def _build_record(decision):
    """
    Convert one validated assistance-decision record
    into one assistance-plan record.
    """

    strategy = decision["assistance_strategy"]

    action_specs = STRATEGY_ACTIONS.get(strategy)

    # ------------------------------------------------------------------------
    # Safe deterministic fallback
    # ------------------------------------------------------------------------

    if action_specs is None:
        actions = [
            {
                "action": "PROVIDE_GENERAL_GUIDANCE",
                "reason": (
                    "Provide general guidance based on documented care "
                    "activity without medical claims."
                ),
            },
            {
                "action": "ENCOURAGE_FUTURE_CHECK_IN",
                "reason": (
                    "Encourage a future check-in to keep the care context "
                    "current."
                ),
            },
        ]

    else:
        actions = [
            {
                "action": action_name,
                "reason": reason,
            }
            for action_name, reason in action_specs
        ]

    # ------------------------------------------------------------------------
    # Preserve upstream safety constraints
    # ------------------------------------------------------------------------

    safety = list(decision.get("safety_constraints") or [])

    for constraint in BASE_SAFETY:
        if constraint not in safety:
            safety.append(constraint)

    # ------------------------------------------------------------------------
    # Explicit NO_DATA safety constraint
    # ------------------------------------------------------------------------

    if (
        strategy == "DATA_COLLECTION_SUPPORT"
        and "NO_DATA_IS_NOT_STABILITY" not in safety
    ):
        safety.append("NO_DATA_IS_NOT_STABILITY")

    # ------------------------------------------------------------------------
    # Build final record
    # ------------------------------------------------------------------------

    return {
        "patient_id": decision["patient_id"],
        "window_start": decision["window_start"],
        "window_end": decision["window_end"],
        "year": decision.get("year"),

        # Reused upstream context
        "context_status": decision["context_status"],
        "current_state": decision.get("care_state"),
        "overall_score": decision.get("overall_score"),

        # Reused upstream transition information
        "transition_type": decision.get("transition_type"),
        "transition_direction": decision.get("transition_direction"),

        # Reused upstream strategy
        "assistance_strategy": strategy,
        "priority": decision.get("priority"),

        # Newly generated planner information
        "actions": actions,

        # Safety
        "safety_constraints": safety,

        # Scope interpretation
        "interpretation": INTERPRETATION,
    }


# ============================================================================
# BUILD PIPELINE
# ============================================================================

def build():

    print("=" * 70)
    print("ELDERDOCAI - ASSISTANCE PLANNER BUILDER")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------------------

    print("\nLoading assistance decisions...")

    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_JSON}"
        )

    decisions = _load_json(INPUT_JSON)

    if not isinstance(decisions, list):
        raise ValueError(
            "assistance_decisions.json must contain a list of records."
        )

    print(
        f"  assistance_decisions records: {len(decisions)}"
    )

    # ------------------------------------------------------------------------
    # Build plans
    # ------------------------------------------------------------------------

    records = [
        _build_record(decision)
        for decision in decisions
    ]

    # ------------------------------------------------------------------------
    # Record-count integrity check
    # ------------------------------------------------------------------------

    if len(records) != len(decisions):
        raise RuntimeError(
            "Planner record count does not match assistance-decision "
            "record count."
        )

    # ------------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------------

    print("\nSaving JSON...")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(
            records,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # CSV fields
    # ------------------------------------------------------------------------

    csv_fields = [
        "patient_id",
        "window_start",
        "window_end",
        "year",
        "context_status",
        "current_state",
        "overall_score",
        "transition_type",
        "transition_direction",
        "assistance_strategy",
        "priority",
        "actions",
        "safety_constraints",
        "interpretation",
    ]

    # ------------------------------------------------------------------------
    # CSV helper
    # ------------------------------------------------------------------------

    def _join(items):

        if isinstance(items, list):
            return " | ".join(
                str(item)
                for item in items
            )

        return str(items)

    # ------------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------------

    print("Saving CSV...")

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for record in records:

            writer.writerow(
                {
                    "patient_id": record["patient_id"],
                    "window_start": record["window_start"],
                    "window_end": record["window_end"],
                    "year": record["year"],
                    "context_status": record["context_status"],
                    "current_state": record["current_state"],
                    "overall_score": record["overall_score"],
                    "transition_type": record["transition_type"],
                    "transition_direction": record[
                        "transition_direction"
                    ],
                    "assistance_strategy": record[
                        "assistance_strategy"
                    ],
                    "priority": record["priority"],
                    "actions": _join(
                        [
                            action["action"]
                            for action in record["actions"]
                        ]
                    ),
                    "safety_constraints": _join(
                        record["safety_constraints"]
                    ),
                    "interpretation": record[
                        "interpretation"
                    ],
                }
            )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    action_counts = Counter(
        action["action"]
        for record in records
        for action in record["actions"]
    )

    strategy_counts = Counter(
        record["assistance_strategy"]
        for record in records
    )

    priority_counts = Counter(
        record["priority"]
        for record in records
    )

    context_counts = Counter(
        record["context_status"]
        for record in records
    )

    patients = len(
        set(
            record["patient_id"]
            for record in records
        )
    )

    # ------------------------------------------------------------------------
    # Output summary
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ASSISTANCE PLANNER BUILDER - SUMMARY")
    print("=" * 70)

    print(
        f"Patients processed:        {patients}"
    )

    print(
        f"Plan records:              {len(records)}"
    )

    print("\nOutputs:")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV:  {OUTPUT_CSV}")

    print("\nAction distribution:")

    for action in sorted(action_counts):
        print(
            f"  {action:<38} "
            f"{action_counts[action]}"
        )

    print("\nAssistance strategy distribution:")

    for strategy in sorted(strategy_counts):
        print(
            f"  {strategy:<35} "
            f"{strategy_counts[strategy]}"
        )

    print("\nPriority distribution:")

    for priority in (
        "LOW",
        "MODERATE",
        "HIGH",
    ):
        print(
            f"  {priority:<12} "
            f"{priority_counts.get(priority, 0)}"
        )

    print("\nContext status distribution:")

    for status in (
        "ACTIVE",
        "INITIAL",
        "DATA_GAP",
    ):
        print(
            f"  {status:<12} "
            f"{context_counts.get(status, 0)}"
        )

    # ------------------------------------------------------------------------
    # Safety summary
    # ------------------------------------------------------------------------

    print("\nImportant:")
    print(
        "  The assistance planner produces system-behavior actions"
    )
    print(
        "  from documented care activity and temporal changes."
    )
    print(
        "  It does not diagnose disease."
    )
    print(
        "  It does not predict medical risk."
    )
    print(
        "  NO_DATA is not interpreted as STABLE."
    )

    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    build()