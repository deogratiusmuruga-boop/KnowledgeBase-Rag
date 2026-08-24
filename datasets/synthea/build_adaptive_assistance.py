"""
ElderDocAI - Adaptive Assistance Builder

Reads:
    elderdocai/processed/care_state_transitions.json

Produces:
    elderdocai/processed/adaptive_assistance.json
    elderdocai/processed/adaptive_assistance.csv

Purpose:
    Convert care-state and transition information into an
    interpretable adaptive-assistance signal.

Important:
    - Rule-based only.
    - Does not diagnose disease.
    - Does not predict medical risk.
    - Does not prescribe treatment.
    - Represents documented care activity only.
"""

import csv
import json
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "care_state_transitions.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "adaptive_assistance.json"
)

OUTPUT_CSV = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "adaptive_assistance.csv"
)


# ============================================================
# RULE CONFIGURATION
# ============================================================

STATE_ASSISTANCE = {
    "NO_DATA": {
        "mode": "WAIT_FOR_DATA",
        "priority": "LOW",
        "reason": "No documented clinical activity is available for this window.",
    },
    "STABLE": {
        "mode": "MAINTENANCE_SUPPORT",
        "priority": "LOW",
        "reason": "Documented care activity is relatively stable.",
    },
    "LOW_ACTIVITY": {
        "mode": "LIGHT_SUPPORT",
        "priority": "LOW",
        "reason": "Documented care activity is relatively low.",
    },
    "MODERATE_ACTIVITY": {
        "mode": "CONTEXTUAL_SUPPORT",
        "priority": "MODERATE",
        "reason": "Documented care activity is at a moderate level.",
    },
    "HIGH_ACTIVITY": {
        "mode": "ENHANCED_SUPPORT",
        "priority": "HIGH",
        "reason": "Documented care activity is relatively high.",
    },
}


TRANSITION_ADJUSTMENTS = {
    "INITIAL_STATE": {
        "mode": "INITIAL_CONTEXT",
        "priority": "LOW",
        "reason": "This is the first observed care-state window.",
    },
    "NO_CHANGE": {
        "mode": None,
        "priority": None,
        "reason": None,
    },
    "INCREASING_ACTIVITY": {
        "mode": "MONITORING_SUPPORT",
        "priority": "MODERATE",
        "reason": "Documented care activity increased compared with the previous window.",
    },
    "DECREASING_ACTIVITY": {
        "mode": "FOLLOW_UP_SUPPORT",
        "priority": "MODERATE",
        "reason": "Documented care activity decreased compared with the previous window.",
    },
    "STATE_ESCALATION": {
        "mode": "ADAPTIVE_ESCALATION",
        "priority": "HIGH",
        "reason": "The documented care state moved to a higher activity level.",
    },
    "STATE_DEESCALATION": {
        "mode": "ADAPTIVE_DEESCALATION",
        "priority": "MODERATE",
        "reason": "The documented care state moved to a lower activity level.",
    },
    "GAP": {
        "mode": "WAIT_FOR_DATA",
        "priority": "LOW",
        "reason": "A temporal gap or NO_DATA window prevents interpretation of a continuous transition.",
    },
}


PRIORITY_ORDER = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
}


# ============================================================
# HELPERS
# ============================================================

def select_priority(*priorities):
    """
    Select the strongest priority.
    """

    valid = [
        p for p in priorities
        if p in PRIORITY_ORDER
    ]

    if not valid:
        return "LOW"

    return max(
        valid,
        key=lambda p: PRIORITY_ORDER[p]
    )


def build_dimension_reasons(changed_dimensions):
    """
    Convert changed dimensions into interpretable reasons.
    """

    reasons = []

    for item in changed_dimensions:

        dimension = item.get("dimension")
        direction = item.get("direction")
        delta = item.get("delta", 0)

        if not dimension:
            continue

        if direction == "INCREASED":
            reasons.append(
                f"{dimension} increased "
                f"(Δ{delta:+.4f})"
            )

        elif direction == "DECREASED":
            reasons.append(
                f"{dimension} decreased "
                f"(Δ{delta:+.4f})"
            )

    return reasons


def determine_assistance(record):
    """
    Determine adaptive assistance from the current state
    and the observed transition.
    """

    current_state = record.get(
        "current_state",
        "NO_DATA"
    )

    transition_type = record.get(
        "transition_type",
        "INITIAL_STATE"
    )

    transition_direction = record.get(
        "transition_direction",
        "UNKNOWN"
    )

    current_score = record.get(
        "current_score"
    )

    score_delta = record.get(
        "score_delta"
    )

    changed_dimensions = record.get(
        "changed_dimensions",
        []
    )

    # --------------------------------------------------------
    # BASE STATE RULE
    # --------------------------------------------------------

    state_rule = STATE_ASSISTANCE.get(
        current_state,
        STATE_ASSISTANCE["NO_DATA"]
    )

    base_mode = state_rule["mode"]
    base_priority = state_rule["priority"]

    reasons = [
        state_rule["reason"]
    ]

    # --------------------------------------------------------
    # TRANSITION RULE
    # --------------------------------------------------------

    transition_rule = TRANSITION_ADJUSTMENTS.get(
        transition_type,
        {}
    )

    transition_mode = transition_rule.get(
        "mode"
    )

    transition_priority = transition_rule.get(
        "priority"
    )

    transition_reason = transition_rule.get(
        "reason"
    )

    if transition_reason:
        reasons.append(transition_reason)

    # --------------------------------------------------------
    # GAP OVERRIDES EVERYTHING
    # --------------------------------------------------------

    if transition_type == "GAP":

        return {
            "assistance_mode": "WAIT_FOR_DATA",
            "priority": "LOW",
            "reason_codes": [
                "NO_DATA_OR_TEMPORAL_GAP"
            ],
            "reasons": [
                "Assistance adaptation is paused because "
                "the timeline contains a NO_DATA window."
            ],
        }

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    if transition_type == "INITIAL_STATE":

        return {
            "assistance_mode": "INITIAL_CONTEXT",
            "priority": base_priority,
            "reason_codes": [
                "INITIAL_CARE_STATE"
            ],
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # STATE ESCALATION
    # --------------------------------------------------------

    if transition_type == "STATE_ESCALATION":

        dimension_reasons = build_dimension_reasons(
            changed_dimensions
        )

        reasons.extend(
            dimension_reasons
        )

        return {
            "assistance_mode": "ADAPTIVE_ESCALATION",
            "priority": "HIGH",
            "reason_codes": [
                "STATE_ESCALATION",
                "INCREASED_CARE_ACTIVITY",
            ],
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # STATE DEESCALATION
    # --------------------------------------------------------

    if transition_type == "STATE_DEESCALATION":

        dimension_reasons = build_dimension_reasons(
            changed_dimensions
        )

        reasons.extend(
            dimension_reasons
        )

        return {
            "assistance_mode": "ADAPTIVE_DEESCALATION",
            "priority": "MODERATE",
            "reason_codes": [
                "STATE_DEESCALATION",
                "DECREASED_CARE_ACTIVITY",
            ],
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # INCREASING ACTIVITY
    # --------------------------------------------------------

    if transition_type == "INCREASING_ACTIVITY":

        dimension_reasons = build_dimension_reasons(
            changed_dimensions
        )

        reasons.extend(
            dimension_reasons
        )

        return {
            "assistance_mode": "MONITORING_SUPPORT",
            "priority": select_priority(
                base_priority,
                "MODERATE"
            ),
            "reason_codes": [
                "INCREASING_ACTIVITY"
            ],
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # DECREASING ACTIVITY
    # --------------------------------------------------------

    if transition_type == "DECREASING_ACTIVITY":

        dimension_reasons = build_dimension_reasons(
            changed_dimensions
        )

        reasons.extend(
            dimension_reasons
        )

        return {
            "assistance_mode": "FOLLOW_UP_SUPPORT",
            "priority": select_priority(
                base_priority,
                "MODERATE"
            ),
            "reason_codes": [
                "DECREASING_ACTIVITY"
            ],
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # NO CHANGE
    # --------------------------------------------------------

    if transition_type == "NO_CHANGE":

        return {
            "assistance_mode": base_mode,
            "priority": base_priority,
            "reason_codes": [
                "NO_MAJOR_STATE_CHANGE"
            ],
            "reasons": [
                state_rule["reason"],
                "No major care-state transition was detected.",
            ],
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return {
        "assistance_mode": base_mode,
        "priority": base_priority,
        "reason_codes": [
            "DEFAULT_RULE"
        ],
        "reasons": reasons,
    }


# ============================================================
# BUILD ADAPTIVE ASSISTANCE
# ============================================================

def build_adaptive_assistance(transitions):

    results = []

    for record in transitions:

        assistance = determine_assistance(
            record
        )

        result = {
            "patient_id": record.get(
                "patient_id"
            ),

            "window_start": record.get(
                "window_start"
            ),

            "window_end": record.get(
                "window_end"
            ),

            "previous_state": record.get(
                "previous_state"
            ),

            "current_state": record.get(
                "current_state"
            ),

            "previous_score": record.get(
                "previous_score"
            ),

            "current_score": record.get(
                "current_score"
            ),

            "score_delta": record.get(
                "score_delta"
            ),

            "transition_type": record.get(
                "transition_type"
            ),

            "transition_magnitude": record.get(
                "transition_magnitude"
            ),

            "transition_direction": record.get(
                "transition_direction"
            ),

            "changed_dimensions": record.get(
                "changed_dimensions",
                []
            ),

            "assistance_mode": assistance[
                "assistance_mode"
            ],

            "priority": assistance[
                "priority"
            ],

            "reason_codes": assistance[
                "reason_codes"
            ],

            "reasons": assistance[
                "reasons"
            ],

            "interpretation": (
                "Adaptive assistance is selected from "
                "documented care-state and transition "
                "information. It does not represent a "
                "diagnosis or medical risk prediction."
            ),
        }

        results.append(result)

    return results


# ============================================================
# CSV EXPORT
# ============================================================

def save_csv(records):

    rows = []

    for record in records:

        changed_dimensions = record.get(
            "changed_dimensions",
            []
        )

        reasons = record.get(
            "reasons",
            []
        )

        reason_codes = record.get(
            "reason_codes",
            []
        )

        dimension_text = " | ".join(
            (
                f"{x.get('dimension')}: "
                f"{x.get('direction')} "
                f"(Δ{x.get('delta', 0):+.4f})"
            )
            for x in changed_dimensions
        )

        rows.append(
            {
                "patient_id": record.get(
                    "patient_id"
                ),

                "window_start": record.get(
                    "window_start"
                ),

                "window_end": record.get(
                    "window_end"
                ),

                "previous_state": record.get(
                    "previous_state"
                ),

                "current_state": record.get(
                    "current_state"
                ),

                "previous_score": record.get(
                    "previous_score"
                ),

                "current_score": record.get(
                    "current_score"
                ),

                "score_delta": record.get(
                    "score_delta"
                ),

                "transition_type": record.get(
                    "transition_type"
                ),

                "transition_magnitude": record.get(
                    "transition_magnitude"
                ),

                "transition_direction": record.get(
                    "transition_direction"
                ),

                "assistance_mode": record.get(
                    "assistance_mode"
                ),

                "priority": record.get(
                    "priority"
                ),

                "reason_codes": " | ".join(
                    reason_codes
                ),

                "changed_dimensions": dimension_text,

                "reasons": " | ".join(
                    reasons
                ),
            }
        )

    fieldnames = [
        "patient_id",
        "window_start",
        "window_end",
        "previous_state",
        "current_state",
        "previous_score",
        "current_score",
        "score_delta",
        "transition_type",
        "transition_magnitude",
        "transition_direction",
        "assistance_mode",
        "priority",
        "reason_codes",
        "changed_dimensions",
        "reasons",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ELDERDOCAI - ADAPTIVE ASSISTANCE BUILDER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD TRANSITIONS
    # --------------------------------------------------------

    print("Loading care-state transitions...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        transitions = json.load(f)

    print(
        f"Loaded {len(transitions)} transition record(s)."
    )
    print()

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    print("Building adaptive assistance decisions...")

    results = build_adaptive_assistance(
        transitions
    )

    print(
        f"Generated {len(results)} assistance record(s)."
    )
    print()

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    print("Saving JSON...")

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    print("Saving CSV...")

    save_csv(results)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    mode_counts = Counter(
        x.get("assistance_mode")
        for x in results
    )

    priority_counts = Counter(
        x.get("priority")
        for x in results
    )

    print()
    print("=" * 70)
    print("ELDERDOCAI - ADAPTIVE ASSISTANCE BUILDER")
    print("=" * 70)
    print()

    print(
        f"Input transition records: {len(transitions)}"
    )

    print(
        f"Assistance records:        {len(results)}"
    )

    print()
    print("Assistance-mode distribution:")

    for mode, count in sorted(
        mode_counts.items()
    ):

        print(
            f"  {mode:<30} {count}"
        )

    print()
    print("Priority distribution:")

    for priority, count in sorted(
        priority_counts.items()
    ):

        print(
            f"  {priority:<30} {count}"
        )

    print()
    print("Outputs:")

    print(
        f"  JSON: {OUTPUT_JSON}"
    )

    print(
        f"  CSV:  {OUTPUT_CSV}"
    )

    print()
    print("Adaptive assistance design:")
    print("  State-aware: YES")
    print("  Transition-aware: YES")
    print("  Dimension-aware: YES")
    print("  Rule-based: YES")
    print("  Medical diagnosis: NO")
    print("  Medical risk prediction: NO")
    print()

    print("Important:")
    print(
        "  Assistance modes represent system adaptation "
        "to documented care activity."
    )
    print(
        "  They do not represent diagnosis or medical risk."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()