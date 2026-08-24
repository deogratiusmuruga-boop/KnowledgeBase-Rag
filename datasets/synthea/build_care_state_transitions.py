"""
ElderDocAI - Care-State Transition Builder

Reads:
    elderdocai/processed/care_state_timeline.json

Produces:
    elderdocai/processed/care_state_transitions.json
    elderdocai/processed/care_state_transitions.csv

Purpose:
    Identify interpretable transitions between consecutive observed
    care-state windows.

Important:
    - This is rule-based.
    - It does not diagnose disease.
    - It does not predict medical risk.
    - It describes documented changes in care activity only.
"""

import csv
import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "care_state_timeline.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "care_state_transitions.json"
)

OUTPUT_CSV = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "care_state_transitions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SCORE_CHANGE_THRESHOLD = 0.10

STATE_ORDER = {
    "STABLE": 0,
    "LOW_ACTIVITY": 1,
    "MODERATE_ACTIVITY": 2,
    "HIGH_ACTIVITY": 3,
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_dimension_score(record, dimension):
    """Return a dimension score from a timeline record."""
    dimensions = record.get("dimensions", {})
    data = dimensions.get(dimension, {})

    return safe_float(data.get("score", 0.0))


def classify_direction(previous_score, current_score):
    """
    Classify numerical score movement.

    Threshold:
        <= -0.10 -> decreasing
        >= +0.10 -> increasing
        otherwise -> no meaningful change
    """

    delta = current_score - previous_score

    if delta >= SCORE_CHANGE_THRESHOLD:
        return "INCREASING_ACTIVITY"

    if delta <= -SCORE_CHANGE_THRESHOLD:
        return "DECREASING_ACTIVITY"

    return "NO_CHANGE"


def classify_transition(previous_state, current_state, previous_score, current_score):
    """
    Determine the overall transition type.
    """

    if previous_state is None:
        return "INITIAL_STATE"

    if previous_state == "NO_DATA" or current_state == "NO_DATA":
        return "GAP"

    if previous_state == current_state:
        direction = classify_direction(
            previous_score,
            current_score
        )

        return direction

    previous_rank = STATE_ORDER.get(previous_state)
    current_rank = STATE_ORDER.get(current_state)

    if previous_rank is not None and current_rank is not None:

        if current_rank > previous_rank:
            return "STATE_ESCALATION"

        if current_rank < previous_rank:
            return "STATE_DEESCALATION"

    return "STATE_CHANGE"


def identify_changed_dimensions(previous_record, current_record):
    """
    Compare dimension scores between two care-state records.

    Returns dimensions that changed meaningfully.
    """

    dimensions = [
        "Medication Burden",
        "Clinical Activity",
        "Observation Intensity",
        "Encounter Intensity",
        "Condition Burden",
        "Recent Clinical Activity",
        "Care Complexity",
        "Preventive Care Activity",
    ]

    changes = []

    for dimension in dimensions:

        previous_score = get_dimension_score(
            previous_record,
            dimension
        )

        current_score = get_dimension_score(
            current_record,
            dimension
        )

        delta = current_score - previous_score

        if abs(delta) >= SCORE_CHANGE_THRESHOLD:

            if delta > 0:
                direction = "INCREASED"
            else:
                direction = "DECREASED"

            changes.append(
                {
                    "dimension": dimension,
                    "previous_score": round(previous_score, 4),
                    "current_score": round(current_score, 4),
                    "delta": round(delta, 4),
                    "direction": direction,
                }
            )

    return changes


def build_supporting_evidence(changed_dimensions):
    """
    Generate a concise human-readable explanation.
    """

    if not changed_dimensions:
        return "No major care-state dimension changed."

    increased = [
        x["dimension"]
        for x in changed_dimensions
        if x["direction"] == "INCREASED"
    ]

    decreased = [
        x["dimension"]
        for x in changed_dimensions
        if x["direction"] == "DECREASED"
    ]

    parts = []

    if increased:
        parts.append(
            "Increased: " + ", ".join(increased)
        )

    if decreased:
        parts.append(
            "Decreased: " + ", ".join(decreased)
        )

    return "; ".join(parts)


def calculate_transition_magnitude(delta):
    """
    Convert absolute score change into an interpretable magnitude.
    """

    magnitude = abs(delta)

    if magnitude < 0.10:
        return "MINOR"

    if magnitude < 0.25:
        return "MODERATE"

    return "LARGE"


# ============================================================
# MAIN TRANSITION BUILDER
# ============================================================

def build_transitions(timeline_records):

    grouped = {}

    for record in timeline_records:

        patient_id = record.get("patient_id")

        if not patient_id:
            continue

        grouped.setdefault(patient_id, []).append(record)

    transitions = []

    for patient_id, records in grouped.items():

        records.sort(
            key=lambda x: x.get("window_start", "")
        )

        previous = None

        for current in records:

            current_state = current.get(
                "care_state",
                "NO_DATA"
            )

            current_score = safe_float(
                current.get("overall_score", 0.0)
            )

            # ------------------------------------------------
            # INITIAL STATE
            # ------------------------------------------------

            if previous is None:

                transition_type = "INITIAL_STATE"

                transition = {
                    "patient_id": patient_id,
                    "window_start": current.get("window_start"),
                    "window_end": current.get("window_end"),
                    "previous_state": None,
                    "current_state": current_state,
                    "previous_score": None,
                    "current_score": round(current_score, 4),
                    "score_delta": None,
                    "transition_type": transition_type,
                    "transition_magnitude": None,
                    "transition_direction": "INITIAL",
                    "changed_dimensions": [],
                    "supporting_evidence": "Initial observed care-state window.",
                }

                transitions.append(transition)

                previous = current

                continue

            # ------------------------------------------------
            # PREVIOUS VALUES
            # ------------------------------------------------

            previous_state = previous.get(
                "care_state",
                "NO_DATA"
            )

            previous_score = safe_float(
                previous.get("overall_score", 0.0)
            )

            delta = current_score - previous_score

            # ------------------------------------------------
            # TRANSITION TYPE
            # ------------------------------------------------

            transition_type = classify_transition(
                previous_state,
                current_state,
                previous_score,
                current_score,
            )

            # ------------------------------------------------
            # GAP HANDLING
            # ------------------------------------------------

            if (
                previous_state == "NO_DATA"
                or current_state == "NO_DATA"
            ):

                transition = {
                    "patient_id": patient_id,
                    "window_start": current.get("window_start"),
                    "window_end": current.get("window_end"),
                    "previous_state": previous_state,
                    "current_state": current_state,
                    "previous_score": round(previous_score, 4),
                    "current_score": round(current_score, 4),
                    "score_delta": round(delta, 4),
                    "transition_type": "GAP",
                    "transition_magnitude": None,
                    "transition_direction": "UNKNOWN",
                    "changed_dimensions": [],
                    "supporting_evidence": (
                        "Transition blocked because at least one "
                        "adjacent window contains NO_DATA."
                    ),
                }

                transitions.append(transition)

                previous = current

                continue

            # ------------------------------------------------
            # DIMENSION CHANGES
            # ------------------------------------------------

            changed_dimensions = identify_changed_dimensions(
                previous,
                current
            )

            supporting_evidence = build_supporting_evidence(
                changed_dimensions
            )

            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            if transition_type == "STATE_ESCALATION":
                direction = "INCREASING"

            elif transition_type == "STATE_DEESCALATION":
                direction = "DECREASING"

            elif delta > SCORE_CHANGE_THRESHOLD:
                direction = "INCREASING"

            elif delta < -SCORE_CHANGE_THRESHOLD:
                direction = "DECREASING"

            else:
                direction = "UNCHANGED"

            # ------------------------------------------------
            # MAGNITUDE
            # ------------------------------------------------

            magnitude = calculate_transition_magnitude(
                delta
            )

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            transition = {
                "patient_id": patient_id,
                "window_start": current.get("window_start"),
                "window_end": current.get("window_end"),
                "previous_state": previous_state,
                "current_state": current_state,
                "previous_score": round(previous_score, 4),
                "current_score": round(current_score, 4),
                "score_delta": round(delta, 4),
                "transition_type": transition_type,
                "transition_magnitude": magnitude,
                "transition_direction": direction,
                "changed_dimensions": changed_dimensions,
                "supporting_evidence": supporting_evidence,
            }

            transitions.append(transition)

            previous = current

    return transitions


# ============================================================
# CSV EXPORT
# ============================================================

def save_csv(transitions):

    rows = []

    for record in transitions:

        changed_dimensions = record.get(
            "changed_dimensions",
            []
        )

        dimension_text = " | ".join(
            (
                f"{x['dimension']}: "
                f"{x['direction']} "
                f"(Δ{x['delta']:+.4f})"
            )
            for x in changed_dimensions
        )

        rows.append(
            {
                "patient_id": record.get("patient_id"),
                "window_start": record.get("window_start"),
                "window_end": record.get("window_end"),
                "previous_state": record.get("previous_state"),
                "current_state": record.get("current_state"),
                "previous_score": record.get("previous_score"),
                "current_score": record.get("current_score"),
                "score_delta": record.get("score_delta"),
                "transition_type": record.get("transition_type"),
                "transition_magnitude": record.get(
                    "transition_magnitude"
                ),
                "transition_direction": record.get(
                    "transition_direction"
                ),
                "changed_dimensions": dimension_text,
                "supporting_evidence": record.get(
                    "supporting_evidence"
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
        "changed_dimensions",
        "supporting_evidence",
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
    print("ELDERDOCAI - CARE STATE TRANSITION BUILDER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("Loading care-state timeline...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        timeline_records = json.load(f)

    print(
        f"Loaded {len(timeline_records)} timeline record(s)."
    )
    print()

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    print("Building care-state transitions...")

    transitions = build_transitions(
        timeline_records
    )

    print(
        f"Generated {len(transitions)} transition record(s)."
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
            transitions,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    print("Saving CSV...")

    save_csv(transitions)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    from collections import Counter

    transition_counts = Counter(
        x.get("transition_type")
        for x in transitions
    )

    direction_counts = Counter(
        x.get("transition_direction")
        for x in transitions
    )

    print()
    print("=" * 70)
    print("ELDERDOCAI - CARE STATE TRANSITION BUILDER")
    print("=" * 70)
    print()

    print(
        f"Input timeline records:  {len(timeline_records)}"
    )

    print(
        f"Transition records:      {len(transitions)}"
    )

    print()
    print("Transition distribution:")

    for transition_type, count in sorted(
        transition_counts.items()
    ):

        print(
            f"  {transition_type:<35} {count}"
        )

    print()
    print("Direction distribution:")

    for direction, count in sorted(
        direction_counts.items()
    ):

        print(
            f"  {direction:<35} {count}"
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
    print("Transition design:")
    print("  Score threshold: 0.10")
    print("  Magnitude: MINOR / MODERATE / LARGE")
    print("  Dimension-level changes: enabled")
    print("  NO_DATA transitions: blocked")
    print()

    print("Important:")
    print("  This system does not diagnose disease.")
    print("  This system does not predict medical risk.")
    print("  Transitions represent documented changes in care activity only.")

    print("=" * 70)


if __name__ == "__main__":
    main()