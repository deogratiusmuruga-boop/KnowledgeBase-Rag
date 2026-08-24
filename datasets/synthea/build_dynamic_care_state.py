"""
build_dynamic_care_state.py

ElderDocAI
Dynamic Care State Builder

Input:
    elderdocai/processed/clinical_features.json

Outputs:
    elderdocai/processed/dynamic_care_states.json
    elderdocai/processed/dynamic_care_states.csv

Purpose:
    Transform clinical features into an interpretable,
    rule-based Dynamic Care State.

Important:
    - This is NOT a diagnostic model.
    - This does NOT predict medical risk.
    - Scores represent documented care activity and complexity.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "clinical_features.json"
)

JSON_OUTPUT = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "dynamic_care_states.json"
)

CSV_OUTPUT = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "dynamic_care_states.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

LOW_THRESHOLD = 0.34
HIGH_THRESHOLD = 0.67


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value: float) -> float:
    """Keep a value between 0 and 1."""

    return max(0.0, min(1.0, value))


def safe_float(value: Any) -> float:
    """Safely convert a value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_level(score: float) -> str:
    """Convert a 0-1 score into LOW/MODERATE/HIGH."""

    if score >= HIGH_THRESHOLD:
        return "HIGH"

    if score >= LOW_THRESHOLD:
        return "MODERATE"

    return "LOW"


def percentile_rank(
    value: float,
    population_values: List[float],
) -> float:
    """
    Calculate a simple population-relative percentile rank.

    A patient with a value higher than most patients receives
    a higher normalized score.

    This avoids arbitrary assumptions such as:
        5 medications = HIGH
    """

    if not population_values:
        return 0.0

    values = sorted(population_values)

    if len(values) == 1:
        return 0.0

    below_or_equal = sum(
        1 for x in values if x <= value
    )

    rank = (below_or_equal - 1) / (len(values) - 1)

    return round(clamp(rank), 4)


def make_dimension(
    name: str,
    score: float,
    raw_features: Dict[str, Any],
    explanation: str,
) -> Dict[str, Any]:
    """Create standardized care-state dimension."""

    score = round(clamp(score), 4)

    return {
        "name": name,
        "raw_features": raw_features,
        "score": score,
        "level": get_level(score),
        "explanation": explanation,
    }


# ============================================================
# POPULATION FEATURE EXTRACTION
# ============================================================

def get_population_values(
    records: List[Dict[str, Any]],
    feature_name: str,
) -> List[float]:
    """Extract one numerical feature across the population."""

    values = []

    for record in records:
        values.append(
            safe_float(record.get(feature_name, 0))
        )

    return values


# ============================================================
# DIMENSION BUILDERS
# ============================================================

def medication_burden(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    unique_medications = safe_int(
        record.get("unique_medication_count", 0)
    )

    active_medications = safe_int(
        record.get("active_medication_status_count", 0)
    )

    medication_events = safe_int(
        record.get("medication_event_count", 0)
    )

    unique_score = percentile_rank(
        unique_medications,
        population["unique_medication_count"],
    )

    active_score = percentile_rank(
        active_medications,
        population["active_medication_status_count"],
    )

    event_score = percentile_rank(
        medication_events,
        population["medication_event_count"],
    )

    score = (
        unique_score * 0.45
        + active_score * 0.40
        + event_score * 0.15
    )

    return make_dimension(
        "Medication Burden",
        score,
        {
            "unique_medication_count": unique_medications,
            "active_medication_status_count": active_medications,
            "medication_event_count": medication_events,
        },
        (
            "Medication burden is based on the number of unique medications, "
            "currently active medication statuses, and medication events "
            "recorded in the clinical history."
        ),
    )


def clinical_activity(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    dated_events = safe_int(
        record.get("dated_event_count", 0)
    )

    events_per_year = safe_float(
        record.get("events_per_year", 0)
    )

    encounters = safe_int(
        record.get("encounter_event_count", 0)
    )

    procedures = safe_int(
        record.get("procedure_event_count", 0)
    )

    event_score = percentile_rank(
        dated_events,
        population["dated_event_count"],
    )

    yearly_score = percentile_rank(
        events_per_year,
        population["events_per_year"],
    )

    encounter_score = percentile_rank(
        encounters,
        population["encounter_event_count"],
    )

    procedure_score = percentile_rank(
        procedures,
        population["procedure_event_count"],
    )

    score = (
        event_score * 0.25
        + yearly_score * 0.30
        + encounter_score * 0.30
        + procedure_score * 0.15
    )

    return make_dimension(
        "Clinical Activity",
        score,
        {
            "dated_event_count": dated_events,
            "events_per_year": events_per_year,
            "encounter_event_count": encounters,
            "procedure_event_count": procedures,
        },
        (
            "Clinical activity reflects the volume and frequency of "
            "documented clinical events, encounters, and procedures."
        ),
    )


def observation_intensity(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    observation_events = safe_int(
        record.get("observation_event_count", 0)
    )

    unique_observations = safe_int(
        record.get("unique_observation_count", 0)
    )

    recent_observations = safe_int(
        record.get("recent_observation_event_count_365d", 0)
    )

    event_score = percentile_rank(
        observation_events,
        population["observation_event_count"],
    )

    unique_score = percentile_rank(
        unique_observations,
        population["unique_observation_count"],
    )

    recent_score = percentile_rank(
        recent_observations,
        population["recent_observation_event_count_365d"],
    )

    score = (
        event_score * 0.35
        + unique_score * 0.25
        + recent_score * 0.40
    )

    return make_dimension(
        "Observation Intensity",
        score,
        {
            "observation_event_count": observation_events,
            "unique_observation_count": unique_observations,
            "recent_observation_event_count_365d": recent_observations,
        },
        (
            "Observation intensity reflects the volume, diversity, "
            "and recent frequency of recorded clinical observations."
        ),
    )


def encounter_intensity(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    encounters = safe_int(
        record.get("encounter_event_count", 0)
    )

    recent_encounters = safe_int(
        record.get("recent_encounter_count_365d", 0)
    )

    encounter_score = percentile_rank(
        encounters,
        population["encounter_event_count"],
    )

    recent_score = percentile_rank(
        recent_encounters,
        population["recent_encounter_count_365d"],
    )

    score = (
        encounter_score * 0.45
        + recent_score * 0.55
    )

    return make_dimension(
        "Encounter Intensity",
        score,
        {
            "encounter_event_count": encounters,
            "recent_encounter_count_365d": recent_encounters,
        },
        (
            "Encounter intensity combines the overall number of "
            "healthcare encounters with encounters occurring during "
            "the most recent 365-day period."
        ),
    )


def condition_burden(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    unique_conditions = safe_int(
        record.get("unique_condition_count", 0)
    )

    condition_events = safe_int(
        record.get("condition_event_count", 0)
    )

    profile_conditions = safe_int(
        record.get("condition_count_profile", 0)
    )

    recent_conditions = safe_int(
        record.get("recent_condition_count_365d", 0)
    )

    unique_score = percentile_rank(
        unique_conditions,
        population["unique_condition_count"],
    )

    event_score = percentile_rank(
        condition_events,
        population["condition_event_count"],
    )

    profile_score = percentile_rank(
        profile_conditions,
        population["condition_count_profile"],
    )

    recent_score = percentile_rank(
        recent_conditions,
        population["recent_condition_count_365d"],
    )

    score = (
        unique_score * 0.35
        + event_score * 0.25
        + profile_score * 0.25
        + recent_score * 0.15
    )

    return make_dimension(
        "Condition Burden",
        score,
        {
            "unique_condition_count": unique_conditions,
            "condition_event_count": condition_events,
            "condition_count_profile": profile_conditions,
            "recent_condition_count_365d": recent_conditions,
        },
        (
            "Condition burden reflects the number and recurrence of "
            "documented conditions. It represents documented care "
            "complexity and does not indicate disease severity."
        ),
    )


def recent_clinical_activity(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    recent_events = safe_int(
        record.get("recent_event_count_365d", 0)
    )

    recent_medications = safe_int(
        record.get("recent_medication_event_count_365d", 0)
    )

    recent_observations = safe_int(
        record.get("recent_observation_event_count_365d", 0)
    )

    recent_conditions = safe_int(
        record.get("recent_condition_count_365d", 0)
    )

    recent_encounters = safe_int(
        record.get("recent_encounter_count_365d", 0)
    )

    event_score = percentile_rank(
        recent_events,
        population["recent_event_count_365d"],
    )

    medication_score = percentile_rank(
        recent_medications,
        population["recent_medication_event_count_365d"],
    )

    observation_score = percentile_rank(
        recent_observations,
        population["recent_observation_event_count_365d"],
    )

    condition_score = percentile_rank(
        recent_conditions,
        population["recent_condition_count_365d"],
    )

    encounter_score = percentile_rank(
        recent_encounters,
        population["recent_encounter_count_365d"],
    )

    score = (
        event_score * 0.25
        + medication_score * 0.15
        + observation_score * 0.25
        + condition_score * 0.15
        + encounter_score * 0.20
    )

    return make_dimension(
        "Recent Clinical Activity",
        score,
        {
            "recent_event_count_365d": recent_events,
            "recent_medication_event_count_365d": recent_medications,
            "recent_observation_event_count_365d": recent_observations,
            "recent_condition_count_365d": recent_conditions,
            "recent_encounter_count_365d": recent_encounters,
        },
        (
            "Recent clinical activity emphasizes events occurring "
            "during the most recent 365-day period, providing the "
            "main temporal component of the dynamic care state."
        ),
    )


def care_complexity(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    conditions = percentile_rank(
        safe_int(record.get("unique_condition_count", 0)),
        population["unique_condition_count"],
    )

    medications = percentile_rank(
        safe_int(record.get("unique_medication_count", 0)),
        population["unique_medication_count"],
    )

    encounters = percentile_rank(
        safe_int(record.get("encounter_event_count", 0)),
        population["encounter_event_count"],
    )

    observations = percentile_rank(
        safe_int(record.get("unique_observation_count", 0)),
        population["unique_observation_count"],
    )

    procedures = percentile_rank(
        safe_int(record.get("procedure_event_count", 0)),
        population["procedure_event_count"],
    )

    diagnostic_reports = percentile_rank(
        safe_int(record.get("diagnostic_report_event_count", 0)),
        population["diagnostic_report_event_count"],
    )

    score = (
        conditions * 0.25
        + medications * 0.20
        + encounters * 0.20
        + observations * 0.15
        + procedures * 0.10
        + diagnostic_reports * 0.10
    )

    return make_dimension(
        "Care Complexity",
        score,
        {
            "unique_condition_count": safe_int(
                record.get("unique_condition_count", 0)
            ),
            "unique_medication_count": safe_int(
                record.get("unique_medication_count", 0)
            ),
            "encounter_event_count": safe_int(
                record.get("encounter_event_count", 0)
            ),
            "unique_observation_count": safe_int(
                record.get("unique_observation_count", 0)
            ),
            "procedure_event_count": safe_int(
                record.get("procedure_event_count", 0)
            ),
            "diagnostic_report_event_count": safe_int(
                record.get("diagnostic_report_event_count", 0)
            ),
        },
        (
            "Care complexity combines multiple dimensions of documented "
            "care activity, including conditions, medications, encounters, "
            "observations, procedures, and diagnostic reports."
        ),
    )


def preventive_care_activity(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    immunizations = safe_int(
        record.get("immunization_event_count", 0)
    )

    care_plans = safe_int(
        record.get("care_plan_event_count", 0)
    )

    immunization_score = percentile_rank(
        immunizations,
        population["immunization_event_count"],
    )

    care_plan_score = percentile_rank(
        care_plans,
        population["care_plan_event_count"],
    )

    score = (
        immunization_score * 0.60
        + care_plan_score * 0.40
    )

    return make_dimension(
        "Preventive Care Activity",
        score,
        {
            "immunization_event_count": immunizations,
            "care_plan_event_count": care_plans,
        },
        (
            "Preventive care activity reflects documented immunization "
            "and care-plan activity. It describes recorded activity "
            "rather than adherence or clinical effectiveness."
        ),
    )


# ============================================================
# OVERALL CARE STATE
# ============================================================

def calculate_overall_score(
    dimensions: Dict[str, Dict[str, Any]]
) -> float:

    weights = {
        "Medication Burden": 0.10,
        "Clinical Activity": 0.15,
        "Observation Intensity": 0.10,
        "Encounter Intensity": 0.15,
        "Condition Burden": 0.15,
        "Recent Clinical Activity": 0.20,
        "Care Complexity": 0.10,
        "Preventive Care Activity": 0.05,
    }

    score = sum(
        dimensions[name]["score"] * weight
        for name, weight in weights.items()
    )

    return round(clamp(score), 4)


def determine_care_state(
    dimensions: Dict[str, Dict[str, Any]]
) -> str:
    """
    Determine the overall care state.

    Recent activity receives the greatest influence because this
    layer is intended to represent a dynamic care state.
    """

    overall_score = calculate_overall_score(dimensions)

    recent_score = dimensions[
        "Recent Clinical Activity"
    ]["score"]

    complexity_score = dimensions[
        "Care Complexity"
    ]["score"]

    # High current activity
    if (
        recent_score >= HIGH_THRESHOLD
        and overall_score >= HIGH_THRESHOLD
    ):
        return "HIGH_ACTIVITY"

    # Moderate current activity
    if (
        recent_score >= LOW_THRESHOLD
        or overall_score >= LOW_THRESHOLD
    ):
        if (
            recent_score >= HIGH_THRESHOLD
            or overall_score >= HIGH_THRESHOLD
        ):
            return "HIGH_ACTIVITY"

        return "MODERATE_ACTIVITY"

    # Low historical complexity but little recent activity
    if complexity_score < LOW_THRESHOLD:
        return "STABLE"

    return "LOW_ACTIVITY"


# ============================================================
# BUILD ONE CARE STATE
# ============================================================

def build_dynamic_care_state(
    record: Dict[str, Any],
    population: Dict[str, List[float]],
) -> Dict[str, Any]:

    dimensions = {
        "Medication Burden": medication_burden(
            record,
            population,
        ),

        "Clinical Activity": clinical_activity(
            record,
            population,
        ),

        "Observation Intensity": observation_intensity(
            record,
            population,
        ),

        "Encounter Intensity": encounter_intensity(
            record,
            population,
        ),

        "Condition Burden": condition_burden(
            record,
            population,
        ),

        "Recent Clinical Activity": recent_clinical_activity(
            record,
            population,
        ),

        "Care Complexity": care_complexity(
            record,
            population,
        ),

        "Preventive Care Activity": preventive_care_activity(
            record,
            population,
        ),
    }

    overall_score = calculate_overall_score(
        dimensions
    )

    care_state = determine_care_state(
        dimensions
    )

    return {
        "patient_id": record.get("patient_id"),
        "care_state": care_state,
        "overall_score": overall_score,
        "dimensions": dimensions,
        "temporal_context": {
            "first_event_date": record.get(
                "first_event_date"
            ),
            "last_event_date": record.get(
                "last_event_date"
            ),
            "timeline_days": record.get(
                "timeline_days"
            ),
            "recent_window_days": 365,
        },
        "interpretation": (
            "Rule-based representation of documented clinical "
            "activity and care-management complexity. This output "
            "is not a diagnosis and does not predict medical risk."
        ),
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_records() -> List[Dict[str, Any]]:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "clinical_features.json must contain a list of records."
        )

    return data


# ============================================================
# BUILD POPULATION DISTRIBUTIONS
# ============================================================

def build_population(
    records: List[Dict[str, Any]]
) -> Dict[str, List[float]]:

    feature_names = [
        "unique_medication_count",
        "active_medication_status_count",
        "medication_event_count",

        "dated_event_count",
        "events_per_year",
        "encounter_event_count",
        "procedure_event_count",

        "observation_event_count",
        "unique_observation_count",
        "recent_observation_event_count_365d",

        "recent_encounter_count_365d",

        "unique_condition_count",
        "condition_event_count",
        "condition_count_profile",
        "recent_condition_count_365d",

        "recent_event_count_365d",
        "recent_medication_event_count_365d",

        "diagnostic_report_event_count",
        "immunization_event_count",
        "care_plan_event_count",
    ]

    population = {}

    for feature_name in feature_names:
        population[feature_name] = get_population_values(
            records,
            feature_name,
        )

    return population


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    records: List[Dict[str, Any]]
) -> None:

    JSON_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    records: List[Dict[str, Any]]
) -> None:

    rows = []

    for record in records:

        row = {
            "patient_id": record["patient_id"],
            "care_state": record["care_state"],
            "overall_score": record["overall_score"],
        }

        for name, dimension in record["dimensions"].items():

            prefix = (
                name
                .lower()
                .replace(" ", "_")
            )

            row[f"{prefix}_score"] = dimension["score"]
            row[f"{prefix}_level"] = dimension["level"]

        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    CSV_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    records: List[Dict[str, Any]]
) -> None:

    print("\n" + "=" * 70)
    print("ELDERDOCAI - DYNAMIC CARE STATE BUILDER")
    print("=" * 70)

    print(f"Input records: {len(records)}")

    state_counts: Dict[str, int] = {}

    scores = []

    for record in records:

        state = record["care_state"]

        state_counts[state] = (
            state_counts.get(state, 0) + 1
        )

        scores.append(
            record["overall_score"]
        )

    print("\nCare-state distribution:")

    for state in [
        "STABLE",
        "LOW_ACTIVITY",
        "MODERATE_ACTIVITY",
        "HIGH_ACTIVITY",
    ]:

        print(
            f"  {state:<20} "
            f"{state_counts.get(state, 0)}"
        )

    if scores:
        print(
            "\nOverall score statistics:"
        )

        print(
            f"  Minimum: {min(scores):.4f}"
        )

        print(
            f"  Maximum: {max(scores):.4f}"
        )

        print(
            f"  Mean:    {mean(scores):.4f}"
        )

    print("\nOutputs:")

    print(
        f"  JSON: {JSON_OUTPUT}"
    )

    print(
        f"  CSV:  {CSV_OUTPUT}"
    )

    print("\nDimensions:")

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

    for index, dimension in enumerate(
        dimensions,
        start=1,
    ):
        print(
            f"  {index}. {dimension}"
        )

    print("\nImportant:")

    print(
        "  This is a rule-based care-state representation."
    )

    print(
        "  It does not diagnose disease."
    )

    print(
        "  It does not predict medical risk."
    )

    print(
        "  Scores are normalized relative to this dataset."
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\nLoading clinical features...")

    records = load_records()

    print(
        f"Loaded {len(records)} clinical feature records."
    )

    print(
        "\nBuilding population-relative feature distributions..."
    )

    population = build_population(records)

    print(
        "Population distributions ready."
    )

    print(
        "\nBuilding dynamic care states..."
    )

    dynamic_states = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        state = build_dynamic_care_state(
            record,
            population,
        )

        dynamic_states.append(state)

        if index <= 5:

            print(
                f"  Record {index}: "
                f"{state['care_state']} "
                f"(score={state['overall_score']:.4f})"
            )

    print(
        "\nSaving JSON..."
    )

    save_json(dynamic_states)

    print(
        "Saving CSV..."
    )

    save_csv(dynamic_states)

    print_summary(
        dynamic_states
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()