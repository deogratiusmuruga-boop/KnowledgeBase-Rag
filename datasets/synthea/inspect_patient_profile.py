"""
======================================================================
ELDERDOCAI CLINICAL FEATURE BUILDER
======================================================================

Purpose:
    Build structured clinical features from ElderDocAI patient profiles.

Input:
    elderdocai/processed/patient_profiles.json

Outputs:
    elderdocai/processed/clinical_features.csv
    elderdocai/processed/clinical_features.json

IMPORTANT:
    timeline_events is the authoritative source for event-level
    clinical activity.

The profile-level fields such as:
    - medication_event_count
    - observation_count
    - encounter_count
    - resource_counts

are used as supporting aggregate information.

This script does NOT make clinical diagnoses or risk predictions.
It only transforms existing patient data into structured features.
======================================================================
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


# =====================================================================
# PATH CONFIGURATION
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "elderdocai" / "processed"

INPUT_FILE = PROCESSED_DIR / "patient_profiles.json"

OUTPUT_CSV = PROCESSED_DIR / "clinical_features.csv"
OUTPUT_JSON = PROCESSED_DIR / "clinical_features.json"


# =====================================================================
# CONSTANTS
# =====================================================================

RECENT_DAYS = 365

RESOURCE_TYPES = {
    "Observation",
    "Procedure",
    "DiagnosticReport",
    "MedicationRequest",
    "MedicationAdministration",
    "Encounter",
    "Condition",
    "Immunization",
    "CarePlan",
    "Patient",
    "AllergyIntolerance",
}

MEDICATION_RESOURCE_TYPES = {
    "MedicationRequest",
    "MedicationAdministration",
}


# =====================================================================
# GENERAL HELPERS
# =====================================================================

def clean_string(value: Any) -> str:
    """Return a normalized string."""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    if value is None:
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """Safely convert a value to float."""

    if value is None:
        return default

    try:
        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def get_list(
    profile: dict[str, Any],
    key: str,
) -> list:
    """Safely retrieve a list."""

    value = profile.get(key, [])

    if isinstance(value, list):
        return value

    return []


def get_dict(
    profile: dict[str, Any],
    key: str,
) -> dict:
    """Safely retrieve a dictionary."""

    value = profile.get(key, {})

    if isinstance(value, dict):
        return value

    return {}


def unique_nonempty(
    values: list[Any],
) -> list[str]:
    """Return unique non-empty strings preserving order."""

    result = []
    seen = set()

    for value in values:

        text = clean_string(value)

        if not text:
            continue

        if text not in seen:

            seen.add(text)
            result.append(text)

    return result


# =====================================================================
# DATE HELPERS
# =====================================================================

def parse_date(
    value: Any,
) -> datetime | None:
    """
    Parse common FHIR/Synthea date formats.

    Returned datetime values are normalized to naive UTC.
    """

    if value is None:
        return None

    text = clean_string(value)

    if not text:
        return None

    # ---------------------------------------------------------------
    # ISO datetime
    # ---------------------------------------------------------------

    try:

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)

        if parsed.tzinfo is not None:

            parsed = (
                parsed
                .astimezone(timezone.utc)
                .replace(tzinfo=None)
            )

        return parsed

    except (ValueError, TypeError):
        pass

    # ---------------------------------------------------------------
    # Date-only
    # ---------------------------------------------------------------

    try:

        return datetime.strptime(
            text[:10],
            "%Y-%m-%d",
        )

    except (ValueError, TypeError):
        return None


# =====================================================================
# TIMELINE EVENT HELPERS
# =====================================================================

def get_timeline_events(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Retrieve the authoritative timeline event list.

    ElderDocAI patient profiles contain event-level clinical data
    inside the `timeline_events` field.
    """

    events = profile.get(
        "timeline_events",
        [],
    )

    if not isinstance(events, list):
        return []

    return [
        event
        for event in events
        if isinstance(event, dict)
    ]


def event_resource_type(
    event: dict[str, Any],
) -> str:
    """Return the event resource type."""

    return clean_string(
        event.get("resource_type")
        or event.get("resourceType")
    )


def event_date(
    event: dict[str, Any],
) -> datetime | None:
    """
    Extract event date.

    Actual Synthea/ElderDocAI schema uses:
        date
    """

    return parse_date(
        event.get("date")
        or event.get("event_date")
        or event.get("effectiveDateTime")
        or event.get("issued")
    )


def event_code(
    event: dict[str, Any],
) -> str:
    """Extract event clinical code/display."""

    return clean_string(
        event.get("code")
        or event.get("display")
        or event.get("name")
    )


def event_value(
    event: dict[str, Any],
) -> Any:
    """Extract observation value."""

    return event.get("value")


def event_unit(
    event: dict[str, Any],
) -> str:
    """Extract observation unit."""

    return clean_string(
        event.get("unit")
        or event.get("value_unit")
    )


def event_status(
    event: dict[str, Any],
) -> str:
    """Extract event status."""

    return clean_string(
        event.get("status")
    )


# =====================================================================
# EVENT COUNTING
# =====================================================================

def count_resource_events(
    events: list[dict[str, Any]],
    resource_type: str,
) -> int:
    """Count events by FHIR resource type."""

    return sum(
        1
        for event in events
        if event_resource_type(event) == resource_type
    )


def resource_counter(
    events: list[dict[str, Any]],
) -> Counter:
    """Return resource type distribution."""

    return Counter(
        event_resource_type(event)
        for event in events
        if event_resource_type(event)
    )


# =====================================================================
# CONDITION FEATURES
# =====================================================================

def extract_condition_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:

    profile_conditions = get_list(
        profile,
        "conditions",
    )

    condition_names = []

    for item in profile_conditions:

        if isinstance(item, str):

            condition_names.append(item)

        elif isinstance(item, dict):

            condition_names.append(
                item.get("code")
                or item.get("name")
                or item.get("display")
                or item.get("description")
                or ""
            )

    condition_events = [
        event
        for event in events
        if event_resource_type(event) == "Condition"
    ]

    event_condition_names = [
        event_code(event)
        for event in condition_events
    ]

    all_conditions = unique_nonempty(
        condition_names
        + event_condition_names
    )

    return {

        "condition_count_profile":
            len(profile_conditions),

        "unique_condition_count":
            len(all_conditions),

        "condition_event_count":
            len(condition_events),

        "conditions":
            all_conditions,

        "recent_condition_count":
            0,
    }


# =====================================================================
# ALLERGY FEATURES
# =====================================================================

def extract_allergy_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:

    profile_allergies = get_list(
        profile,
        "allergies",
    )

    allergy_names = []

    for item in profile_allergies:

        if isinstance(item, str):

            allergy_names.append(item)

        elif isinstance(item, dict):

            allergy_names.append(
                item.get("code")
                or item.get("name")
                or item.get("display")
                or item.get("description")
                or ""
            )

    allergy_events = [
        event
        for event in events
        if event_resource_type(event)
        == "AllergyIntolerance"
    ]

    allergy_names.extend(
        event_code(event)
        for event in allergy_events
    )

    allergy_names = unique_nonempty(
        allergy_names
    )

    return {

        "allergy_count":
            len(allergy_names),

        "allergies":
            allergy_names,

        "allergy_event_count":
            len(allergy_events),
    }


# =====================================================================
# MEDICATION FEATURES
# =====================================================================

def extract_medication_name(
    item: Any,
) -> str:

    if isinstance(item, str):
        return clean_string(item)

    if not isinstance(item, dict):
        return ""

    return clean_string(
        item.get("medication")
        or item.get("code")
        or item.get("name")
        or item.get("display")
        or item.get("medication_name")
    )


def extract_medication_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:

    medication_entries = get_list(
        profile,
        "medications",
    )

    medication_names = unique_nonempty(
        [
            extract_medication_name(item)
            for item in medication_entries
        ]
    )

    medication_events = [
        event
        for event in events
        if event_resource_type(event)
        in MEDICATION_RESOURCE_TYPES
    ]

    medication_event_names = unique_nonempty(
        [
            event_code(event)
            for event in medication_events
        ]
    )

    all_medications = unique_nonempty(
        medication_names
        + medication_event_names
    )

    # ---------------------------------------------------------------
    # Status counts
    # ---------------------------------------------------------------

    status_counter = Counter()

    for item in medication_entries:

        if isinstance(item, dict):

            statuses = item.get(
                "statuses",
                [],
            )

            if isinstance(statuses, list):

                for status in statuses:

                    status_text = clean_string(
                        status
                    ).lower()

                    if status_text:
                        status_counter[
                            status_text
                        ] += 1

            else:

                status = clean_string(
                    item.get("status")
                ).lower()

                if status:
                    status_counter[
                        status
                    ] += 1

    for event in medication_events:

        status = event_status(
            event
        ).lower()

        if status:
            status_counter[
                status
            ] += 1

    # ---------------------------------------------------------------
    # Dates
    # ---------------------------------------------------------------

    medication_dates = [
        event_date(event)
        for event in medication_events
    ]

    medication_dates = [
        date
        for date in medication_dates
        if date is not None
    ]

    medication_first_date = ""

    medication_last_date = ""

    if medication_dates:

        medication_first_date = (
            min(medication_dates)
            .date()
            .isoformat()
        )

        medication_last_date = (
            max(medication_dates)
            .date()
            .isoformat()
        )

    return {

        "unique_medication_count":
            len(all_medications),

        "medication_profile_count":
            len(medication_entries),

        "medication_event_count":
            len(medication_events),

        "active_medication_status_count":
            status_counter.get(
                "active",
                0,
            ),

        "completed_medication_status_count":
            status_counter.get(
                "completed",
                0,
            ),

        "medication_first_date":
            medication_first_date,

        "medication_last_date":
            medication_last_date,

        "medications":
            all_medications,
    }


# =====================================================================
# OBSERVATION FEATURES
# =====================================================================

def extract_observation_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:

    observation_entries = get_list(
        profile,
        "observations",
    )

    observation_events = [
        event
        for event in events
        if event_resource_type(event)
        == "Observation"
    ]

    observation_codes = unique_nonempty(
        [
            event_code(event)
            for event in observation_events
        ]
    )

    numeric_values = []

    units = []

    observation_dates = []

    for event in observation_events:

        date = event_date(event)

        if date is not None:
            observation_dates.append(date)

        value = event_value(event)

        numeric_value = safe_float(
            value
        )

        if numeric_value is not None:

            numeric_values.append(
                numeric_value
            )

        unit = event_unit(event)

        if unit:
            units.append(unit)

    latest_observation_date = ""

    if observation_dates:

        latest_observation_date = (
            max(observation_dates)
            .date()
            .isoformat()
        )

    return {

        "observation_count_profile":
            safe_int(
                profile.get(
                    "observation_count"
                ),
                len(observation_entries),
            ),

        "observation_event_count":
            len(observation_events),

        "unique_observation_count":
            len(observation_codes),

        "observations_with_numeric_values":
            len(numeric_values),

        "observations_with_units":
            len(unique_nonempty(units)),

        "numeric_observation_min":
            min(numeric_values)
            if numeric_values
            else None,

        "numeric_observation_max":
            max(numeric_values)
            if numeric_values
            else None,

        "numeric_observation_mean":
            mean(numeric_values)
            if numeric_values
            else None,

        "numeric_observation_median":
            median(numeric_values)
            if numeric_values
            else None,

        "unique_observation_codes":
            len(observation_codes),

        "latest_observation_date":
            latest_observation_date,
    }


# =====================================================================
# TEMPORAL FEATURES
# =====================================================================

def extract_temporal_features(
    events: list[dict[str, Any]],
) -> dict[str, Any]:

    dated_events = []

    undated_events = []

    for event in events:

        if event_date(event) is None:
            undated_events.append(event)

        else:
            dated_events.append(event)

    dates = [
        event_date(event)
        for event in dated_events
    ]

    dates = [
        date
        for date in dates
        if date is not None
    ]

    # ---------------------------------------------------------------
    # No dates
    # ---------------------------------------------------------------

    if not dates:

        return {

            "first_event_date":
                "",

            "last_event_date":
                "",

            "timeline_days":
                0,

            "dated_event_count":
                0,

            "undated_event_count":
                len(events),

            "events_per_year":
                0.0,

            "recent_event_count_365d":
                0,

            "recent_medication_event_count_365d":
                0,

            "recent_observation_event_count_365d":
                0,

            "recent_condition_event_count_365d":
                0,

            "recent_encounter_count_365d":
                0,
        }

    # ---------------------------------------------------------------
    # Timeline
    # ---------------------------------------------------------------

    first_date = min(dates)

    last_date = max(dates)

    timeline_days = (
        last_date - first_date
    ).days

    events_per_year = 0.0

    if timeline_days > 0:

        events_per_year = (
            len(dated_events)
            /
            (timeline_days / 365.25)
        )

    # ---------------------------------------------------------------
    # Recent window
    # ---------------------------------------------------------------

    recent_cutoff = (
        last_date
        - timedelta(days=RECENT_DAYS)
    )

    recent_events = [
        event
        for event in dated_events
        if event_date(event) >= recent_cutoff
    ]

    recent_medications = [
        event
        for event in recent_events
        if event_resource_type(event)
        in MEDICATION_RESOURCE_TYPES
    ]

    recent_observations = [
        event
        for event in recent_events
        if event_resource_type(event)
        == "Observation"
    ]

    recent_conditions = [
        event
        for event in recent_events
        if event_resource_type(event)
        == "Condition"
    ]

    recent_encounters = [
        event
        for event in recent_events
        if event_resource_type(event)
        == "Encounter"
    ]

    return {

        "first_event_date":
            first_date.date().isoformat(),

        "last_event_date":
            last_date.date().isoformat(),

        "timeline_days":
            timeline_days,

        "dated_event_count":
            len(dated_events),

        "undated_event_count":
            len(undated_events),

        "events_per_year":
            round(
                events_per_year,
                2,
            ),

        "recent_event_count_365d":
            len(recent_events),

        "recent_medication_event_count_365d":
            len(recent_medications),

        "recent_observation_event_count_365d":
            len(recent_observations),

        "recent_condition_event_count_365d":
            len(recent_conditions),

        "recent_encounter_count_365d":
            len(recent_encounters),
    }


# =====================================================================
# PROFILE-LEVEL COUNTS
# =====================================================================

def extract_profile_counts(
    profile: dict[str, Any],
) -> dict[str, Any]:

    resource_counts = get_dict(
        profile,
        "resource_counts",
    )

    return {

        "profile_condition_count":
            safe_int(
                resource_counts.get(
                    "Condition"
                ),
                len(
                    get_list(
                        profile,
                        "conditions",
                    )
                ),
            ),

        "profile_allergy_count":
            safe_int(
                resource_counts.get(
                    "AllergyIntolerance"
                ),
                len(
                    get_list(
                        profile,
                        "allergies",
                    )
                ),
            ),

        "profile_medication_event_count":
            safe_int(
                profile.get(
                    "medication_event_count"
                ),
                0,
            ),

        "profile_observation_count":
            safe_int(
                profile.get(
                    "observation_count"
                ),
                0,
            ),

        "profile_procedure_count":
            safe_int(
                resource_counts.get(
                    "Procedure"
                ),
                len(
                    get_list(
                        profile,
                        "procedures",
                    )
                ),
            ),

        "profile_diagnostic_report_count":
            safe_int(
                resource_counts.get(
                    "DiagnosticReport"
                ),
                len(
                    get_list(
                        profile,
                        "diagnostic_reports",
                    )
                ),
            ),

        "profile_immunization_count":
            safe_int(
                resource_counts.get(
                    "Immunization"
                ),
                len(
                    get_list(
                        profile,
                        "immunizations",
                    )
                ),
            ),

        "profile_encounter_count":
            safe_int(
                profile.get(
                    "encounter_count"
                ),
                0,
            ),
    }


# =====================================================================
# PATIENT FEATURE BUILDER
# =====================================================================

def build_patient_features(
    profile: dict[str, Any],
) -> dict[str, Any]:

    patient_id = clean_string(
        profile.get(
            "patient_id"
        )
    )

    # ---------------------------------------------------------------
    # AUTHORITATIVE EVENT SOURCE
    # ---------------------------------------------------------------

    events = get_timeline_events(
        profile
    )

    temporal = extract_temporal_features(
        events
    )

    conditions = extract_condition_features(
        profile,
        events,
    )

    allergies = extract_allergy_features(
        profile,
        events,
    )

    medications = extract_medication_features(
        profile,
        events,
    )

    observations = extract_observation_features(
        profile,
        events,
    )

    profile_counts = extract_profile_counts(
        profile
    )

    # ---------------------------------------------------------------
    # Resource counts directly from timeline
    # ---------------------------------------------------------------

    encounter_count = count_resource_events(
        events,
        "Encounter",
    )

    procedure_count = count_resource_events(
        events,
        "Procedure",
    )

    diagnostic_report_count = count_resource_events(
        events,
        "DiagnosticReport",
    )

    immunization_count = count_resource_events(
        events,
        "Immunization",
    )

    care_plan_count = count_resource_events(
        events,
        "CarePlan",
    )

    # ---------------------------------------------------------------
    # Total events
    #
    # THIS MUST REPRESENT THE ACTUAL TIMELINE.
    # ---------------------------------------------------------------

    total_event_count = len(events)

    # ---------------------------------------------------------------
    # Build feature row
    # ---------------------------------------------------------------

    features = {

        "patient_id":
            patient_id,

        # -----------------------------------------------------------
        # Timeline
        # -----------------------------------------------------------

        **temporal,

        # -----------------------------------------------------------
        # Conditions
        # -----------------------------------------------------------

        **conditions,

        # -----------------------------------------------------------
        # Allergies
        # -----------------------------------------------------------

        **allergies,

        # -----------------------------------------------------------
        # Medications
        # -----------------------------------------------------------

        **medications,

        # -----------------------------------------------------------
        # Observations
        # -----------------------------------------------------------

        **observations,

        # -----------------------------------------------------------
        # Event-level resource counts
        # -----------------------------------------------------------

        "encounter_event_count":
            encounter_count,

        "procedure_event_count":
            procedure_count,

        "diagnostic_report_event_count":
            diagnostic_report_count,

        "immunization_event_count":
            immunization_count,

        "care_plan_event_count":
            care_plan_count,

        # -----------------------------------------------------------
        # Profile-level counts
        # -----------------------------------------------------------

        **profile_counts,

        # -----------------------------------------------------------
        # Authoritative total
        # -----------------------------------------------------------

        "total_event_count":
            total_event_count,
    }

    return features


# =====================================================================
# JSON CLEANING
# =====================================================================

def clean_for_json(
    value: Any,
) -> Any:

    if isinstance(value, dict):

        return {
            str(key):
                clean_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, list):

        return [
            clean_for_json(item)
            for item in value
        ]

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

    return value


# =====================================================================
# CSV WRITER
# =====================================================================

def write_csv(
    features: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Write standards-compliant CSV.

    This fixes the previous problem where lists/strings containing
    commas were manually concatenated and produced extra columns.
    """

    if not features:

        raise ValueError(
            "No features available for CSV output."
        )

    columns = list(
        features[0].keys()
    )

    with output_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
        )

        writer.writeheader()

        for feature in features:

            row = {}

            for column in columns:

                value = feature.get(
                    column
                )

                if isinstance(
                    value,
                    (list, dict),
                ):

                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                row[column] = value

            writer.writerow(row)


# =====================================================================
# OUTPUT VALIDATION
# =====================================================================

def validate_features(
    features: list[dict[str, Any]],
) -> None:

    if not features:

        raise ValueError(
            "No clinical features were generated."
        )

    patient_ids = [
        clean_string(
            feature.get(
                "patient_id"
            )
        )
        for feature in features
    ]

    missing_ids = [
        patient_id
        for patient_id in patient_ids
        if not patient_id
    ]

    duplicate_count = (
        len(patient_ids)
        - len(set(patient_ids))
    )

    if missing_ids:

        raise ValueError(
            f"{len(missing_ids)} patients "
            "have missing IDs."
        )

    if duplicate_count > 0:

        raise ValueError(
            f"{duplicate_count} duplicate "
            "patient IDs found."
        )

    # ---------------------------------------------------------------
    # Numeric feature validation
    # ---------------------------------------------------------------

    numeric_fields = [

        "timeline_days",
        "dated_event_count",
        "undated_event_count",
        "events_per_year",

        "recent_event_count_365d",
        "recent_medication_event_count_365d",
        "recent_observation_event_count_365d",
        "recent_condition_event_count_365d",
        "recent_encounter_count_365d",

        "condition_count_profile",
        "unique_condition_count",
        "condition_event_count",

        "allergy_count",
        "allergy_event_count",

        "unique_medication_count",
        "medication_profile_count",
        "medication_event_count",
        "active_medication_status_count",
        "completed_medication_status_count",

        "observation_count_profile",
        "observation_event_count",
        "unique_observation_count",
        "observations_with_numeric_values",
        "observations_with_units",
        "unique_observation_codes",

        "encounter_event_count",
        "procedure_event_count",
        "diagnostic_report_event_count",
        "immunization_event_count",
        "care_plan_event_count",

        "profile_condition_count",
        "profile_allergy_count",
        "profile_medication_event_count",
        "profile_observation_count",
        "profile_procedure_count",
        "profile_diagnostic_report_count",
        "profile_immunization_count",
        "profile_encounter_count",

        "total_event_count",
    ]

    for feature in features:

        for field in numeric_fields:

            value = feature.get(field)

            if value is None:
                continue

            if not isinstance(
                value,
                (int, float),
            ):

                raise ValueError(
                    f"Non-numeric value found "
                    f"in {field}: {value!r}"
                )


# =====================================================================
# SOURCE CONSISTENCY CHECK
# =====================================================================

def validate_against_source(
    profiles: list[dict[str, Any]],
    features: list[dict[str, Any]],
) -> dict[str, int]:

    feature_map = {
        feature["patient_id"]: feature
        for feature in features
    }

    mismatches = {
        "total_event_count": 0,
        "medication_event_count": 0,
        "observation_event_count": 0,
        "condition_event_count": 0,
        "encounter_event_count": 0,
        "procedure_event_count": 0,
        "diagnostic_report_event_count": 0,
        "immunization_event_count": 0,
    }

    for profile in profiles:

        patient_id = clean_string(
            profile.get(
                "patient_id"
            )
        )

        feature = feature_map.get(
            patient_id
        )

        if feature is None:
            continue

        events = get_timeline_events(
            profile
        )

        expected_total = len(events)

        if feature[
            "total_event_count"
        ] != expected_total:

            mismatches[
                "total_event_count"
            ] += 1

        expected_medications = sum(
            1
            for event in events
            if event_resource_type(event)
            in MEDICATION_RESOURCE_TYPES
        )

        if feature[
            "medication_event_count"
        ] != expected_medications:

            mismatches[
                "medication_event_count"
            ] += 1

        expected_observations = sum(
            1
            for event in events
            if event_resource_type(event)
            == "Observation"
        )

        if feature[
            "observation_event_count"
        ] != expected_observations:

            mismatches[
                "observation_event_count"
            ] += 1

        expected_conditions = sum(
            1
            for event in events
            if event_resource_type(event)
            == "Condition"
        )

        if feature[
            "condition_event_count"
        ] != expected_conditions:

            mismatches[
                "condition_event_count"
            ] += 1

        expected_encounters = sum(
            1
            for event in events
            if event_resource_type(event)
            == "Encounter"
        )

        if feature[
            "encounter_event_count"
        ] != expected_encounters:

            mismatches[
                "encounter_event_count"
            ] += 1

        expected_procedures = sum(
            1
            for event in events
            if event_resource_type(event)
            == "Procedure"
        )

        if feature[
            "procedure_event_count"
        ] != expected_procedures:

            mismatches[
                "procedure_event_count"
            ] += 1

        expected_reports = sum(
            1
            for event in events
            if event_resource_type(event)
            == "DiagnosticReport"
        )

        if feature[
            "diagnostic_report_event_count"
        ] != expected_reports:

            mismatches[
                "diagnostic_report_event_count"
            ] += 1

        expected_immunizations = sum(
            1
            for event in events
            if event_resource_type(event)
            == "Immunization"
        )

        if feature[
            "immunization_event_count"
        ] != expected_immunizations:

            mismatches[
                "immunization_event_count"
            ] += 1

    return mismatches


# =====================================================================
# GLOBAL TOTALS
# =====================================================================

def calculate_global_totals(
    features: list[dict[str, Any]],
) -> dict[str, int]:

    fields = [

        "total_event_count",
        "medication_event_count",
        "observation_event_count",
        "condition_event_count",
        "encounter_event_count",
        "procedure_event_count",
        "diagnostic_report_event_count",
        "immunization_event_count",
        "care_plan_event_count",
        "allergy_event_count",
    ]

    totals = {}

    for field in fields:

        totals[field] = sum(
            safe_int(
                feature.get(field)
            )
            for feature in features
        )

    return totals


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print("=" * 70)
    print(
        "ELDERDOCAI CLINICAL FEATURE BUILDER"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Input check
    # ---------------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Patient profile file not found:\n"
            f"{INPUT_FILE}"
        )

    print()
    print(
        f"Input: {INPUT_FILE}"
    )

    # ---------------------------------------------------------------
    # Load profiles
    # ---------------------------------------------------------------

    print()
    print(
        "Loading patient profiles..."
    )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        profiles = json.load(file)

    if not isinstance(
        profiles,
        list,
    ):

        raise ValueError(
            "patient_profiles.json must "
            "contain a list."
        )

    print(
        f"Profiles loaded: {len(profiles):,}"
    )

    # ---------------------------------------------------------------
    # Source timeline statistics
    # ---------------------------------------------------------------

    source_timeline_total = sum(
        len(
            get_timeline_events(profile)
        )
        for profile in profiles
        if isinstance(profile, dict)
    )

    print()
    print(
        f"Source timeline events: "
        f"{source_timeline_total:,}"
    )

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------

    print()
    print(
        "Building clinical features..."
    )

    features = []

    for index, profile in enumerate(
        profiles,
        start=1,
    ):

        if not isinstance(
            profile,
            dict,
        ):

            print(
                f"[WARN] Skipping invalid "
                f"profile {index}."
            )

            continue

        feature = build_patient_features(
            profile
        )

        features.append(
            feature
        )

        if (
            index % 25 == 0
            or index == len(profiles)
        ):

            print(
                f"  Processed "
                f"{index:,}/"
                f"{len(profiles):,}"
            )

    # ---------------------------------------------------------------
    # Validate structure
    # ---------------------------------------------------------------

    print()
    print(
        "Validating feature structure..."
    )

    validate_features(
        features
    )

    print(
        "[PASS] Feature structure valid."
    )

    # ---------------------------------------------------------------
    # Source consistency
    # ---------------------------------------------------------------

    print()
    print(
        "Cross-checking against source..."
    )

    mismatches = validate_against_source(
        profiles,
        features,
    )

    for field, count in mismatches.items():

        if count == 0:

            print(
                f"[PASS] {field}: "
                f"0 mismatches"
            )

        else:

            print(
                f"[FAIL] {field}: "
                f"{count} mismatches"
            )

    if any(
        count > 0
        for count in mismatches.values()
    ):

        raise ValueError(
            "Source consistency validation failed."
        )

    # ---------------------------------------------------------------
    # Clean JSON
    # ---------------------------------------------------------------

    clean_features = [
        clean_for_json(feature)
        for feature in features
    ]

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    print()
    print(
        "Saving JSON..."
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            clean_features,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------

    print(
        "Saving CSV..."
    )

    write_csv(
        clean_features,
        OUTPUT_CSV,
    )

    # ---------------------------------------------------------------
    # Global totals
    # ---------------------------------------------------------------

    totals = calculate_global_totals(
        clean_features
    )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FEATURE BUILD COMPLETE"
    )
    print("=" * 70)

    print(
        f"Patients processed:          "
        f"{len(clean_features):,}"
    )

    print()
    print(
        "GLOBAL EVENT TOTALS"
    )

    print(
        f"Timeline events:             "
        f"{totals['total_event_count']:,}"
    )

    print(
        f"Medication events:           "
        f"{totals['medication_event_count']:,}"
    )

    print(
        f"Observation events:          "
        f"{totals['observation_event_count']:,}"
    )

    print(
        f"Condition events:            "
        f"{totals['condition_event_count']:,}"
    )

    print(
        f"Encounter events:            "
        f"{totals['encounter_event_count']:,}"
    )

    print(
        f"Procedure events:            "
        f"{totals['procedure_event_count']:,}"
    )

    print(
        f"Diagnostic reports:          "
        f"{totals['diagnostic_report_event_count']:,}"
    )

    print(
        f"Immunizations:               "
        f"{totals['immunization_event_count']:,}"
    )

    print(
        f"Care plans:                  "
        f"{totals['care_plan_event_count']:,}"
    )

    print(
        f"Allergy events:              "
        f"{totals['allergy_event_count']:,}"
    )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"CSV:  {OUTPUT_CSV}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print()
    print(
        "[PASS] Clinical features successfully built."
    )

    print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()