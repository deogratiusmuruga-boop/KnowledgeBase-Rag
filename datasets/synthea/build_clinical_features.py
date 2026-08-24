"""
======================================================================
ELDERDOCAI CLINICAL FEATURE BUILDER
======================================================================

Purpose:
    Build structured clinical features from the validated ElderDocAI
    patient profiles.

Input:
    elderdocai/processed/patient_profiles.json

Outputs:
    elderdocai/processed/clinical_features.csv
    elderdocai/processed/clinical_features.json

The feature layer captures:

    - Patient timeline information
    - Condition history
    - Allergy history
    - Medication history
    - Medication activity/status
    - Observation statistics
    - Laboratory measurements
    - Encounter activity
    - Procedure activity
    - Diagnostic report activity
    - Immunization history
    - Temporal activity
    - Recent clinical events
    - Resource distribution

This script does NOT make clinical diagnoses or risk predictions.
It only converts the validated patient profile data into structured
features for subsequent ElderDocAI processing.
======================================================================
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


# ---------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = (
    BASE_DIR
    / "elderdocai"
    / "processed"
)

INPUT_FILE = PROCESSED_DIR / "patient_profiles.json"

OUTPUT_CSV = PROCESSED_DIR / "clinical_features.csv"
OUTPUT_JSON = PROCESSED_DIR / "clinical_features.json"


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

RECENT_DAYS = 365

MEDICATION_RESOURCE_TYPES = {
    "MedicationRequest",
    "MedicationAdministration",
}

OBSERVATION_RESOURCE_TYPE = "Observation"

CONDITION_RESOURCE_TYPE = "Condition"

ENCOUNTER_RESOURCE_TYPE = "Encounter"

PROCEDURE_RESOURCE_TYPE = "Procedure"

DIAGNOSTIC_REPORT_RESOURCE_TYPE = "DiagnosticReport"

IMMUNIZATION_RESOURCE_TYPE = "Immunization"

ALLERGY_RESOURCE_TYPE = "AllergyIntolerance"

CARE_PLAN_RESOURCE_TYPE = "CarePlan"


# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""

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


def clean_string(value: Any) -> str:
    """Return a normalized string."""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def parse_date(value: Any) -> datetime | None:
    """
    Parse a date/time string safely.

    Supports common FHIR date formats such as:

        YYYY-MM-DD
        YYYY-MM-DDTHH:MM:SS
        YYYY-MM-DDTHH:MM:SSZ
        YYYY-MM-DDTHH:MM:SS+09:00

    Important:
        All returned datetime objects are normalized to naive UTC
        datetimes. This prevents comparisons between timezone-aware
        and timezone-naive datetime objects.
    """

    if value is None:
        return None

    text = clean_string(value)

    if not text:
        return None

    # ---------------------------------------------------------------
    # ISO parsing
    # ---------------------------------------------------------------

    try:
        # Python fromisoformat understands offsets such as +09:00.
        # Convert trailing Z into explicit UTC.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)

        # -----------------------------------------------------------
        # Normalize timezone-aware datetime to naive UTC.
        # -----------------------------------------------------------

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
    # Date-only fallback
    # ---------------------------------------------------------------

    try:
        return datetime.strptime(
            text[:10],
            "%Y-%m-%d",
        )

    except (ValueError, TypeError):
        return None


def days_between(
    start: datetime | None,
    end: datetime | None,
) -> int:
    """Return number of days between two dates."""

    if start is None or end is None:
        return 0

    return max(
        0,
        (end - start).days,
    )


def unique_nonempty(
    values: list[Any],
) -> list[str]:
    """Return unique non-empty strings while preserving order."""

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


def get_list(
    profile: dict[str, Any],
    key: str,
) -> list:
    """Safely retrieve a list from a profile."""

    value = profile.get(
        key,
        [],
    )

    if isinstance(value, list):
        return value

    return []


def get_dict(
    profile: dict[str, Any],
    key: str,
) -> dict:
    """Safely retrieve a dictionary from a profile."""

    value = profile.get(
        key,
        {},
    )

    if isinstance(value, dict):
        return value

    return {}


# ---------------------------------------------------------------------
# EVENT HELPERS
# ---------------------------------------------------------------------

def event_resource_type(
    event: dict[str, Any],
) -> str:
    """Extract event resource type."""

    return clean_string(
        event.get("resource_type")
        or event.get("resourceType")
    )


def event_date(
    event: dict[str, Any],
) -> datetime | None:
    """Extract and parse an event date."""

    return parse_date(
        event.get("event_date")
        or event.get("date")
        or event.get("effectiveDateTime")
    )


def event_code(
    event: dict[str, Any],
) -> str:
    """Extract event code/name."""

    return clean_string(
        event.get("code")
        or event.get("display")
        or event.get("name")
    )


def event_value(
    event: dict[str, Any],
) -> Any:
    """Extract event value."""

    if "value" in event:
        return event.get("value")

    return None


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


def is_numeric_value(
    value: Any,
) -> bool:
    """Determine whether an event value is numeric."""

    return safe_float(value) is not None


# ---------------------------------------------------------------------
# PROFILE TIMELINE
# ---------------------------------------------------------------------

def get_timeline_events(
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Retrieve timeline events from a patient profile."""

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


def sort_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort events chronologically."""

    return sorted(
        events,
        key=lambda event: (
            event_date(event) is None,
            event_date(event) or datetime.max,
        ),
    )


# ---------------------------------------------------------------------
# CONDITION FEATURES
# ---------------------------------------------------------------------

def extract_condition_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract condition-related features."""

    condition_entries = get_list(
        profile,
        "conditions",
    )

    condition_names = []

    for item in condition_entries:

        if isinstance(item, str):

            condition_names.append(item)

        elif isinstance(item, dict):

            name = (
                item.get("code")
                or item.get("name")
                or item.get("display")
                or item.get("description")
            )

            if name:
                condition_names.append(name)

    condition_names = unique_nonempty(
        condition_names
    )

    condition_events = [
        event
        for event in events
        if event_resource_type(event)
        == CONDITION_RESOURCE_TYPE
    ]

    event_condition_names = unique_nonempty(
        [
            event_code(event)
            for event in condition_events
        ]
    )

    all_conditions = unique_nonempty(
        condition_names
        + event_condition_names
    )

    return {
        "condition_count_profile": len(
            condition_entries
        ),
        "unique_condition_count": len(
            all_conditions
        ),
        "condition_event_count": len(
            condition_events
        ),
        "conditions": all_conditions,
        "recent_condition_count": 0,
    }


# ---------------------------------------------------------------------
# ALLERGY FEATURES
# ---------------------------------------------------------------------

def extract_allergy_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract allergy-related features."""

    allergy_entries = get_list(
        profile,
        "allergies",
    )

    allergy_names = []

    for item in allergy_entries:

        if isinstance(item, str):

            allergy_names.append(item)

        elif isinstance(item, dict):

            name = (
                item.get("code")
                or item.get("name")
                or item.get("display")
                or item.get("description")
            )

            if name:
                allergy_names.append(name)

    allergy_events = [
        event
        for event in events
        if event_resource_type(event)
        == ALLERGY_RESOURCE_TYPE
    ]

    allergy_names.extend(
        event_code(event)
        for event in allergy_events
    )

    allergy_names = unique_nonempty(
        allergy_names
    )

    return {
        "allergy_count": len(
            allergy_names
        ),
        "allergies": allergy_names,
    }


# ---------------------------------------------------------------------
# MEDICATION FEATURES
# ---------------------------------------------------------------------

def extract_medication_name(
    item: Any,
) -> str:
    """Extract medication name from profile medication entry."""

    if isinstance(item, str):
        return clean_string(item)

    if not isinstance(item, dict):
        return ""

    return clean_string(
        item.get("code")
        or item.get("name")
        or item.get("display")
        or item.get("medication")
        or item.get("medication_name")
    )


def extract_medication_status(
    item: Any,
) -> str:
    """Extract medication status."""

    if not isinstance(item, dict):
        return ""

    return clean_string(
        item.get("status")
    )


def extract_medication_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract medication-related features."""

    medication_entries = get_list(
        profile,
        "medications",
    )

    medication_names = [
        extract_medication_name(item)
        for item in medication_entries
    ]

    medication_names = unique_nonempty(
        medication_names
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

    status_counter = Counter()

    for item in medication_entries:

        status = extract_medication_status(item)

        if status:
            status_counter[
                status.lower()
            ] += 1

    for event in medication_events:

        status = event_status(event)

        if status:
            status_counter[
                status.lower()
            ] += 1

    active_medications = status_counter.get(
        "active",
        0,
    )

    completed_medications = status_counter.get(
        "completed",
        0,
    )

    medication_event_dates = [
        event_date(event)
        for event in medication_events
    ]

    medication_event_dates = [
        date
        for date in medication_event_dates
        if date is not None
    ]

    medication_first_date = (
        min(
            medication_event_dates
        )
        .date()
        .isoformat()
        if medication_event_dates
        else ""
    )

    medication_last_date = (
        max(
            medication_event_dates
        )
        .date()
        .isoformat()
        if medication_event_dates
        else ""
    )

    return {
        "unique_medication_count": len(
            all_medications
        ),
        "medication_profile_count": len(
            medication_entries
        ),
        "medication_event_count": len(
            medication_events
        ),
        "active_medication_status_count":
            active_medications,
        "completed_medication_status_count":
            completed_medications,
        "medication_first_date":
            medication_first_date,
        "medication_last_date":
            medication_last_date,
        "medications":
            all_medications,
    }


# ---------------------------------------------------------------------
# OBSERVATION FEATURES
# ---------------------------------------------------------------------

def extract_observation_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract observation and laboratory features."""

    observation_entries = get_list(
        profile,
        "observations",
    )

    observation_events = [
        event
        for event in events
        if event_resource_type(event)
        == OBSERVATION_RESOURCE_TYPE
    ]

    observation_codes = unique_nonempty(
        [
            event_code(event)
            for event in observation_events
        ]
    )

    numeric_values = []

    units = []

    dated_observations = []

    for event in observation_events:

        date = event_date(event)

        if date is not None:
            dated_observations.append(date)

        value = event_value(event)

        numeric_value = safe_float(value)

        if numeric_value is not None:
            numeric_values.append(
                numeric_value
            )

        unit = event_unit(event)

        if unit:
            units.append(unit)

    latest_observation_date = ""

    if dated_observations:

        latest_observation_date = (
            max(dated_observations)
            .date()
            .isoformat()
        )

    return {
        "observation_count_profile":
            len(observation_entries),

        "observation_event_count":
            len(observation_events),

        "unique_observation_count":
            len(observation_codes),

        "observations_with_numeric_values":
            len(numeric_values),

        "observations_with_units":
            len(unique_nonempty(units)),

        "numeric_observation_min":
            (
                min(numeric_values)
                if numeric_values
                else None
            ),

        "numeric_observation_max":
            (
                max(numeric_values)
                if numeric_values
                else None
            ),

        "numeric_observation_mean":
            (
                mean(numeric_values)
                if numeric_values
                else None
            ),

        "numeric_observation_median":
            (
                median(numeric_values)
                if numeric_values
                else None
            ),

        "unique_observation_codes":
            len(observation_codes),

        "latest_observation_date":
            latest_observation_date,
    }


# ---------------------------------------------------------------------
# GENERAL RESOURCE FEATURES
# ---------------------------------------------------------------------

def count_resource_events(
    events: list[dict[str, Any]],
    resource_type: str,
) -> int:
    """Count events of a specific FHIR resource type."""

    return sum(
        1
        for event in events
        if event_resource_type(event)
        == resource_type
    )


def extract_resource_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract general resource activity."""

    return {
        "encounter_event_count":
            count_resource_events(
                events,
                ENCOUNTER_RESOURCE_TYPE,
            ),

        "procedure_event_count":
            count_resource_events(
                events,
                PROCEDURE_RESOURCE_TYPE,
            ),

        "diagnostic_report_event_count":
            count_resource_events(
                events,
                DIAGNOSTIC_REPORT_RESOURCE_TYPE,
            ),

        "immunization_event_count":
            count_resource_events(
                events,
                IMMUNIZATION_RESOURCE_TYPE,
            ),

        "care_plan_event_count":
            count_resource_events(
                events,
                CARE_PLAN_RESOURCE_TYPE,
            ),

        "allergy_event_count":
            count_resource_events(
                events,
                ALLERGY_RESOURCE_TYPE,
            ),
    }


# ---------------------------------------------------------------------
# TEMPORAL FEATURES
# ---------------------------------------------------------------------

def extract_temporal_features(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract temporal activity features.

    Important Windows compatibility fix:
        This function does NOT use datetime.timestamp().

    Some Synthea patients have historical events before 1970.
    On Windows, calling timestamp() on such dates can produce:

        OSError: [Errno 22] Invalid argument

    Therefore recent-event calculations use datetime.timedelta
    directly instead of converting dates to Unix timestamps.
    """

    dated_events = [
        event
        for event in events
        if event_date(event) is not None
    ]

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
    # No dated events
    # ---------------------------------------------------------------

    if not dates:

        return {
            "first_event_date": "",
            "last_event_date": "",
            "timeline_days": 0,
            "dated_event_count": 0,
            "undated_event_count": len(events),
            "events_per_year": 0.0,
            "recent_event_count_365d": 0,
            "recent_medication_event_count_365d": 0,
            "recent_observation_event_count_365d": 0,
            "recent_condition_event_count_365d": 0,
            "recent_encounter_count_365d": 0,
        }

    # ---------------------------------------------------------------
    # Timeline boundaries
    # ---------------------------------------------------------------

    first_date = min(dates)

    last_date = max(dates)

    timeline_days = days_between(
        first_date,
        last_date,
    )

    # ---------------------------------------------------------------
    # Events per year
    # ---------------------------------------------------------------

    events_per_year = 0.0

    if timeline_days > 0:

        events_per_year = (
            len(dated_events)
            / (timeline_days / 365.25)
        )

    # ---------------------------------------------------------------
    # RECENT EVENT CALCULATION
    #
    # IMPORTANT:
    # Do NOT use .timestamp().
    #
    # This works correctly for historical dates such as:
    #
    #   1939
    #   1940
    #   1950
    #
    # and therefore works safely on Windows.
    # ---------------------------------------------------------------

    recent_cutoff = (
        last_date
        - timedelta(days=RECENT_DAYS)
    )

    # ---------------------------------------------------------------
    # Find recent events
    # ---------------------------------------------------------------

    recent_events = []

    for event in dated_events:

        date = event_date(event)

        if date is None:
            continue

        if date >= recent_cutoff:

            recent_events.append(event)

    # ---------------------------------------------------------------
    # Recent medication events
    # ---------------------------------------------------------------

    recent_medications = [
        event
        for event in recent_events
        if event_resource_type(event)
        in MEDICATION_RESOURCE_TYPES
    ]

    # ---------------------------------------------------------------
    # Recent observation events
    # ---------------------------------------------------------------

    recent_observations = [
        event
        for event in recent_events
        if event_resource_type(event)
        == OBSERVATION_RESOURCE_TYPE
    ]

    # ---------------------------------------------------------------
    # Recent condition events
    # ---------------------------------------------------------------

    recent_conditions = [
        event
        for event in recent_events
        if event_resource_type(event)
        == CONDITION_RESOURCE_TYPE
    ]

    # ---------------------------------------------------------------
    # Recent encounter events
    # ---------------------------------------------------------------

    recent_encounters = [
        event
        for event in recent_events
        if event_resource_type(event)
        == ENCOUNTER_RESOURCE_TYPE
    ]

    # ---------------------------------------------------------------
    # Return temporal features
    # ---------------------------------------------------------------

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
            len(events) - len(dated_events),

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


# ---------------------------------------------------------------------
# RESOURCE COUNTS FROM PROFILE
# ---------------------------------------------------------------------

def extract_profile_counts(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Extract existing profile-level counts."""

    resource_counts = get_dict(
        profile,
        "resource_counts",
    )

    return {
        "profile_condition_count":
            safe_int(
                profile.get(
                    "condition_count"
                ),
                safe_int(
                    profile.get(
                        "condition_count_profile"
                    ),
                    0,
                ),
            ),

        "profile_allergy_count":
            safe_int(
                profile.get(
                    "allergy_count"
                ),
                0,
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
                safe_int(
                    profile.get(
                        "procedure_count"
                    ),
                    0,
                ),
            ),

        "profile_diagnostic_report_count":
            safe_int(
                resource_counts.get(
                    "DiagnosticReport"
                ),
                safe_int(
                    profile.get(
                        "diagnostic_report_count"
                    ),
                    0,
                ),
            ),

        "profile_immunization_count":
            safe_int(
                resource_counts.get(
                    "Immunization"
                ),
                safe_int(
                    profile.get(
                        "immunization_count"
                    ),
                    0,
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


# ---------------------------------------------------------------------
# PATIENT FEATURE BUILDER
# ---------------------------------------------------------------------

def build_patient_features(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete feature representation for one patient."""

    patient_id = clean_string(
        profile.get("patient_id")
    )

    events = get_timeline_events(
        profile
    )

    events = sort_events(
        events
    )

    # ---------------------------------------------------------------
    # Feature groups
    # ---------------------------------------------------------------

    timeline_features = (
        extract_temporal_features(
            profile,
            events,
        )
    )

    condition_features = (
        extract_condition_features(
            profile,
            events,
        )
    )

    allergy_features = (
        extract_allergy_features(
            profile,
            events,
        )
    )

    medication_features = (
        extract_medication_features(
            profile,
            events,
        )
    )

    observation_features = (
        extract_observation_features(
            profile,
            events,
        )
    )

    resource_features = (
        extract_resource_features(
            profile,
            events,
        )
    )

    profile_counts = (
        extract_profile_counts(
            profile,
        )
    )

    # ---------------------------------------------------------------
    # Combine features
    # ---------------------------------------------------------------

    features = {

        "patient_id":
            patient_id,

        # Timeline
        **timeline_features,

        # Conditions
        **condition_features,

        # Allergies
        **allergy_features,

        # Medications
        **medication_features,

        # Observations
        **observation_features,

        # General clinical activity
        **resource_features,

        # Existing profile counts
        **profile_counts,

        # Patient-level total event count
        "total_event_count":
            len(events),
    }

    return features


# ---------------------------------------------------------------------
# JSON SERIALIZATION
# ---------------------------------------------------------------------

def clean_for_json(
    value: Any,
) -> Any:
    """Convert values into JSON-safe representations."""

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

        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return None

        return value

    return value


# ---------------------------------------------------------------------
# CSV SERIALIZATION
# ---------------------------------------------------------------------

def csv_escape(
    value: Any,
) -> str:
    """
    Convert a Python value to a CSV-safe representation.

    Lists and dictionaries are stored as JSON strings so that
    medications, conditions, and similar feature arrays remain
    recoverable.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        (list, dict),
    ):

        return json.dumps(
            value,
            ensure_ascii=False,
        )

    text = str(value)

    # Standard CSV escaping.
    if any(
        character in text
        for character in [
            ",",
            '"',
            "\n",
            "\r",
        ]
    ):

        text = (
            '"'
            + text.replace(
                '"',
                '""',
            )
            + '"'
        )

    return text


def write_csv(
    features: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Write patient features to CSV without requiring pandas."""

    if not features:

        raise ValueError(
            "No patient features available for CSV output."
        )

    columns = list(
        features[0].keys()
    )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        file.write(
            ",".join(columns)
            + "\n"
        )

        for row in features:

            values = [
                csv_escape(
                    row.get(column)
                )
                for column in columns
            ]

            file.write(
                ",".join(values)
                + "\n"
            )


# ---------------------------------------------------------------------
# INTEGRITY CHECK
# ---------------------------------------------------------------------

def validate_output(
    features: list[dict[str, Any]],
) -> None:
    """Perform basic output integrity checks."""

    patient_ids = [
        feature.get(
            "patient_id",
            "",
        )
        for feature in features
    ]

    missing_ids = sum(
        1
        for patient_id in patient_ids
        if not clean_string(
            patient_id
        )
    )

    duplicate_ids = (
        len(patient_ids)
        - len(set(patient_ids))
    )

    if missing_ids > 0:

        raise ValueError(
            f"Output contains "
            f"{missing_ids} missing patient IDs."
        )

    if duplicate_ids > 0:

        raise ValueError(
            f"Output contains "
            f"{duplicate_ids} duplicate patient IDs."
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("ELDERDOCAI CLINICAL FEATURE BUILDER")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Check input
    # ---------------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Patient profile file not found:\n"
            f"{INPUT_FILE}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Input file:")
    print(
        f"  {INPUT_FILE}"
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
            "patient_profiles.json "
            "must contain a list of profiles."
        )

    print(
        f"Profiles loaded:          "
        f"{len(profiles):,}"
    )

    # ---------------------------------------------------------------
    # Build features
    # ---------------------------------------------------------------

    print()
    print(
        "Building clinical features..."
    )

    feature_rows = []

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
                f"profile at index {index}."
            )

            continue

        features = (
            build_patient_features(
                profile
            )
        )

        feature_rows.append(
            features
        )

        if (
            index % 25 == 0
            or index == len(profiles)
        ):

            print(
                f"  Processed "
                f"{index:,}/"
                f"{len(profiles):,} patients"
            )

    # ---------------------------------------------------------------
    # Integrity validation
    # ---------------------------------------------------------------

    print()
    print(
        "Checking feature integrity..."
    )

    validate_output(
        feature_rows
    )

    print(
        "[PASS] Patient feature "
        "structure is valid."
    )

    # ---------------------------------------------------------------
    # Clean features for serialization
    # ---------------------------------------------------------------

    clean_features = [
        clean_for_json(
            feature
        )
        for feature in feature_rows
    ]

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------

    print()
    print(
        "Saving JSON features..."
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
        "Saving CSV features..."
    )

    write_csv(
        clean_features,
        OUTPUT_CSV,
    )

    # ---------------------------------------------------------------
    # Calculate summary statistics
    # ---------------------------------------------------------------

    total_events = sum(
        safe_int(
            feature.get(
                "total_event_count"
            )
        )
        for feature in clean_features
    )

    total_medication_events = sum(
        safe_int(
            feature.get(
                "medication_event_count"
            )
        )
        for feature in clean_features
    )

    total_observation_events = sum(
        safe_int(
            feature.get(
                "observation_event_count"
            )
        )
        for feature in clean_features
    )

    total_condition_events = sum(
        safe_int(
            feature.get(
                "condition_event_count"
            )
        )
        for feature in clean_features
    )

    total_encounters = sum(
        safe_int(
            feature.get(
                "encounter_event_count"
            )
        )
        for feature in clean_features
    )

    total_procedures = sum(
        safe_int(
            feature.get(
                "procedure_event_count"
            )
        )
        for feature in clean_features
    )

    total_diagnostic_reports = sum(
        safe_int(
            feature.get(
                "diagnostic_report_event_count"
            )
        )
        for feature in clean_features
    )

    total_immunizations = sum(
        safe_int(
            feature.get(
                "immunization_event_count"
            )
        )
        for feature in clean_features
    )

    total_numeric_observations = sum(
        safe_int(
            feature.get(
                "observations_with_numeric_values"
            )
        )
        for feature in clean_features
    )

    # ---------------------------------------------------------------
    # Recent activity statistics
    # ---------------------------------------------------------------

    total_recent_events = sum(
        safe_int(
            feature.get(
                "recent_event_count_365d"
            )
        )
        for feature in clean_features
    )

    total_recent_medications = sum(
        safe_int(
            feature.get(
                "recent_medication_event_count_365d"
            )
        )
        for feature in clean_features
    )

    total_recent_observations = sum(
        safe_int(
            feature.get(
                "recent_observation_event_count_365d"
            )
        )
        for feature in clean_features
    )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ELDERDOCAI CLINICAL FEATURE BUILDER"
    )
    print("=" * 70)

    print(
        f"Patients processed:             "
        f"{len(clean_features):,}"
    )

    print(
        f"Timeline events represented:    "
        f"{total_events:,}"
    )

    print(
        f"Medication events:              "
        f"{total_medication_events:,}"
    )

    print(
        f"Observation events:             "
        f"{total_observation_events:,}"
    )

    print(
        f"Numeric observations:           "
        f"{total_numeric_observations:,}"
    )

    print(
        f"Condition events:               "
        f"{total_condition_events:,}"
    )

    print(
        f"Encounter events:               "
        f"{total_encounters:,}"
    )

    print(
        f"Procedure events:               "
        f"{total_procedures:,}"
    )

    print(
        f"Diagnostic report events:       "
        f"{total_diagnostic_reports:,}"
    )

    print(
        f"Immunization events:            "
        f"{total_immunizations:,}"
    )

    print(
        f"Recent 365-day events:           "
        f"{total_recent_events:,}"
    )

    print(
        f"Recent 365-day medication events:"
        f" {total_recent_medications:,}"
    )

    print(
        f"Recent 365-day observations:     "
        f"{total_recent_observations:,}"
    )

    print()
    print(
        f"CSV output:  {OUTPUT_CSV}"
    )

    print(
        f"JSON output: {OUTPUT_JSON}"
    )

    print()
    print("Feature groups:")

    print(
        "  - Timeline features"
    )

    print(
        "  - Condition features"
    )

    print(
        "  - Allergy features"
    )

    print(
        "  - Medication features"
    )

    print(
        "  - Observation features"
    )

    print(
        "  - Encounter features"
    )

    print(
        "  - Procedure features"
    )

    print(
        "  - Diagnostic report features"
    )

    print(
        "  - Immunization features"
    )

    print(
        "  - Temporal activity features"
    )

    print(
        "  - Recent 365-day activity"
    )

    print(
        "  - Resource distribution"
    )

    print()
    print(
        "[PASS] Clinical features "
        "successfully built."
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()