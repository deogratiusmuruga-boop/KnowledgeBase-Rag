"""
ElderDocAI Patient Profile Builder
===================================

Purpose
-------
Convert the resolved Synthea longitudinal patient timeline into
one structured patient profile per patient.

Input
-----
elderdocai/processed/patient_timeline_resolved.csv

Outputs
-------
1. elderdocai/processed/patient_profiles.csv
2. elderdocai/processed/patient_profiles.json

The original timeline is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================
# PATHS
# =============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "patient_timeline_resolved.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "patient_profiles.csv"
)

OUTPUT_JSON = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "patient_profiles.json"
)


# =============================================================
# EXPECTED COLUMNS
# =============================================================

EXPECTED_COLUMNS = [
    "patient_id",
    "resource_type",
    "event_date",
    "date_source",
    "code",
    "value",
    "unit",
    "status",
]


# =============================================================
# HELPERS
# =============================================================

def clean_value(value: Any) -> str | None:
    """
    Convert pandas values into clean strings.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def unique_clean_values(
    series: pd.Series,
) -> list[str]:
    """
    Return unique non-empty values while preserving order.
    """

    result = []

    seen = set()

    for value in series:

        value = clean_value(value)

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def count_nonempty(series: pd.Series) -> int:
    """
    Count non-empty values.
    """

    count = 0

    for value in series:

        if clean_value(value) is not None:
            count += 1

    return count


def build_observation_summary(
    patient_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Build a compact observation summary.

    Each unique observation code is represented with:

    - observation name
    - number of measurements
    - latest date
    - latest value
    - latest unit
    """

    observations = patient_df[
        patient_df["resource_type"] == "Observation"
    ].copy()

    if observations.empty:
        return []

    observations = observations[
        observations["code"].notna()
    ]

    if observations.empty:
        return []

    summaries = []

    for code, group in observations.groupby(
        "code",
        sort=False,
    ):

        code = clean_value(code)

        if not code:
            continue

        group = group.sort_values(
            "event_date",
            na_position="last",
        )

        latest = group.iloc[-1]

        summaries.append(
            {
                "code": code,
                "measurement_count": int(len(group)),
                "latest_date": clean_value(
                    latest["event_date"]
                ),
                "latest_value": clean_value(
                    latest["value"]
                ),
                "latest_unit": clean_value(
                    latest["unit"]
                ),
            }
        )

    return summaries


def build_medication_summary(
    patient_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Build medication-level summary.

    Combines MedicationRequest and
    MedicationAdministration events.
    """

    medications = patient_df[
        patient_df["resource_type"].isin(
            [
                "MedicationRequest",
                "MedicationAdministration",
            ]
        )
    ].copy()

    if medications.empty:
        return []

    medications = medications[
        medications["code"].notna()
    ]

    if medications.empty:
        return []

    summaries = []

    for medication, group in medications.groupby(
        "code",
        sort=False,
    ):

        medication = clean_value(medication)

        if not medication:
            continue

        group = group.sort_values(
            "event_date",
            na_position="last",
        )

        latest = group.iloc[-1]

        statuses = unique_clean_values(
            group["status"]
        )

        summaries.append(
            {
                "medication": medication,
                "event_count": int(len(group)),
                "resource_types": unique_clean_values(
                    group["resource_type"]
                ),
                "statuses": statuses,
                "latest_date": clean_value(
                    latest["event_date"]
                ),
                "latest_status": clean_value(
                    latest["status"]
                ),
            }
        )

    return summaries


def build_resource_counts(
    patient_df: pd.DataFrame,
) -> dict[str, int]:
    """
    Count events by FHIR resource type.
    """

    counts = (
        patient_df["resource_type"]
        .value_counts()
        .to_dict()
    )

    return {
        str(resource_type): int(count)
        for resource_type, count in counts.items()
    }


def build_profile(
    patient_id: str,
    patient_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build one longitudinal profile for one patient.
    """

    patient_df = patient_df.copy()

    patient_df["event_date"] = pd.to_datetime(
        patient_df["event_date"],
        errors="coerce",
    )

    patient_df = patient_df.sort_values(
        "event_date",
        na_position="last",
    )

    dated_events = patient_df[
        patient_df["event_date"].notna()
    ]

    # ---------------------------------------------------------
    # Timeline coverage
    # ---------------------------------------------------------

    if not dated_events.empty:

        first_date = dated_events[
            "event_date"
        ].min()

        last_date = dated_events[
            "event_date"
        ].max()

        timeline_days = (
            last_date - first_date
        ).days

        first_date_str = first_date.strftime(
            "%Y-%m-%d"
        )

        last_date_str = last_date.strftime(
            "%Y-%m-%d"
        )

    else:

        first_date_str = None
        last_date_str = None
        timeline_days = 0

    # ---------------------------------------------------------
    # Conditions
    # ---------------------------------------------------------

    conditions = unique_clean_values(
        patient_df.loc[
            patient_df["resource_type"] == "Condition",
            "code",
        ]
    )

    # ---------------------------------------------------------
    # Allergies
    # ---------------------------------------------------------

    allergies = unique_clean_values(
        patient_df.loc[
            patient_df["resource_type"]
            == "AllergyIntolerance",
            "code",
        ]
    )

    # ---------------------------------------------------------
    # Immunizations
    # ---------------------------------------------------------

    immunizations = unique_clean_values(
        patient_df.loc[
            patient_df["resource_type"]
            == "Immunization",
            "code",
        ]
    )

    # ---------------------------------------------------------
    # Procedures
    # ---------------------------------------------------------

    procedures = unique_clean_values(
        patient_df.loc[
            patient_df["resource_type"]
            == "Procedure",
            "code",
        ]
    )

    # ---------------------------------------------------------
    # Diagnostic reports
    # ---------------------------------------------------------

    diagnostic_reports = unique_clean_values(
        patient_df.loc[
            patient_df["resource_type"]
            == "DiagnosticReport",
            "code",
        ]
    )

    # ---------------------------------------------------------
    # Care plans
    # ---------------------------------------------------------

    care_plans = unique_clean_values(
        patient_df.loc[
            patient_df["resource_type"]
            == "CarePlan",
            "code",
        ]
    )

    # ---------------------------------------------------------
    # Encounters
    # ---------------------------------------------------------

    encounter_count = int(
        (
            patient_df["resource_type"]
            == "Encounter"
        ).sum()
    )

    # ---------------------------------------------------------
    # Observations
    # ---------------------------------------------------------

    observation_mask = (
        patient_df["resource_type"]
        == "Observation"
    )

    observation_count = int(
        observation_mask.sum()
    )

    observations_with_values = count_nonempty(
        patient_df.loc[
            observation_mask,
            "value",
        ]
    )

    observations = build_observation_summary(
        patient_df
    )

    # ---------------------------------------------------------
    # Medications
    # ---------------------------------------------------------

    medication_mask = patient_df[
        "resource_type"
    ].isin(
        [
            "MedicationRequest",
            "MedicationAdministration",
        ]
    )

    medication_event_count = int(
        medication_mask.sum()
    )

    medications = build_medication_summary(
        patient_df
    )

    # ---------------------------------------------------------
    # Timeline events
    # ---------------------------------------------------------

    timeline_events = []

    for _, row in patient_df.iterrows():

        event = {
            "date": clean_value(
                row["event_date"]
                if not pd.isna(row["event_date"])
                else None
            ),
            "resource_type": clean_value(
                row["resource_type"]
            ),
            "date_source": clean_value(
                row["date_source"]
            ),
            "code": clean_value(
                row["code"]
            ),
            "value": clean_value(
                row["value"]
            ),
            "unit": clean_value(
                row["unit"]
            ),
            "status": clean_value(
                row["status"]
            ),
        }

        timeline_events.append(event)

    # ---------------------------------------------------------
    # Profile
    # ---------------------------------------------------------

    profile = {
        "patient_id": patient_id,

        "timeline": {
            "first_event_date": first_date_str,
            "last_event_date": last_date_str,
            "timeline_days": int(timeline_days),
            "total_events": int(len(patient_df)),
            "dated_events": int(len(dated_events)),
            "undated_events": int(
                len(patient_df) - len(dated_events)
            ),
        },

        "conditions": conditions,

        "allergies": allergies,

        "medications": medications,

        "medication_event_count": medication_event_count,

        "observations": observations,

        "observation_count": observation_count,

        "observations_with_values": (
            observations_with_values
        ),

        "procedures": procedures,

        "diagnostic_reports": diagnostic_reports,

        "immunizations": immunizations,

        "care_plans": care_plans,

        "encounter_count": encounter_count,

        "resource_counts": build_resource_counts(
            patient_df
        ),

        "timeline_events": timeline_events,
    }

    return profile


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("ELDERDOCAI PATIENT PROFILE BUILDER")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print()
    print("Loading resolved patient timeline...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------
    # Parse dates
    # ---------------------------------------------------------

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Patient count
    # ---------------------------------------------------------

    patient_ids = (
        df["patient_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"Patients found: {len(patient_ids):,}"
    )

    # ---------------------------------------------------------
    # Build profiles
    # ---------------------------------------------------------

    profiles = []

    print()
    print("Building patient profiles...")

    for number, patient_id in enumerate(
        patient_ids,
        start=1,
    ):

        patient_df = df[
            df["patient_id"] == patient_id
        ]

        profile = build_profile(
            patient_id,
            patient_df,
        )

        profiles.append(profile)

        if number % 25 == 0:

            print(
                f"  Processed {number:,}/"
                f"{len(patient_ids):,} patients"
            )

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving JSON profiles...")

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            profiles,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Build CSV summary
    # ---------------------------------------------------------

    print("Building CSV profile summary...")

    csv_rows = []

    for profile in profiles:

        csv_rows.append(
            {
                "patient_id": profile["patient_id"],

                "first_event_date": profile[
                    "timeline"
                ]["first_event_date"],

                "last_event_date": profile[
                    "timeline"
                ]["last_event_date"],

                "timeline_days": profile[
                    "timeline"
                ]["timeline_days"],

                "total_events": profile[
                    "timeline"
                ]["total_events"],

                "dated_events": profile[
                    "timeline"
                ]["dated_events"],

                "undated_events": profile[
                    "timeline"
                ]["undated_events"],

                "condition_count": len(
                    profile["conditions"]
                ),

                "allergy_count": len(
                    profile["allergies"]
                ),

                "medication_count": len(
                    profile["medications"]
                ),

                "medication_event_count": profile[
                    "medication_event_count"
                ],

                "observation_count": profile[
                    "observation_count"
                ],

                "observations_with_values": profile[
                    "observations_with_values"
                ],

                "procedure_count": len(
                    profile["procedures"]
                ),

                "diagnostic_report_count": len(
                    profile["diagnostic_reports"]
                ),

                "immunization_count": len(
                    profile["immunizations"]
                ),

                "care_plan_count": len(
                    profile["care_plans"]
                ),

                "encounter_count": profile[
                    "encounter_count"
                ],

                "resource_counts": json.dumps(
                    profile["resource_counts"],
                    ensure_ascii=False,
                ),

                "conditions": "; ".join(
                    profile["conditions"]
                ),

                "allergies": "; ".join(
                    profile["allergies"]
                ),

                "medications": "; ".join(
                    medication["medication"]
                    for medication
                    in profile["medications"]
                ),

                "procedures": "; ".join(
                    profile["procedures"]
                ),

                "diagnostic_reports": "; ".join(
                    profile["diagnostic_reports"]
                ),

                "immunizations": "; ".join(
                    profile["immunizations"]
                ),

                "care_plans": "; ".join(
                    profile["care_plans"]
                ),
            }
        )

    profiles_df = pd.DataFrame(
        csv_rows
    )

    profiles_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total_profiles = len(profiles)

    total_events = len(df)

    total_medications = sum(
        profile["medication_event_count"]
        for profile in profiles
    )

    total_observations = sum(
        profile["observation_count"]
        for profile in profiles
    )

    total_conditions = sum(
        len(profile["conditions"])
        for profile in profiles
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ELDERDOCAI PATIENT PROFILE BUILDER")
    print("=" * 70)

    print(
        f"Patients processed:          "
        f"{total_profiles:,}"
    )

    print(
        f"Timeline events processed:  "
        f"{total_events:,}"
    )

    print(
        f"Medication events:          "
        f"{total_medications:,}"
    )

    print(
        f"Observation events:         "
        f"{total_observations:,}"
    )

    print(
        f"Unique condition entries:   "
        f"{total_conditions:,}"
    )

    print()
    print(
        f"CSV output:  {OUTPUT_CSV}"
    )

    print(
        f"JSON output: {OUTPUT_JSON}"
    )

    print()
    print("Output structure:")

    print(
        "  patient_profiles.csv"
    )

    print(
        "    → One summary row per patient"
    )

    print(
        "  patient_profiles.json"
    )

    print(
        "    → Full patient profiles"
    )

    print(
        "    → Includes complete chronological events"
    )

    print()
    print("[PASS] Patient profiles successfully built.")

    print("=" * 70)


if __name__ == "__main__":
    main()