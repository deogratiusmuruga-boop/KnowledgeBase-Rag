"""
ElderDocAI Patient Timeline Validator

Validates:
    elderdocai/processed/patient_timeline.csv

Checks:
    1. Dataset structure
    2. Missing values
    3. Patient counts
    4. Resource-type distribution
    5. Date validity
    6. Chronological ordering
    7. Duplicate events
    8. Medication reference quality
    9. Observation quality
    10. Timeline coverage per patient
"""

from pathlib import Path
import re

import pandas as pd


# =============================================================
# Configuration
# =============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "patient_timeline.csv"
)


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
# Helpers
# =============================================================

def print_section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_check(label: str, passed: bool, details: str = ""):
    symbol = "PASS" if passed else "WARN"

    if details:
        print(f"[{symbol}] {label}: {details}")
    else:
        print(f"[{symbol}] {label}")


# =============================================================
# Load dataset
# =============================================================

print_section("ELDERDOCAI PATIENT TIMELINE VALIDATOR")

print(f"Input file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nTimeline file was not found:\n{INPUT_FILE}\n\n"
        "Run build_patient_timeline.py first."
    )


df = pd.read_csv(INPUT_FILE)


# =============================================================
# Basic dataset information
# =============================================================

print_section("1. DATASET OVERVIEW")

print(f"Rows:              {len(df):,}")
print(f"Columns:           {len(df.columns)}")
print(f"Patients:          {df['patient_id'].nunique():,}")


# =============================================================
# Column validation
# =============================================================

print_section("2. COLUMN VALIDATION")

actual_columns = df.columns.tolist()

missing_columns = [
    column
    for column in EXPECTED_COLUMNS
    if column not in actual_columns
]

extra_columns = [
    column
    for column in actual_columns
    if column not in EXPECTED_COLUMNS
]


if not missing_columns:
    print_check(
        "Expected columns",
        True,
        "All expected columns are present.",
    )
else:
    print_check(
        "Expected columns",
        False,
        f"Missing: {missing_columns}",
    )


if extra_columns:
    print(
        f"Additional columns: {extra_columns}"
    )
else:
    print("Additional columns: none")


# =============================================================
# Missing-value analysis
# =============================================================

print_section("3. MISSING VALUE ANALYSIS")

missing = df.isna().sum()

for column in EXPECTED_COLUMNS:
    count = int(missing[column])

    percentage = (
        count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{column:15s}: "
        f"{count:10,} "
        f"({percentage:6.2f}%)"
    )


# =============================================================
# Patient ID validation
# =============================================================

print_section("4. PATIENT ID VALIDATION")

missing_patient_ids = int(
    df["patient_id"].isna().sum()
)

empty_patient_ids = int(
    (
        df["patient_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    ).sum()
)


print_check(
    "Missing patient IDs",
    missing_patient_ids == 0,
    f"{missing_patient_ids:,}",
)


print_check(
    "Empty patient IDs",
    empty_patient_ids == 0,
    f"{empty_patient_ids:,}",
)


patient_counts = df.groupby("patient_id").size()

print()
print(f"Minimum events per patient: {patient_counts.min():,}")
print(f"Maximum events per patient: {patient_counts.max():,}")
print(f"Average events per patient: {patient_counts.mean():,.2f}")
print(f"Median events per patient:  {patient_counts.median():,.0f}")


# =============================================================
# Resource type distribution
# =============================================================

print_section("5. RESOURCE TYPE DISTRIBUTION")

resource_counts = (
    df["resource_type"]
    .value_counts()
)

for resource_type, count in resource_counts.items():
    percentage = count / len(df) * 100

    print(
        f"{resource_type:30s} "
        f"{count:10,} "
        f"({percentage:6.2f}%)"
    )


# =============================================================
# Date validation
# =============================================================

print_section("6. DATE VALIDATION")

parsed_dates = pd.to_datetime(
    df["event_date"],
    errors="coerce",
)


invalid_date_mask = (
    df["event_date"].notna()
    & parsed_dates.isna()
)

invalid_date_count = int(
    invalid_date_mask.sum()
)

valid_date_count = int(
    parsed_dates.notna().sum()
)

missing_date_count = int(
    df["event_date"].isna().sum()
)


print_check(
    "Invalid dates",
    invalid_date_count == 0,
    f"{invalid_date_count:,}",
)

print(
    f"Valid dates:    {valid_date_count:,}"
)

print(
    f"Missing dates:  {missing_date_count:,}"
)


if valid_date_count > 0:

    print(
        f"Earliest date:  "
        f"{parsed_dates.min().date()}"
    )

    print(
        f"Latest date:    "
        f"{parsed_dates.max().date()}"
    )


# =============================================================
# Date source analysis
# =============================================================

print_section("7. DATE SOURCE DISTRIBUTION")

date_sources = (
    df["date_source"]
    .fillna("MISSING")
    .value_counts()
)

for source, count in date_sources.items():

    percentage = count / len(df) * 100

    print(
        f"{str(source):30s} "
        f"{count:10,} "
        f"({percentage:6.2f}%)"
    )


# =============================================================
# Chronological ordering
# =============================================================

print_section("8. CHRONOLOGICAL ORDER VALIDATION")


dated_df = df[
    df["event_date"].notna()
].copy()

dated_df["parsed_date"] = pd.to_datetime(
    dated_df["event_date"],
    errors="coerce",
)


out_of_order_patients = 0
total_patient_checks = 0


for patient_id, patient_df in dated_df.groupby(
    "patient_id",
    sort=False,
):

    total_patient_checks += 1

    dates = patient_df["parsed_date"]

    if not dates.is_monotonic_increasing:
        out_of_order_patients += 1


print(
    f"Patients checked:       {total_patient_checks:,}"
)

print(
    f"Patients out of order:  {out_of_order_patients:,}"
)


print_check(
    "Chronological ordering",
    out_of_order_patients == 0,
    (
        "All patient timelines are ordered."
        if out_of_order_patients == 0
        else
        f"{out_of_order_patients:,} patients contain out-of-order events."
    ),
)


# =============================================================
# Duplicate event detection
# =============================================================

print_section("9. DUPLICATE EVENT ANALYSIS")


duplicate_columns = [
    "patient_id",
    "resource_type",
    "event_date",
    "date_source",
    "code",
    "value",
    "unit",
    "status",
]


duplicate_mask = df.duplicated(
    subset=duplicate_columns,
    keep=False,
)


duplicate_rows = int(
    duplicate_mask.sum()
)

duplicate_groups = int(
    df[
        duplicate_mask
    ].drop_duplicates(
        subset=duplicate_columns
    ).shape[0]
)


print(
    f"Duplicate rows:    {duplicate_rows:,}"
)

print(
    f"Duplicate groups:  {duplicate_groups:,}"
)


print_check(
    "Duplicate events",
    duplicate_rows == 0,
    (
        "No exact duplicate events found."
        if duplicate_rows == 0
        else
        f"{duplicate_rows:,} rows belong to duplicate event groups."
    ),
)


if duplicate_rows > 0:

    print()
    print("Sample duplicate events:")

    print(
        df[
            duplicate_mask
        ][duplicate_columns]
        .head(10)
        .to_string(index=False)
    )


# =============================================================
# Medication analysis
# =============================================================

print_section("10. MEDICATION VALIDATION")


medication_types = [
    "MedicationRequest",
    "MedicationAdministration",
    "MedicationStatement",
]


medication_df = df[
    df["resource_type"].isin(
        medication_types
    )
].copy()


print(
    f"Medication events: "
    f"{len(medication_df):,}"
)


if len(medication_df) > 0:

    print()

    print(
        "Medication resource types:"
    )

    for resource_type, count in (
        medication_df["resource_type"]
        .value_counts()
        .items()
    ):

        print(
            f"  {resource_type:30s} "
            f"{count:,}"
        )


    # ---------------------------------------------------------
    # Detect unresolved UUID references
    # ---------------------------------------------------------

    code_series = (
        medication_df["code"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    uuid_pattern = re.compile(
        r"^(urn:uuid:)?"
        r"[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}$"
    )


    uuid_mask = code_series.apply(
        lambda value: bool(
            uuid_pattern.match(value)
        )
    )


    unresolved_uuid_count = int(
        uuid_mask.sum()
    )


    empty_medication_count = int(
        code_series.eq("").sum()
    )


    print()
    print(
        f"Medication codes:             "
        f"{len(code_series):,}"
    )

    print(
        f"Unresolved UUID references:    "
        f"{unresolved_uuid_count:,}"
    )

    print(
        f"Empty medication codes:        "
        f"{empty_medication_count:,}"
    )


    print_check(
        "Medication name resolution",
        unresolved_uuid_count == 0,
        (
            "All medication codes are resolved."
            if unresolved_uuid_count == 0
            else
            f"{unresolved_uuid_count:,} medication events still use UUID references."
        ),
    )


    # ---------------------------------------------------------
    # Medication status
    # ---------------------------------------------------------

    print()
    print("Medication status distribution:")

    medication_status = (
        medication_df["status"]
        .fillna("MISSING")
        .value_counts()
    )

    for status, count in medication_status.items():

        print(
            f"  {str(status):20s} "
            f"{count:,}"
        )


    # ---------------------------------------------------------
    # Medication samples
    # ---------------------------------------------------------

    print()
    print("Medication samples:")

    print(
        medication_df[
            [
                "patient_id",
                "resource_type",
                "event_date",
                "date_source",
                "code",
                "status",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


else:

    print_check(
        "Medication events",
        False,
        "No medication events found.",
    )


# =============================================================
# Observation analysis
# =============================================================

print_section("11. OBSERVATION VALIDATION")


observation_df = df[
    df["resource_type"] == "Observation"
].copy()


print(
    f"Observation events: "
    f"{len(observation_df):,}"
)


if len(observation_df) > 0:

    value_count = int(
        observation_df["value"]
        .notna()
        .sum()
    )

    unit_count = int(
        observation_df["unit"]
        .notna()
        .sum()
    )

    code_count = int(
        observation_df["code"]
        .notna()
        .sum()
    )


    print(
        f"Observations with values: "
        f"{value_count:,}"
    )

    print(
        f"Observations with units:  "
        f"{unit_count:,}"
    )

    print(
        f"Observations with codes:  "
        f"{code_count:,}"
    )


    print()
    print("Observation samples:")

    print(
        observation_df[
            [
                "patient_id",
                "event_date",
                "code",
                "value",
                "unit",
                "status",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


else:

    print_check(
        "Observation events",
        False,
        "No Observation events found.",
    )


# =============================================================
# Empty codes
# =============================================================

print_section("12. CODE QUALITY")


empty_codes = int(
    df["code"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)


print(
    f"Empty codes: {empty_codes:,}"
)


print_check(
    "Event codes",
    empty_codes == 0,
    (
        "All events have codes."
        if empty_codes == 0
        else
        f"{empty_codes:,} events have empty codes."
    ),
)


# =============================================================
# Patient timeline coverage
# =============================================================

print_section("13. PATIENT TIMELINE COVERAGE")


coverage = (
    dated_df
    .groupby("patient_id")["parsed_date"]
    .agg(
        first_event="min",
        last_event="max",
        event_count="count",
    )
)


print(
    f"Patients with dated events: "
    f"{len(coverage):,}"
)


if len(coverage) > 0:

    coverage["timeline_days"] = (
        coverage["last_event"]
        - coverage["first_event"]
    ).dt.days


    print(
        f"Shortest timeline: "
        f"{coverage['timeline_days'].min():,} days"
    )

    print(
        f"Longest timeline:  "
        f"{coverage['timeline_days'].max():,} days"
    )

    print(
        f"Average timeline:   "
        f"{coverage['timeline_days'].mean():,.2f} days"
    )

    print(
        f"Median timeline:    "
        f"{coverage['timeline_days'].median():,.0f} days"
    )


# =============================================================
# Event distribution per patient
# =============================================================

print_section("14. EVENTS PER PATIENT")


events_per_patient = (
    df.groupby("patient_id")
    .size()
)


print(
    f"Minimum:  {events_per_patient.min():,}"
)

print(
    f"Maximum:  {events_per_patient.max():,}"
)

print(
    f"Mean:     {events_per_patient.mean():,.2f}"
)

print(
    f"Median:   {events_per_patient.median():,.0f}"
)


# =============================================================
# Final assessment
# =============================================================

print_section("15. VALIDATION SUMMARY")


critical_checks = {
    "Expected columns": len(missing_columns) == 0,
    "Patient IDs": missing_patient_ids == 0,
    "Invalid dates": invalid_date_count == 0,
    "Medication events": len(medication_df) > 0,
    "Observation events": len(observation_df) > 0,
}


all_critical_passed = all(
    critical_checks.values()
)


for name, passed in critical_checks.items():

    print_check(
        name,
        passed,
    )


print()

if all_critical_passed:

    print(
        "VALIDATION RESULT: PASS"
    )

    print(
        "The patient timeline is structurally ready "
        "for the next ElderDocAI processing stage."
    )

else:

    print(
        "VALIDATION RESULT: REVIEW REQUIRED"
    )

    print(
        "One or more critical validation checks "
        "require investigation."
    )


print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)