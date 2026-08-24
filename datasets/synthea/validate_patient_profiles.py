import json
from pathlib import Path

import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "elderdocai" / "processed"

CSV_FILE = PROCESSED_DIR / "patient_profiles.csv"
JSON_FILE = PROCESSED_DIR / "patient_profiles.json"


EXPECTED_CSV_COLUMNS = [
    "patient_id",
    "first_event_date",
    "last_event_date",
    "timeline_days",
    "total_events",
    "dated_events",
    "undated_events",
    "condition_count",
    "allergy_count",
    "medication_count",
    "medication_event_count",
    "observation_count",
    "observations_with_values",
    "procedure_count",
    "diagnostic_report_count",
    "immunization_count",
    "care_plan_count",
    "encounter_count",
    "resource_counts",
    "conditions",
    "allergies",
    "medications",
    "procedures",
    "diagnostic_reports",
    "immunizations",
    "care_plans",
]


# =====================================================================
# HELPERS
# =====================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def is_missing(value):
    if value is None:
        return True

    if isinstance(value, float) and pd.isna(value):
        return True

    if isinstance(value, str) and not value.strip():
        return True

    return False


def normalize_profiles(data):
    """
    Normalize supported JSON profile structures.

    Expected current structure:
        [
            {
                "patient_id": "...",
                ...
            }
        ]

    Also supports:
        {"profiles": [...]}
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("profiles"), list):
            return data["profiles"]

        if isinstance(data.get("patient_profiles"), list):
            return data["patient_profiles"]

    return []


def get_patient_id(profile):
    if not isinstance(profile, dict):
        return None

    return profile.get("patient_id")


def as_list(value):
    if isinstance(value, list):
        return value

    if value is None:
        return []

    return [value]


def extract_event_date(event):
    if not isinstance(event, dict):
        return None

    for key in [
        "event_date",
        "date",
        "effectiveDateTime",
        "performedPeriod.start",
        "period.start",
        "authoredOn",
        "onsetDateTime",
        "occurrenceDateTime",
        "recordedDate",
    ]:
        value = event.get(key)

        if not is_missing(value):
            return value

    return None


def extract_resource_type(event):
    if not isinstance(event, dict):
        return None

    return (
        event.get("resource_type")
        or event.get("resourceType")
        or event.get("type")
    )


def extract_code(event):
    if not isinstance(event, dict):
        return None

    for key in [
        "code",
        "name",
        "medication",
        "medication_name",
        "display",
    ]:
        value = event.get(key)

        if not is_missing(value):
            return value

    return None


def extract_value(event):
    if not isinstance(event, dict):
        return None

    for key in [
        "value",
        "valueQuantity",
        "valueString",
        "valueCodeableConcept",
        "result",
    ]:
        value = event.get(key)

        if not is_missing(value):
            return value

    return None


def extract_unit(event):
    if not isinstance(event, dict):
        return None

    for key in [
        "unit",
        "value_unit",
        "valueUnit",
    ]:
        value = event.get(key)

        if not is_missing(value):
            return value

    value_quantity = event.get("valueQuantity")

    if isinstance(value_quantity, dict):
        return value_quantity.get("unit")

    return None


def validate_event_list(events):
    """
    Validate timeline_events.

    Returns:
        total_events
        events_without_date
        events_without_resource
    """

    total = 0
    without_date = 0
    without_resource = 0

    for event in events:

        if not isinstance(event, dict):
            continue

        total += 1

        if is_missing(extract_event_date(event)):
            without_date += 1

        if is_missing(extract_resource_type(event)):
            without_resource += 1

    return total, without_date, without_resource


# =====================================================================
# START
# =====================================================================

print("=" * 70)
print("ELDERDOCAI PATIENT PROFILE VALIDATOR")
print("=" * 70)

print(f"CSV input:  {CSV_FILE}")
print(f"JSON input: {JSON_FILE}")


# =====================================================================
# 1. FILE VALIDATION
# =====================================================================

print_header("1. FILE VALIDATION")

if CSV_FILE.exists():
    print("[PASS] patient_profiles.csv exists.")
else:
    print("[FAIL] patient_profiles.csv does not exist.")

if JSON_FILE.exists():
    print("[PASS] patient_profiles.json exists.")
else:
    print("[FAIL] patient_profiles.json does not exist.")


if not CSV_FILE.exists() or not JSON_FILE.exists():
    print()
    print("[FAIL] Required input files are missing.")
    raise SystemExit(1)


# =====================================================================
# 2. LOAD CSV
# =====================================================================

print_header("2. CSV PROFILE VALIDATION")

try:
    df = pd.read_csv(CSV_FILE)

    print(f"Rows:       {len(df):,}")
    print(f"Columns:    {len(df.columns)}")
    print(f"File size:  {CSV_FILE.stat().st_size:,} bytes")

    print()
    print("CSV columns:")

    for column in df.columns:
        print(f"  - {column}")

except Exception as exc:
    print(f"[FAIL] Could not load CSV: {exc}")
    raise SystemExit(1)


# =====================================================================
# 3. CSV COLUMN STRUCTURE
# =====================================================================

print_header("3. CSV COLUMN STRUCTURE")

missing_columns = [
    column
    for column in EXPECTED_CSV_COLUMNS
    if column not in df.columns
]

additional_columns = [
    column
    for column in df.columns
    if column not in EXPECTED_CSV_COLUMNS
]

if not missing_columns:
    print("[PASS] All expected CSV columns are present.")
else:
    print("[WARN] Some expected columns are missing:")

    for column in missing_columns:
        print(f"  - {column}")

if additional_columns:
    print()
    print("Additional columns:")

    for column in additional_columns:
        print(f"  - {column}")

print()
print(f"Total columns found: {len(df.columns)}")


# =====================================================================
# 4. PATIENT ID VALIDATION
# =====================================================================

print_header("4. PATIENT ID VALIDATION")

patient_ids = df["patient_id"]

missing_ids = patient_ids.isna().sum()

empty_ids = (
    patient_ids.astype(str)
    .str.strip()
    .eq("")
    .sum()
)

duplicate_ids = patient_ids.duplicated().sum()

unique_ids = patient_ids.nunique()

print(f"Patient rows:       {len(df):,}")
print(f"Unique patient IDs: {unique_ids:,}")
print(f"Missing IDs:        {missing_ids:,}")
print(f"Empty IDs:          {empty_ids:,}")
print(f"Duplicate IDs:      {duplicate_ids:,}")

if missing_ids == 0 and empty_ids == 0:
    print("[PASS] No missing or empty patient IDs.")
else:
    print("[FAIL] Missing or empty patient IDs detected.")

if duplicate_ids == 0:
    print("[PASS] One summary row per patient.")
else:
    print("[FAIL] Duplicate patient IDs detected.")


# =====================================================================
# 5. CSV MISSING VALUE ANALYSIS
# =====================================================================

print_header("5. CSV MISSING VALUE ANALYSIS")

for column in df.columns:

    missing = df[column].isna().sum()

    percentage = (
        missing / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{column:<30}: "
        f"{missing:>8,} "
        f"({percentage:>6.2f}%)"
    )


# =====================================================================
# 6. PROFILE STATISTICS
# =====================================================================

print_header("6. PROFILE STATISTICS")

numeric_fields = [
    "timeline_days",
    "total_events",
    "dated_events",
    "undated_events",
    "condition_count",
    "allergy_count",
    "medication_count",
    "medication_event_count",
    "observation_count",
    "observations_with_values",
    "procedure_count",
    "diagnostic_report_count",
    "immunization_count",
    "care_plan_count",
    "encounter_count",
]

available_numeric_fields = [
    field
    for field in numeric_fields
    if field in df.columns
]

print("Available numeric profile fields:")

for field in available_numeric_fields:

    series = pd.to_numeric(
        df[field],
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        continue

    print()
    print(f"{field}:")
    print(f"  Minimum: {series.min():,.2f}")
    print(f"  Maximum: {series.max():,.2f}")
    print(f"  Mean:    {series.mean():,.2f}")
    print(f"  Median:  {series.median():,.2f}")


# =====================================================================
# 7. DEMOGRAPHIC VALIDATION
# =====================================================================

print_header("7. DEMOGRAPHIC VALIDATION")

demographic_columns = [
    "birth_date",
    "age",
    "death_date",
    "gender",
]

found_demographics = [
    column
    for column in demographic_columns
    if column in df.columns
]

if found_demographics:

    for column in found_demographics:

        missing = df[column].isna().sum()

        print(
            f"{column:<15}: "
            f"{len(df) - missing:,} present, "
            f"{missing:,} missing"
        )

else:

    print(
        "[INFO] Demographic fields such as "
        "birth_date, age, death_date, and gender "
        "are not part of the current CSV summary schema."
    )


# =====================================================================
# 8. LOAD JSON
# =====================================================================

print_header("8. JSON PROFILE VALIDATION")

try:

    with open(
        JSON_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        json_data = json.load(file)

    print(f"Root type: {type(json_data).__name__}")

    profiles = normalize_profiles(json_data)

    print(f"Profiles:  {len(profiles):,}")

    if not profiles:
        print("[FAIL] No patient profiles found in JSON.")

except Exception as exc:

    print(f"[FAIL] Could not load JSON: {exc}")
    raise SystemExit(1)


# =====================================================================
# 9. JSON PATIENT ID VALIDATION
# =====================================================================

print_header("9. JSON PATIENT ID VALIDATION")

json_patient_ids = [
    get_patient_id(profile)
    for profile in profiles
]

json_missing_ids = sum(
    1
    for patient_id in json_patient_ids
    if is_missing(patient_id)
)

json_unique_ids = len(
    {
        patient_id
        for patient_id in json_patient_ids
        if not is_missing(patient_id)
    }
)

print(f"JSON patient profiles:    {len(profiles):,}")
print(f"JSON patient IDs:         {len(json_patient_ids):,}")
print(f"Unique JSON patient IDs:  {json_unique_ids:,}")
print(f"Profiles without ID:      {json_missing_ids:,}")

if json_missing_ids == 0:
    print("[PASS] All JSON profiles contain patient IDs.")
else:
    print("[FAIL] Some JSON profiles are missing patient IDs.")


# =====================================================================
# 10. CSV ↔ JSON CONSISTENCY
# =====================================================================

print_header("10. CSV ↔ JSON CONSISTENCY")

csv_ids = set(
    df["patient_id"]
    .dropna()
    .astype(str)
)

json_ids = {
    str(patient_id)
    for patient_id in json_patient_ids
    if not is_missing(patient_id)
}

csv_not_json = csv_ids - json_ids
json_not_csv = json_ids - csv_ids

print(f"CSV patients: {len(csv_ids):,}")
print(f"JSON patients: {len(json_ids):,}")

print(
    f"Patients in CSV but not JSON: {len(csv_not_json):,}"
)

print(
    f"Patients in JSON but not CSV: {len(json_not_csv):,}"
)

if not csv_not_json and not json_not_csv:
    print("[PASS] CSV and JSON contain the same patient IDs.")
else:
    print("[FAIL] CSV and JSON patient IDs do not match.")


# =====================================================================
# 11. JSON PROFILE CONTENT VALIDATION
# =====================================================================

print_header("11. JSON PROFILE CONTENT VALIDATION")

profiles_with_events = 0
profiles_with_medications = 0
profiles_with_observations = 0
profiles_with_conditions = 0

total_events = 0
total_medications = 0
total_observations = 0
total_conditions = 0

for profile in profiles:

    if not isinstance(profile, dict):
        continue

    timeline_events = profile.get(
        "timeline_events",
        []
    )

    medications = as_list(
        profile.get("medications", [])
    )

    observations = as_list(
        profile.get("observations", [])
    )

    conditions = as_list(
        profile.get("conditions", [])
    )

    if timeline_events:
        profiles_with_events += 1

    if medications:
        profiles_with_medications += 1

    if observations:
        profiles_with_observations += 1

    if conditions:
        profiles_with_conditions += 1

    total_events += len(timeline_events)
    total_medications += len(medications)
    total_observations += len(observations)
    total_conditions += len(conditions)


print(
    f"Profiles with timeline_events: {profiles_with_events:,}"
)

print(
    f"Profiles with medications:      {profiles_with_medications:,}"
)

print(
    f"Profiles with observations:     {profiles_with_observations:,}"
)

print(
    f"Profiles with conditions:       {profiles_with_conditions:,}"
)

print(
    f"Total JSON timeline events:     {total_events:,}"
)

print(
    f"Total JSON medications:         {total_medications:,}"
)

print(
    f"Total JSON observations:        {total_observations:,}"
)

print(
    f"Total JSON conditions:          {total_conditions:,}"
)

if profiles_with_events > 0:
    print("[PASS] Timeline events are present.")
else:
    print("[FAIL] No timeline_events found.")


# =====================================================================
# 12. SAMPLE PATIENT PROFILE
# =====================================================================

print_header("12. SAMPLE PATIENT PROFILE")

sample_profile = None

for profile in profiles:

    if isinstance(profile, dict):
        sample_profile = profile
        break

if sample_profile:

    sample_patient_id = sample_profile.get(
        "patient_id"
    )

    print(f"Patient ID: {sample_patient_id}")

    print()
    print("Profile fields:")

    for key, value in sample_profile.items():

        if isinstance(value, list):

            print(
                f"  - {key}: "
                f"list ({len(value):,} items)"
            )

        elif isinstance(value, dict):

            print(
                f"  - {key}: "
                f"dictionary ({len(value):,} fields)"
            )

        else:

            print(
                f"  - {key}: "
                f"{type(value).__name__}"
            )

else:

    print("[FAIL] Could not locate a sample profile.")


# =====================================================================
# 13. EVENT STRUCTURE VALIDATION
# =====================================================================

print_header("13. EVENT STRUCTURE VALIDATION")

sampled_events = 0
events_without_date = 0
events_without_resource = 0

for profile in profiles:

    if not isinstance(profile, dict):
        continue

    events = profile.get(
        "timeline_events",
        []
    )

    for event in events[:25]:

        if not isinstance(event, dict):
            continue

        sampled_events += 1

        if is_missing(
            extract_event_date(event)
        ):
            events_without_date += 1

        if is_missing(
            extract_resource_type(event)
        ):
            events_without_resource += 1


print(
    f"Events sampled:             {sampled_events:,}"
)

print(
    f"Events without date:         {events_without_date:,}"
)

print(
    f"Events without resource:     {events_without_resource:,}"
)

if sampled_events == 0:

    print(
        "[FAIL] No timeline events were found for sampling."
    )

elif events_without_resource == 0:

    print(
        "[PASS] Sampled timeline events contain resource types."
    )

else:

    print(
        "[WARN] Some sampled timeline events lack resource types."
    )


# =====================================================================
# 14. MEDICATION PROFILE VALIDATION
# =====================================================================

print_header("14. MEDICATION PROFILE VALIDATION")

medication_entries = []
medications_without_code = 0

for profile in profiles:

    if not isinstance(profile, dict):
        continue

    medications = as_list(
        profile.get("medications", [])
    )

    for medication in medications:

        medication_entries.append(
            medication
        )

        if isinstance(medication, dict):

            code = extract_code(medication)

            if is_missing(code):
                medications_without_code += 1

        elif is_missing(medication):

            medications_without_code += 1


print(
    f"Medication entries in JSON: {len(medication_entries):,}"
)

print(
    f"Medications without code/name: "
    f"{medications_without_code:,}"
)

if medication_entries:

    if medications_without_code == 0:

        print(
            "[PASS] All medication entries contain "
            "a medication name/code."
        )

    else:

        percentage = (
            medications_without_code
            / len(medication_entries)
            * 100
        )

        print(
            f"[INFO] {percentage:.2f}% of medication entries "
            "do not expose a separate code field."
        )

        print(
            "[INFO] This is acceptable if medications are "
            "stored directly as medication names."
        )

else:

    print("[WARN] No medication entries found.")


# =====================================================================
# 15. OBSERVATION PROFILE VALIDATION
# =====================================================================

print_header("15. OBSERVATION PROFILE VALIDATION")

observation_entries = []

observations_with_values = 0
observations_with_units = 0
observations_with_codes = 0

for profile in profiles:

    if not isinstance(profile, dict):
        continue

    observations = as_list(
        profile.get("observations", [])
    )

    for observation in observations:

        observation_entries.append(
            observation
        )

        if isinstance(observation, dict):

            if not is_missing(
                extract_value(observation)
            ):
                observations_with_values += 1

            if not is_missing(
                extract_unit(observation)
            ):
                observations_with_units += 1

            if not is_missing(
                extract_code(observation)
            ):
                observations_with_codes += 1

        elif not is_missing(observation):

            observations_with_values += 1


print(
    f"Observation entries in JSON: {len(observation_entries):,}"
)

print(
    f"Observations with values:     {observations_with_values:,}"
)

print(
    f"Observations with units:      {observations_with_units:,}"
)

print(
    f"Observations with codes:      {observations_with_codes:,}"
)

if observation_entries:

    print(
        "[PASS] Observation data is present."
    )

else:

    print(
        "[WARN] No observation entries found."
    )


# =====================================================================
# 16. TIMELINE COUNT CONSISTENCY
# =====================================================================

print_header("16. TIMELINE COUNT CONSISTENCY")

csv_event_count = None

if "total_events" in df.columns:

    csv_event_count = int(
        pd.to_numeric(
            df["total_events"],
            errors="coerce"
        ).fillna(0).sum()
    )

    print(
        f"CSV total_events sum:       {csv_event_count:,}"
    )

    print(
        f"JSON timeline_events total: {total_events:,}"
    )

    if csv_event_count == total_events:

        print(
            "[PASS] CSV and JSON timeline event counts match."
        )

    else:

        difference = (
            total_events - csv_event_count
        )

        print(
            f"[WARN] Timeline event count difference: "
            f"{difference:,}"
        )

else:

    print(
        "[INFO] total_events column is not present "
        "in CSV; skipping direct count comparison."
    )


# =====================================================================
# 17. PATIENT PROFILE TIMELINE QUALITY
# =====================================================================

print_header("17. PATIENT PROFILE TIMELINE QUALITY")

profiles_with_valid_timeline = 0
profiles_with_empty_timeline = 0

for profile in profiles:

    if not isinstance(profile, dict):
        continue

    events = profile.get(
        "timeline_events",
        []
    )

    if isinstance(events, list) and len(events) > 0:
        profiles_with_valid_timeline += 1
    else:
        profiles_with_empty_timeline += 1


print(
    f"Profiles with timeline events: "
    f"{profiles_with_valid_timeline:,}"
)

print(
    f"Profiles with empty timeline:   "
    f"{profiles_with_empty_timeline:,}"
)

if profiles_with_empty_timeline == 0:

    print(
        "[PASS] Every patient has timeline events."
    )

else:

    print(
        "[WARN] Some patients have empty timelines."
    )


# =====================================================================
# 18. VALIDATION SUMMARY
# =====================================================================

print_header("18. VALIDATION SUMMARY")

checks = []

# CSV
if len(df) > 0:
    print("[PASS] CSV contains patient profiles.")
    checks.append(True)
else:
    print("[FAIL] CSV contains no patient profiles.")
    checks.append(False)


# Patient IDs
if missing_ids == 0 and empty_ids == 0 and duplicate_ids == 0:

    print("[PASS] Patient IDs are valid.")
    checks.append(True)

else:

    print("[FAIL] Patient ID validation failed.")
    checks.append(False)


# JSON
if profiles:

    print("[PASS] JSON profiles are present.")
    checks.append(True)

else:

    print("[FAIL] JSON profiles are missing.")
    checks.append(False)


# CSV/JSON consistency
if not csv_not_json and not json_not_csv:

    print("[PASS] CSV and JSON patient counts/IDs match.")
    checks.append(True)

else:

    print("[FAIL] CSV and JSON patient IDs do not match.")
    checks.append(False)


# Timeline
if total_events > 0:

    print(
        f"[PASS] Chronological timeline events found: "
        f"{total_events:,}"
    )

    checks.append(True)

else:

    print("[FAIL] No chronological timeline events found.")
    checks.append(False)


# Medications
if total_medications > 0:

    print(
        f"[PASS] Medication history is present: "
        f"{total_medications:,} entries."
    )

    checks.append(True)

else:

    print("[FAIL] Medication history is missing.")
    checks.append(False)


# Observations
if total_observations > 0:

    print(
        f"[PASS] Observation history is present: "
        f"{total_observations:,} entries."
    )

    checks.append(True)

else:

    print("[FAIL] Observation history is missing.")
    checks.append(False)


# Conditions
if total_conditions > 0:

    print(
        f"[PASS] Condition history is present: "
        f"{total_conditions:,} entries."
    )

    checks.append(True)

else:

    print("[WARN] No condition history found.")


# =====================================================================
# FINAL RESULT
# =====================================================================

print()
print("=" * 70)

if all(checks):

    print("VALIDATION RESULT: PASS")

    print(
        "The ElderDocAI patient profiles are structurally "
        "ready for the next processing stage."
    )

else:

    print("VALIDATION RESULT: REVIEW REQUIRED")

    print(
        "The patient profiles contain issues that should "
        "be reviewed before continuing."
    )

print("=" * 70)

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)