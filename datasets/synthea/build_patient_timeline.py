import json
import csv
from pathlib import Path
from datetime import datetime


# ============================================================
# ELDERDOCAI PATIENT TIMELINE BUILDER
# ============================================================

FHIR_DIR = Path("elderdocai/fhir")
OUTPUT_DIR = Path("elderdocai/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "patient_timeline.csv"


# ============================================================
# TARGET FHIR RESOURCES
# ============================================================

TARGET_RESOURCES = {
    "Patient",
    "Observation",
    "Condition",
    "MedicationRequest",
    "MedicationAdministration",
    "DiagnosticReport",
    "Procedure",
    "Encounter",
    "CarePlan",
    "Immunization",
    "AllergyIntolerance",
}


# ============================================================
# HELPERS
# ============================================================

def safe_get(resource, *keys):
    """
    Safely retrieve nested dictionary values.

    Example:
        safe_get(resource, "period", "start")
    """

    current = resource

    for key in keys:

        if not isinstance(current, dict):
            return None

        current = current.get(key)

        if current is None:
            return None

    return current


# ============================================================
# CODE EXTRACTION
# ============================================================

def extract_code(resource):
    """
    Extract a human-readable clinical concept from a FHIR resource.

    Handles:
        code.text
        code.coding[].display
        code.coding[].code

    Also handles medicationCodeableConcept for:
        MedicationRequest
        MedicationAdministration
    """

    resource_type = resource.get("resourceType", "")

    # --------------------------------------------------------
    # MedicationRequest
    # --------------------------------------------------------

    if resource_type == "MedicationRequest":

        medication = resource.get("medicationCodeableConcept", {})

        if isinstance(medication, dict):

            text = medication.get("text")

            if text:
                return text

            coding = medication.get("coding", [])

            if coding:

                first = coding[0]

                return (
                    first.get("display")
                    or first.get("code")
                    or ""
                )

        # Some FHIR data may use medicationReference
        medication_reference = resource.get(
            "medicationReference",
            {}
        )

        if isinstance(medication_reference, dict):

            return (
                medication_reference.get("display")
                or medication_reference.get("reference")
                or ""
            )


    # --------------------------------------------------------
    # MedicationAdministration
    # --------------------------------------------------------

    if resource_type == "MedicationAdministration":

        medication = resource.get(
            "medicationCodeableConcept",
            {}
        )

        if isinstance(medication, dict):

            text = medication.get("text")

            if text:
                return text

            coding = medication.get("coding", [])

            if coding:

                first = coding[0]

                return (
                    first.get("display")
                    or first.get("code")
                    or ""
                )

        # Some FHIR data may use medicationReference
        medication_reference = resource.get(
            "medicationReference",
            {}
        )

        if isinstance(medication_reference, dict):

            return (
                medication_reference.get("display")
                or medication_reference.get("reference")
                or ""
            )


    # --------------------------------------------------------
    # Standard code field
    # --------------------------------------------------------

    code = resource.get("code", {})

    if isinstance(code, dict):

        text = code.get("text")

        if text:
            return text

        coding = code.get("coding", [])

        if coding:

            first = coding[0]

            return (
                first.get("display")
                or first.get("code")
                or ""
            )


    # --------------------------------------------------------
    # Immunization
    # --------------------------------------------------------

    if resource_type == "Immunization":

        vaccine = resource.get(
            "vaccineCode",
            {}
        )

        if isinstance(vaccine, dict):

            text = vaccine.get("text")

            if text:
                return text

            coding = vaccine.get("coding", [])

            if coding:

                first = coding[0]

                return (
                    first.get("display")
                    or first.get("code")
                    or ""
                )


    # --------------------------------------------------------
    # AllergyIntolerance
    # --------------------------------------------------------

    if resource_type == "AllergyIntolerance":

        code = resource.get("code", {})

        if isinstance(code, dict):

            text = code.get("text")

            if text:
                return text

            coding = code.get("coding", [])

            if coding:

                first = coding[0]

                return (
                    first.get("display")
                    or first.get("code")
                    or ""
                )


    return ""


# ============================================================
# OBSERVATION VALUE EXTRACTION
# ============================================================

def extract_value(resource):
    """
    Extract the actual clinical value from an Observation.

    Handles:

        valueQuantity
        valueCodeableConcept
        valueString
        valueBoolean
        valueInteger
        valueRange
        valueRatio
        valueDateTime

    Returns:
        value, unit
    """

    resource_type = resource.get("resourceType")

    if resource_type != "Observation":
        return "", ""


    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    quantity = resource.get("valueQuantity")

    if isinstance(quantity, dict):

        value = quantity.get("value")

        unit = (
            quantity.get("unit")
            or quantity.get("code")
            or ""
        )

        if value is not None:

            return str(value), str(unit)


    # --------------------------------------------------------
    # CodeableConcept
    # --------------------------------------------------------

    concept = resource.get(
        "valueCodeableConcept"
    )

    if isinstance(concept, dict):

        text = concept.get("text")

        if text:
            return text, ""

        coding = concept.get("coding", [])

        if coding:

            first = coding[0]

            value = (
                first.get("display")
                or first.get("code")
                or ""
            )

            return value, ""


    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    value_string = resource.get(
        "valueString"
    )

    if value_string is not None:

        return str(value_string), ""


    # --------------------------------------------------------
    # Boolean
    # --------------------------------------------------------

    value_boolean = resource.get(
        "valueBoolean"
    )

    if value_boolean is not None:

        return str(value_boolean), ""


    # --------------------------------------------------------
    # Integer
    # --------------------------------------------------------

    value_integer = resource.get(
        "valueInteger"
    )

    if value_integer is not None:

        return str(value_integer), ""


    # --------------------------------------------------------
    # DateTime
    # --------------------------------------------------------

    value_datetime = resource.get(
        "valueDateTime"
    )

    if value_datetime:

        return str(value_datetime), ""


    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    value_range = resource.get(
        "valueRange"
    )

    if isinstance(value_range, dict):

        low = value_range.get("low", {})
        high = value_range.get("high", {})

        low_value = (
            low.get("value")
            if isinstance(low, dict)
            else None
        )

        high_value = (
            high.get("value")
            if isinstance(high, dict)
            else None
        )

        unit = ""

        if isinstance(low, dict):

            unit = (
                low.get("unit")
                or low.get("code")
                or ""
            )

        if low_value is not None or high_value is not None:

            value = f"{low_value or ''}-{high_value or ''}"

            return value, unit


    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    value_ratio = resource.get(
        "valueRatio"
    )

    if isinstance(value_ratio, dict):

        numerator = value_ratio.get(
            "numerator",
            {}
        )

        denominator = value_ratio.get(
            "denominator",
            {}
        )

        if (
            isinstance(numerator, dict)
            and isinstance(denominator, dict)
        ):

            numerator_value = numerator.get(
                "value"
            )

            denominator_value = denominator.get(
                "value"
            )

            if (
                numerator_value is not None
                or denominator_value is not None
            ):

                value = (
                    f"{numerator_value or ''}/"
                    f"{denominator_value or ''}"
                )

                unit = (
                    numerator.get("unit")
                    or numerator.get("code")
                    or ""
                )

                return value, unit


    return "", ""


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_date(resource):
    """
    Extract the most clinically relevant timestamp available.

    Returns:
        date_value, date_source
    """

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    value = resource.get("effectiveDateTime")

    if value:
        return value, "effectiveDateTime"


    period = resource.get("effectivePeriod")

    if isinstance(period, dict):

        start = period.get("start")

        if start:
            return start, "effectivePeriod.start"


    # --------------------------------------------------------
    # MedicationRequest
    # --------------------------------------------------------

    value = resource.get("authoredOn")

    if value:
        return value, "authoredOn"


    # --------------------------------------------------------
    # Procedure
    # --------------------------------------------------------

    value = resource.get("performedDateTime")

    if value:
        return value, "performedDateTime"


    performed_period = resource.get(
        "performedPeriod"
    )

    if isinstance(performed_period, dict):

        start = performed_period.get("start")

        if start:
            return start, "performedPeriod.start"


    # --------------------------------------------------------
    # Generic occurrenceDateTime
    # --------------------------------------------------------

    value = resource.get(
        "occurrenceDateTime"
    )

    if value:
        return value, "occurrenceDateTime"


    occurrence_period = resource.get(
        "occurrencePeriod"
    )

    if isinstance(occurrence_period, dict):

        start = occurrence_period.get("start")

        if start:
            return start, "occurrencePeriod.start"


    # --------------------------------------------------------
    # Encounter / CarePlan / other period-based resources
    # --------------------------------------------------------

    period = resource.get("period")

    if isinstance(period, dict):

        start = period.get("start")

        if start:
            return start, "period.start"


    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------

    value = resource.get(
        "onsetDateTime"
    )

    if value:
        return value, "onsetDateTime"


    # --------------------------------------------------------
    # Allergy / Condition recorded date
    # --------------------------------------------------------

    value = resource.get(
        "recordedDate"
    )

    if value:
        return value, "recordedDate"


    # --------------------------------------------------------
    # DiagnosticReport
    # --------------------------------------------------------

    value = resource.get(
        "issued"
    )

    if value:
        return value, "issued"


    # --------------------------------------------------------
    # Immunization
    # --------------------------------------------------------

    value = resource.get(
        "occurrenceDateTime"
    )

    if value:
        return value, "occurrenceDateTime"


    return "", ""


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_date(value):

    if not value:
        return ""

    value = str(value)

    try:

        value = value.replace(
            "Z",
            "+00:00"
        )

        dt = datetime.fromisoformat(value)

        return dt.date().isoformat()

    except Exception:

        return value[:10]


# ============================================================
# RESOURCE-SPECIFIC STATUS
# ============================================================

def extract_status(resource):

    status = resource.get(
        "status",
        ""
    )

    if status:
        return str(status)

    return ""


# ============================================================
# MAIN PROCESSING
# ============================================================

rows = []

patient_count = 0

files_processed = 0

events_with_dates = 0

events_without_dates = 0

observations_with_values = 0

medication_events = 0

medications_extracted = 0


# ============================================================
# PROCESS FHIR FILES
# ============================================================

for file_path in FHIR_DIR.glob("*.json"):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            bundle = json.load(f)


        files_processed += 1


        # ----------------------------------------------------
        # Find Patient ID
        # ----------------------------------------------------

        patient_id = None

        for entry in bundle.get(
            "entry",
            []
        ):

            resource = entry.get(
                "resource",
                {}
            )

            if (
                resource.get(
                    "resourceType"
                )
                == "Patient"
            ):

                patient_id = resource.get(
                    "id"
                )

                break


        if not patient_id:

            continue


        patient_count += 1


        # ----------------------------------------------------
        # Process resources
        # ----------------------------------------------------

        for entry in bundle.get(
            "entry",
            []
        ):

            resource = entry.get(
                "resource",
                {}
            )

            resource_type = resource.get(
                "resourceType"
            )


            if resource_type not in TARGET_RESOURCES:

                continue


            # ------------------------------------------------
            # Date
            # ------------------------------------------------

            raw_date, date_source = extract_date(
                resource
            )

            event_date = normalize_date(
                raw_date
            )


            if event_date:

                events_with_dates += 1

            else:

                events_without_dates += 1


            # ------------------------------------------------
            # Clinical code / concept
            # ------------------------------------------------

            code_display = extract_code(
                resource
            )


            # ------------------------------------------------
            # Observation value
            # ------------------------------------------------

            value, unit = extract_value(
                resource
            )


            if (
                resource_type == "Observation"
                and value
            ):

                observations_with_values += 1


            # ------------------------------------------------
            # Medication tracking
            # ------------------------------------------------

            if resource_type in {
                "MedicationRequest",
                "MedicationAdministration",
            }:

                medication_events += 1

                if code_display:

                    medications_extracted += 1


            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            status = extract_status(
                resource
            )


            # ------------------------------------------------
            # Store row
            # ------------------------------------------------

            rows.append({

                "patient_id": patient_id,

                "resource_type": resource_type,

                "event_date": event_date,

                "date_source": date_source,

                "code": code_display,

                "value": value,

                "unit": unit,

                "status": status,

            })


    except Exception as e:

        print(
            f"Error processing "
            f"{file_path.name}: {e}"
        )


# ============================================================
# SORT TIMELINE
# ============================================================

rows.sort(
    key=lambda x: (
        x["patient_id"],
        x["event_date"],
        x["resource_type"],
    )
)


# ============================================================
# WRITE CSV
# ============================================================

FIELDNAMES = [

    "patient_id",

    "resource_type",

    "event_date",

    "date_source",

    "code",

    "value",

    "unit",

    "status",

]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "ELDERDOCAI PATIENT TIMELINE BUILDER"
)

print("=" * 70)

print(
    f"FHIR files processed:       "
    f"{files_processed}"
)

print(
    f"Patients processed:         "
    f"{patient_count}"
)

print(
    f"Timeline events:            "
    f"{len(rows)}"
)

print(
    f"Events with dates:          "
    f"{events_with_dates}"
)

print(
    f"Events without dates:       "
    f"{events_without_dates}"
)

print(
    f"Observations with values:   "
    f"{observations_with_values}"
)

print(
    f"Medication events:          "
    f"{medication_events}"
)

print(
    f"Medications extracted:      "
    f"{medications_extracted}"
)

print(
    f"Output file:                "
    f"{OUTPUT_FILE}"
)

print("=" * 70)

print()

print(
    "Timeline columns:"
)

for field in FIELDNAMES:

    print(
        f"  - {field}"
    )

print()

print(
    "BUILD COMPLETE"
)

print("=" * 70)