"""
ELDERDOCAI - CARE STATE TIMELINE BUILDER
Version 2 - Continuous Temporal Windows

Purpose:
    Build a temporal care-state representation from Synthea FHIR bundles.

Input:
    elderdocai/fhir/*.json

Outputs:
    elderdocai/processed/care_state_timeline.json
    elderdocai/processed/care_state_timeline.csv

Design:
    - Rule-based
    - Interpretable
    - Continuous yearly windows
    - Explicit NO_DATA state
    - Population-relative normalization
    - No diagnosis
    - No medical risk prediction
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ======================================================================
# CONFIGURATION
# ======================================================================

FHIR_DIR = Path("elderdocai/fhir")
OUTPUT_DIR = Path("elderdocai/processed")

JSON_OUTPUT = OUTPUT_DIR / "care_state_timeline.json"
CSV_OUTPUT = OUTPUT_DIR / "care_state_timeline.csv"

WINDOW_MONTHS = 12

SUPPORTED_RESOURCES = {
    "Condition",
    "MedicationRequest",
    "MedicationAdministration",
    "Observation",
    "Encounter",
    "Procedure",
    "DiagnosticReport",
    "Immunization",
    "CarePlan",
}


# ======================================================================
# HELPERS
# ======================================================================

def parse_datetime(value):
    """Parse a FHIR date/dateTime value."""

    if not value:
        return None

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(value)

    except ValueError:
        pass

    try:
        return datetime.strptime(
            value[:10],
            "%Y-%m-%d"
        )

    except ValueError:
        return None


def percentile_rank(value, values):
    """
    Population-relative percentile rank.

    Returns:
        0.0 - 1.0
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return 0.0

    less_or_equal = sum(
        v <= value
        for v in sorted_values
    )

    return (less_or_equal - 1) / (
        len(sorted_values) - 1
    )


def level_from_score(score):

    if score < 0.34:
        return "LOW"

    if score < 0.67:
        return "MODERATE"

    return "HIGH"


def state_from_score(score):

    if score < 0.25:
        return "STABLE"

    if score < 0.50:
        return "LOW_ACTIVITY"

    if score < 0.75:
        return "MODERATE_ACTIVITY"

    return "HIGH_ACTIVITY"


# ======================================================================
# FHIR DATE EXTRACTION
# ======================================================================

def extract_event_date(resource):

    resource_type = resource.get(
        "resourceType"
    )

    # --------------------------------------------------------------
    # Condition
    # --------------------------------------------------------------

    if resource_type == "Condition":

        onset = resource.get(
            "onsetDateTime"
        )

        if onset:
            return parse_datetime(onset)

        recorded = resource.get(
            "recordedDate"
        )

        if recorded:
            return parse_datetime(recorded)

        return None

    # --------------------------------------------------------------
    # MedicationRequest
    # --------------------------------------------------------------

    if resource_type == "MedicationRequest":

        return parse_datetime(
            resource.get("authoredOn")
        )

    # --------------------------------------------------------------
    # MedicationAdministration
    # --------------------------------------------------------------

    if resource_type == "MedicationAdministration":

        return parse_datetime(
            resource.get(
                "effectiveDateTime"
            )
        )

    # --------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------

    if resource_type == "Observation":

        return parse_datetime(
            resource.get(
                "effectiveDateTime"
            )
        )

    # --------------------------------------------------------------
    # Encounter
    # --------------------------------------------------------------

    if resource_type == "Encounter":

        period = resource.get(
            "period",
            {}
        )

        return parse_datetime(
            period.get("start")
        )

    # --------------------------------------------------------------
    # Procedure
    # --------------------------------------------------------------

    if resource_type == "Procedure":

        performed_period = resource.get(
            "performedPeriod"
        )

        if isinstance(
            performed_period,
            dict
        ):

            start = performed_period.get(
                "start"
            )

            if start:
                return parse_datetime(start)

        performed_datetime = resource.get(
            "performedDateTime"
        )

        if performed_datetime:
            return parse_datetime(
                performed_datetime
            )

        return None

    # --------------------------------------------------------------
    # DiagnosticReport
    # --------------------------------------------------------------

    if resource_type == "DiagnosticReport":

        return parse_datetime(
            resource.get(
                "effectiveDateTime"
            )
        )

    # --------------------------------------------------------------
    # Immunization
    # --------------------------------------------------------------

    if resource_type == "Immunization":

        return parse_datetime(
            resource.get(
                "occurrenceDateTime"
            )
        )

    # --------------------------------------------------------------
    # CarePlan
    # --------------------------------------------------------------

    if resource_type == "CarePlan":

        period = resource.get(
            "period",
            {}
        )

        return parse_datetime(
            period.get("start")
        )

    return None


# ======================================================================
# EVENT EXTRACTION
# ======================================================================

def extract_events_from_bundle(bundle):

    events = []

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

        if resource_type not in SUPPORTED_RESOURCES:
            continue

        event_date = extract_event_date(
            resource
        )

        if event_date is None:
            continue

        events.append(
            {
                "date": event_date,
                "resource_type": resource_type,
            }
        )

    events.sort(
        key=lambda x: x["date"]
    )

    return events


# ======================================================================
# LOAD PATIENTS
# ======================================================================

def load_patient_events():

    patient_events = {}

    files = sorted(
        FHIR_DIR.glob("*.json")
    )

    if not files:

        raise FileNotFoundError(
            f"No FHIR JSON files found in {FHIR_DIR}"
        )

    print(
        f"Found {len(files)} FHIR patient file(s)."
    )

    for index, path in enumerate(
        files,
        start=1
    ):

        try:

            with path.open(
                "r",
                encoding="utf-8"
            ) as f:

                bundle = json.load(f)

            patient_id = None

            for entry in bundle.get(
                "entry",
                []
            ):

                resource = entry.get(
                    "resource",
                    {}
                )

                if resource.get(
                    "resourceType"
                ) == "Patient":

                    patient_id = resource.get(
                        "id"
                    )

                    break

            if not patient_id:

                print(
                    f"  WARNING: No patient ID "
                    f"found in {path.name}"
                )

                continue

            events = extract_events_from_bundle(
                bundle
            )

            patient_events[
                patient_id
            ] = events

            if index <= 5:

                print(
                    f"  Patient {index}: "
                    f"{patient_id} "
                    f"({len(events)} dated events)"
                )

        except Exception as exc:

            print(
                f"  ERROR processing "
                f"{path.name}: {exc}"
            )

    return patient_events


# ======================================================================
# CONTINUOUS YEAR WINDOWS
# ======================================================================

def create_continuous_windows(events):

    if not events:
        return []

    years = [
        event["date"].year
        for event in events
    ]

    first_year = min(years)
    last_year = max(years)

    windows = []

    events_by_year = defaultdict(list)

    for event in events:

        events_by_year[
            event["date"].year
        ].append(event)

    for year in range(
        first_year,
        last_year + 1
    ):

        year_events = events_by_year.get(
            year,
            []
        )

        windows.append(
            {
                "year": year,

                "window_start":
                    f"{year}-01-01",

                "window_end":
                    f"{year}-12-31",

                "events":
                    year_events,

                "has_data":
                    len(year_events) > 0,
            }
        )

    return windows


# ======================================================================
# WINDOW FEATURES
# ======================================================================

def calculate_window_features(events):

    counts = Counter(
        event["resource_type"]
        for event in events
    )

    medication_events = (
        counts["MedicationRequest"]
        +
        counts["MedicationAdministration"]
    )

    return {
        "total_event_count":
            len(events),

        "condition_event_count":
            counts["Condition"],

        "medication_event_count":
            medication_events,

        "medication_request_count":
            counts["MedicationRequest"],

        "medication_administration_count":
            counts[
                "MedicationAdministration"
            ],

        "observation_event_count":
            counts["Observation"],

        "encounter_event_count":
            counts["Encounter"],

        "procedure_event_count":
            counts["Procedure"],

        "diagnostic_report_event_count":
            counts["DiagnosticReport"],

        "immunization_event_count":
            counts["Immunization"],

        "care_plan_event_count":
            counts["CarePlan"],
    }


# ======================================================================
# POPULATION DISTRIBUTIONS
# ======================================================================

def build_population_distributions(
    active_windows
):

    feature_names = [
        "total_event_count",
        "condition_event_count",
        "medication_event_count",
        "medication_request_count",
        "medication_administration_count",
        "observation_event_count",
        "encounter_event_count",
        "procedure_event_count",
        "diagnostic_report_event_count",
        "immunization_event_count",
        "care_plan_event_count",
    ]

    distributions = {}

    for feature in feature_names:

        distributions[feature] = [
            window["features"].get(
                feature,
                0
            )
            for window in active_windows
        ]

    return distributions


# ======================================================================
# DIMENSIONS
# ======================================================================

def build_dimensions(
    features,
    distributions
):

    def pct(feature):

        return percentile_rank(
            features.get(feature, 0),
            distributions.get(
                feature,
                []
            )
        )

    medication_score = (
        0.45 * pct(
            "medication_event_count"
        )
        +
        0.35 * pct(
            "medication_request_count"
        )
        +
        0.20 * pct(
            "medication_administration_count"
        )
    )

    clinical_score = (
        0.30 * pct(
            "total_event_count"
        )
        +
        0.25 * pct(
            "encounter_event_count"
        )
        +
        0.25 * pct(
            "procedure_event_count"
        )
        +
        0.20 * pct(
            "diagnostic_report_event_count"
        )
    )

    observation_score = pct(
        "observation_event_count"
    )

    encounter_score = pct(
        "encounter_event_count"
    )

    condition_score = pct(
        "condition_event_count"
    )

    recent_score = (
        0.40 * pct(
            "total_event_count"
        )
        +
        0.25 * pct(
            "encounter_event_count"
        )
        +
        0.20 * pct(
            "observation_event_count"
        )
        +
        0.15 * pct(
            "medication_event_count"
        )
    )

    complexity_score = (
        0.20 * pct(
            "condition_event_count"
        )
        +
        0.20 * pct(
            "medication_event_count"
        )
        +
        0.15 * pct(
            "encounter_event_count"
        )
        +
        0.15 * pct(
            "observation_event_count"
        )
        +
        0.15 * pct(
            "procedure_event_count"
        )
        +
        0.15 * pct(
            "diagnostic_report_event_count"
        )
    )

    preventive_score = (
        0.60 * pct(
            "immunization_event_count"
        )
        +
        0.40 * pct(
            "care_plan_event_count"
        )
    )

    dimensions = {

        "Medication Burden": {
            "score":
                round(
                    medication_score,
                    4
                ),

            "level":
                level_from_score(
                    medication_score
                ),

            "raw_features": {
                "medication_event_count":
                    features[
                        "medication_event_count"
                    ],

                "medication_request_count":
                    features[
                        "medication_request_count"
                    ],

                "medication_administration_count":
                    features[
                        "medication_administration_count"
                    ],
            },

            "explanation":
                "Medication burden reflects "
                "documented medication requests "
                "and administrations within the "
                "time window."
        },

        "Clinical Activity": {
            "score":
                round(
                    clinical_score,
                    4
                ),

            "level":
                level_from_score(
                    clinical_score
                ),

            "raw_features": {
                "total_event_count":
                    features[
                        "total_event_count"
                    ],

                "encounter_event_count":
                    features[
                        "encounter_event_count"
                    ],

                "procedure_event_count":
                    features[
                        "procedure_event_count"
                    ],

                "diagnostic_report_event_count":
                    features[
                        "diagnostic_report_event_count"
                    ],
            },

            "explanation":
                "Clinical activity reflects "
                "documented clinical events, "
                "encounters, procedures, and "
                "diagnostic reports."
        },

        "Observation Intensity": {
            "score":
                round(
                    observation_score,
                    4
                ),

            "level":
                level_from_score(
                    observation_score
                ),

            "raw_features": {
                "observation_event_count":
                    features[
                        "observation_event_count"
                    ],
            },

            "explanation":
                "Observation intensity reflects "
                "the number of documented "
                "observations in the time window."
        },

        "Encounter Intensity": {
            "score":
                round(
                    encounter_score,
                    4
                ),

            "level":
                level_from_score(
                    encounter_score
                ),

            "raw_features": {
                "encounter_event_count":
                    features[
                        "encounter_event_count"
                    ],
            },

            "explanation":
                "Encounter intensity reflects "
                "documented healthcare encounters "
                "during the time window."
        },

        "Condition Burden": {
            "score":
                round(
                    condition_score,
                    4
                ),

            "level":
                level_from_score(
                    condition_score
                ),

            "raw_features": {
                "condition_event_count":
                    features[
                        "condition_event_count"
                    ],
            },

            "explanation":
                "Condition burden reflects "
                "documented condition events. "
                "It does not indicate disease "
                "severity."
        },

        "Recent Clinical Activity": {
            "score":
                round(
                    recent_score,
                    4
                ),

            "level":
                level_from_score(
                    recent_score
                ),

            "raw_features": {
                "total_event_count":
                    features[
                        "total_event_count"
                    ],

                "medication_event_count":
                    features[
                        "medication_event_count"
                    ],

                "observation_event_count":
                    features[
                        "observation_event_count"
                    ],

                "encounter_event_count":
                    features[
                        "encounter_event_count"
                    ],
            },

            "explanation":
                "This dimension represents "
                "documented activity occurring "
                "within the current yearly "
                "temporal window."
        },

        "Care Complexity": {
            "score":
                round(
                    complexity_score,
                    4
                ),

            "level":
                level_from_score(
                    complexity_score
                ),

            "raw_features": {
                "condition_event_count":
                    features[
                        "condition_event_count"
                    ],

                "medication_event_count":
                    features[
                        "medication_event_count"
                    ],

                "encounter_event_count":
                    features[
                        "encounter_event_count"
                    ],

                "observation_event_count":
                    features[
                        "observation_event_count"
                    ],

                "procedure_event_count":
                    features[
                        "procedure_event_count"
                    ],

                "diagnostic_report_event_count":
                    features[
                        "diagnostic_report_event_count"
                    ],
            },

            "explanation":
                "Care complexity combines several "
                "categories of documented care "
                "activity."
        },

        "Preventive Care Activity": {
            "score":
                round(
                    preventive_score,
                    4
                ),

            "level":
                level_from_score(
                    preventive_score
                ),

            "raw_features": {
                "immunization_event_count":
                    features[
                        "immunization_event_count"
                    ],

                "care_plan_event_count":
                    features[
                        "care_plan_event_count"
                    ],
            },

            "explanation":
                "Preventive care activity reflects "
                "documented immunization and "
                "care-plan activity."
        },
    }

    return dimensions


# ======================================================================
# OVERALL SCORE
# ======================================================================

def calculate_overall_score(
    dimensions
):

    weights = {

        "Medication Burden":
            0.12,

        "Clinical Activity":
            0.18,

        "Observation Intensity":
            0.12,

        "Encounter Intensity":
            0.12,

        "Condition Burden":
            0.10,

        "Recent Clinical Activity":
            0.18,

        "Care Complexity":
            0.13,

        "Preventive Care Activity":
            0.05,
    }

    score = sum(
        dimensions[name]["score"]
        * weight

        for name, weight
        in weights.items()
    )

    return round(
        score,
        4
    )


# ======================================================================
# BUILD TEMPORAL TIMELINE
# ======================================================================

def build_timeline(
    patient_events
):

    # --------------------------------------------------------------
    # First create all continuous windows.
    # --------------------------------------------------------------

    patient_windows = {}

    active_windows = []

    total_no_data_windows = 0

    for patient_id, events in patient_events.items():

        windows = create_continuous_windows(
            events
        )

        patient_windows[
            patient_id
        ] = windows

        for window in windows:

            if window["has_data"]:

                window["features"] = (
                    calculate_window_features(
                        window["events"]
                    )
                )

                active_windows.append(
                    window
                )

            else:

                window["features"] = (
                    {
                        "total_event_count": 0,
                        "condition_event_count": 0,
                        "medication_event_count": 0,
                        "medication_request_count": 0,
                        "medication_administration_count": 0,
                        "observation_event_count": 0,
                        "encounter_event_count": 0,
                        "procedure_event_count": 0,
                        "diagnostic_report_event_count": 0,
                        "immunization_event_count": 0,
                        "care_plan_event_count": 0,
                    }
                )

                total_no_data_windows += 1

    print(
        f"Total continuous windows: "
        f"{sum(len(x) for x in patient_windows.values())}"
    )

    print(
        f"Active windows: "
        f"{len(active_windows)}"
    )

    print(
        f"NO_DATA windows: "
        f"{total_no_data_windows}"
    )

    # --------------------------------------------------------------
    # Population distributions use ONLY active windows.
    #
    # NO_DATA must never influence population normalization.
    # --------------------------------------------------------------

    print(
        "Building population-relative "
        "feature distributions..."
    )

    distributions = (
        build_population_distributions(
            active_windows
        )
    )

    # --------------------------------------------------------------
    # Generate states.
    # --------------------------------------------------------------

    results = []

    state_counts = Counter()

    transition_counts = Counter()

    for patient_id, windows in patient_windows.items():

        previous_observed_year = None
        previous_state = None

        for window in windows:

            year = window["year"]

            # ======================================================
            # NO DATA WINDOW
            # ======================================================

            if not window["has_data"]:

                record = {

                    "patient_id":
                        patient_id,

                    "window_start":
                        window[
                            "window_start"
                        ],

                    "window_end":
                        window[
                            "window_end"
                        ],

                    "year":
                        year,

                    "care_state":
                        "NO_DATA",

                    "overall_score":
                        None,

                    "has_documented_activity":
                        False,

                    "event_summary":
                        window[
                            "features"
                        ],

                    "dimensions":
                        {},

                    "transition":
                        "NO_DATA",

                    "transition_direction":
                        "NO_DATA",

                    "interpretation":
                        "No dated clinical activity "
                        "was available for this temporal "
                        "window. This must not be interpreted "
                        "as clinical stability or absence "
                        "of healthcare needs."
                }

                results.append(
                    record
                )

                state_counts[
                    "NO_DATA"
                ] += 1

                continue

            # ======================================================
            # ACTIVE WINDOW
            # ======================================================

            features = window[
                "features"
            ]

            dimensions = build_dimensions(
                features,
                distributions
            )

            overall_score = (
                calculate_overall_score(
                    dimensions
                )
            )

            care_state = (
                state_from_score(
                    overall_score
                )
            )

            # ------------------------------------------------------
            # Determine transition.
            # ------------------------------------------------------

            if previous_state is None:

                transition = (
                    "INITIAL_STATE"
                )

                transition_direction = (
                    "INITIAL"
                )

            elif (
                previous_observed_year
                is not None
                and year
                == previous_observed_year + 1
            ):

                if (
                    previous_state
                    == care_state
                ):

                    transition = (
                        "NO_CHANGE"
                    )

                    transition_direction = (
                        "STABLE"
                    )

                else:

                    state_order = {
                        "STABLE": 0,
                        "LOW_ACTIVITY": 1,
                        "MODERATE_ACTIVITY": 2,
                        "HIGH_ACTIVITY": 3,
                    }

                    previous_level = (
                        state_order[
                            previous_state
                        ]
                    )

                    current_level = (
                        state_order[
                            care_state
                        ]
                    )

                    if (
                        current_level
                        > previous_level
                    ):

                        transition_direction = (
                            "INCREASING_ACTIVITY"
                        )

                    else:

                        transition_direction = (
                            "DECREASING_ACTIVITY"
                        )

                    transition = (
                        f"{previous_state}"
                        f"_TO_"
                        f"{care_state}"
                    )

            else:

                transition = "GAP"

                transition_direction = (
                    "UNKNOWN_DUE_TO_GAP"
                )

            record = {

                "patient_id":
                    patient_id,

                "window_start":
                    window[
                        "window_start"
                    ],

                "window_end":
                    window[
                        "window_end"
                    ],

                "year":
                    year,

                "care_state":
                    care_state,

                "overall_score":
                    overall_score,

                "has_documented_activity":
                    True,

                "event_summary":
                    features,

                "dimensions":
                    dimensions,

                "transition":
                    transition,

                "transition_direction":
                    transition_direction,

                "interpretation":
                    "Rule-based temporal representation "
                    "of documented clinical activity and "
                    "care-management activity. This output "
                    "does not diagnose disease and does not "
                    "predict medical risk."
            }

            results.append(
                record
            )

            state_counts[
                care_state
            ] += 1

            transition_counts[
                transition
            ] += 1

            previous_state = care_state

            previous_observed_year = year

    return (
        results,
        state_counts,
        transition_counts
    )


# ======================================================================
# JSON
# ======================================================================

def save_json(results):

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


# ======================================================================
# CSV
# ======================================================================

def save_csv(results):

    rows = []

    for record in results:

        row = {

            "patient_id":
                record["patient_id"],

            "year":
                record["year"],

            "window_start":
                record["window_start"],

            "window_end":
                record["window_end"],

            "care_state":
                record["care_state"],

            "overall_score":
                record["overall_score"],

            "has_documented_activity":
                record[
                    "has_documented_activity"
                ],

            "transition":
                record["transition"],

            "transition_direction":
                record[
                    "transition_direction"
                ],
        }

        for name, dimension in (
            record["dimensions"].items()
        ):

            prefix = (
                name
                .lower()
                .replace(" ", "_")
            )

            row[
                f"{prefix}_score"
            ] = dimension[
                "score"
            ]

            row[
                f"{prefix}_level"
            ] = dimension[
                "level"
            ]

        for feature, value in (
            record[
                "event_summary"
            ].items()
        ):

            row[feature] = value

        rows.append(row)

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with CSV_OUTPUT.open(
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


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print(
        "ELDERDOCAI - CARE STATE TIMELINE BUILDER"
    )
    print(
        "CONTINUOUS TEMPORAL WINDOW VERSION"
    )
    print("=" * 70)

    print()
    print(
        "Loading FHIR patient bundles..."
    )

    patient_events = (
        load_patient_events()
    )

    print()
    print(
        f"Loaded {len(patient_events)} "
        f"patient timeline(s)."
    )

    print()
    print(
        "Building continuous temporal "
        "care states..."
    )

    (
        results,
        state_counts,
        transition_counts
    ) = build_timeline(
        patient_events
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Saving JSON..."
    )

    save_json(results)

    print(
        "Saving CSV..."
    )

    save_csv(results)

    print()
    print("=" * 70)
    print(
        "ELDERDOCAI - CARE STATE TIMELINE BUILDER"
    )
    print("=" * 70)

    print()
    print(
        f"Patients processed: "
        f"{len(patient_events)}"
    )

    print(
        f"Temporal windows:   "
        f"{len(results)}"
    )

    print()
    print(
        "Care-state distribution:"
    )

    ordered_states = [
        "NO_DATA",
        "STABLE",
        "LOW_ACTIVITY",
        "MODERATE_ACTIVITY",
        "HIGH_ACTIVITY",
    ]

    for state in ordered_states:

        print(
            f"  {state:<20}"
            f"{state_counts.get(state, 0)}"
        )

    print()
    print(
        "Transition distribution:"
    )

    for transition, count in sorted(
        transition_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        print(
            f"  {transition:<40}"
            f"{count}"
        )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  JSON: "
        f"{JSON_OUTPUT.resolve()}"
    )

    print(
        f"  CSV:  "
        f"{CSV_OUTPUT.resolve()}"
    )

    print()
    print(
        "Temporal design:"
    )

    print(
        "  Window size: 12 months"
    )

    print(
        "  Window type: continuous calendar years"
    )

    print(
        "  Missing years: NO_DATA"
    )

    print(
        "  Normalization: population-relative"
    )

    print(
        "  Transitions across gaps: blocked"
    )

    print()
    print(
        "Important:"
    )

    print(
        "  NO_DATA does not mean STABLE."
    )

    print(
        "  Missing clinical activity is not "
        "interpreted as absence of care needs."
    )

    print(
        "  This system does not diagnose disease."
    )

    print(
        "  This system does not predict medical risk."
    )

    print(
        "  Scores represent documented care "
        "activity only."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()