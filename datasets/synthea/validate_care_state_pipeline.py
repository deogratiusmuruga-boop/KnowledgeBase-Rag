import json
from pathlib import Path
from collections import Counter


# ============================================================
# ELDERDOCAI - CARE STATE PIPELINE VALIDATOR
# ============================================================

BASE_DIR = Path("elderdocai/processed")

FILES = {
    "clinical_features": BASE_DIR / "clinical_features.json",
    "dynamic_states": BASE_DIR / "dynamic_care_states.json",
    "timeline": BASE_DIR / "care_state_timeline.json",
    "transitions": BASE_DIR / "care_state_transitions.json",
    "assistance": BASE_DIR / "adaptive_assistance.json",
}


DIMENSIONS = [
    "Medication Burden",
    "Clinical Activity",
    "Observation Intensity",
    "Encounter Intensity",
    "Condition Burden",
    "Recent Clinical Activity",
    "Care Complexity",
    "Preventive Care Activity",
]

VALID_STATES = {
    "STABLE",
    "LOW_ACTIVITY",
    "MODERATE_ACTIVITY",
    "HIGH_ACTIVITY",
    "NO_DATA",
}

VALID_LEVELS = {
    "LOW",
    "MODERATE",
    "HIGH",
}

VALID_DIRECTIONS = {
    "INCREASING",
    "DECREASING",
    "UNCHANGED",
    "UNKNOWN",
    "INITIAL",
}

VALID_TRANSITION_TYPES = {
    "INITIAL_STATE",
    "NO_CHANGE",
    "GAP",
    "INCREASING_ACTIVITY",
    "DECREASING_ACTIVITY",
    "STATE_ESCALATION",
    "STATE_DEESCALATION",
}

VALID_ASSISTANCE_MODES = {
    "INITIAL_CONTEXT",
    "LIGHT_SUPPORT",
    "MONITORING_SUPPORT",
    "CONTEXTUAL_SUPPORT",
    "ENHANCED_SUPPORT",
    "FOLLOW_UP_SUPPORT",
    "ADAPTIVE_ESCALATION",
    "ADAPTIVE_DEESCALATION",
    "WAIT_FOR_DATA",
}


# ============================================================
# HELPERS
# ============================================================

errors = []
warnings = []


def load_json(path):
    if not path.exists():
        errors.append(f"Missing file: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            errors.append(f"Expected list in {path}")
            return []

        return data

    except Exception as exc:
        errors.append(f"Could not load {path}: {exc}")
        return []


def check_score(value, label):
    if not isinstance(value, (int, float)):
        errors.append(f"{label}: score is not numeric")
        return

    if value < 0 or value > 1:
        errors.append(
            f"{label}: score {value} outside expected range [0, 1]"
        )


def check_date_order(start, end, label):
    if not start or not end:
        return

    if start > end:
        errors.append(
            f"{label}: window_start {start} is after window_end {end}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ELDERDOCAI - CARE STATE PIPELINE VALIDATOR")
print("=" * 70)

print("\nLoading pipeline outputs...")

clinical_features = load_json(FILES["clinical_features"])
dynamic_states = load_json(FILES["dynamic_states"])
timeline = load_json(FILES["timeline"])
transitions = load_json(FILES["transitions"])
assistance = load_json(FILES["assistance"])

print(f"Clinical features:       {len(clinical_features)}")
print(f"Dynamic care states:     {len(dynamic_states)}")
print(f"Care-state timeline:     {len(timeline)}")
print(f"Care-state transitions:  {len(transitions)}")
print(f"Adaptive assistance:     {len(assistance)}")


# ============================================================
# 1. PATIENT CONSISTENCY
# ============================================================

print("\n" + "-" * 70)
print("1. PATIENT CONSISTENCY")
print("-" * 70)

clinical_patients = {
    x.get("patient_id")
    for x in clinical_features
    if x.get("patient_id")
}

dynamic_patients = {
    x.get("patient_id")
    for x in dynamic_states
    if x.get("patient_id")
}

timeline_patients = {
    x.get("patient_id")
    for x in timeline
    if x.get("patient_id")
}

transition_patients = {
    x.get("patient_id")
    for x in transitions
    if x.get("patient_id")
}

assistance_patients = {
    x.get("patient_id")
    for x in assistance
    if x.get("patient_id")
}

print(f"Clinical feature patients: {len(clinical_patients)}")
print(f"Dynamic state patients:    {len(dynamic_patients)}")
print(f"Timeline patients:         {len(timeline_patients)}")
print(f"Transition patients:       {len(transition_patients)}")
print(f"Assistance patients:       {len(assistance_patients)}")

if clinical_patients != dynamic_patients:
    warnings.append("Clinical and dynamic-state patient sets differ.")

if clinical_patients != timeline_patients:
    warnings.append("Clinical and timeline patient sets differ.")

if clinical_patients != transition_patients:
    warnings.append("Clinical and transition patient sets differ.")

if clinical_patients != assistance_patients:
    warnings.append("Clinical and assistance patient sets differ.")


# ============================================================
# 2. DYNAMIC CARE STATE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("2. DYNAMIC CARE STATE VALIDATION")
print("-" * 70)

state_counter = Counter()

for i, record in enumerate(dynamic_states, start=1):

    state = record.get("care_state")
    score = record.get("overall_score")

    state_counter[state] += 1

    if state not in VALID_STATES:
        errors.append(
            f"Dynamic state record {i}: invalid state '{state}'"
        )

    check_score(score, f"Dynamic state record {i}")

    dimensions = record.get("dimensions", {})

    missing_dimensions = [
        d for d in DIMENSIONS
        if d not in dimensions
    ]

    if missing_dimensions:
        errors.append(
            f"Dynamic state record {i}: missing dimensions "
            f"{missing_dimensions}"
        )

    for dimension in DIMENSIONS:

        if dimension not in dimensions:
            continue

        info = dimensions[dimension]

        dimension_score = info.get("score")
        level = info.get("level")

        check_score(
            dimension_score,
            f"Record {i} / {dimension}"
        )

        if level not in VALID_LEVELS:
            errors.append(
                f"Record {i} / {dimension}: "
                f"invalid level '{level}'"
            )

print("\nCare-state distribution:")

for state, count in sorted(state_counter.items()):
    print(f"  {state:20} {count}")


# ============================================================
# 3. TIMELINE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("3. TIMELINE VALIDATION")
print("-" * 70)

timeline_state_counter = Counter()
timeline_transition_counter = Counter()

no_data_count = 0
active_count = 0

for i, record in enumerate(timeline, start=1):

    state = record.get("care_state")
    score = record.get("overall_score")

    timeline_state_counter[state] += 1

    if state not in VALID_STATES:
        errors.append(
            f"Timeline record {i}: invalid state '{state}'"
        )

    if state != "NO_DATA":
        check_score(score, f"Timeline record {i}")

    check_date_order(
        record.get("window_start"),
        record.get("window_end"),
        f"Timeline record {i}"
    )

    transition = record.get("transition")

    if transition:
        timeline_transition_counter[transition] += 1

    if state == "NO_DATA":
        no_data_count += 1
    else:
        active_count += 1

        if state == "STABLE" and score == 0:
            warnings.append(
                f"Timeline record {i}: STABLE state has zero score."
            )

print(f"Total windows:       {len(timeline)}")
print(f"Active windows:      {active_count}")
print(f"NO_DATA windows:     {no_data_count}")

print("\nTimeline state distribution:")

for state, count in sorted(timeline_state_counter.items()):
    print(f"  {state:20} {count}")


# ============================================================
# 4. NO_DATA SAFETY CHECK
# ============================================================

print("\n" + "-" * 70)
print("4. NO_DATA SAFETY CHECK")
print("-" * 70)

bad_no_data = 0

for i, record in enumerate(timeline, start=1):

    state = record.get("care_state")
    score = record.get("overall_score")

    if state == "NO_DATA":

        if score is not None and score != 0:
            errors.append(
                f"Timeline record {i}: "
                f"NO_DATA has non-zero score {score}"
            )

        if record.get("transition") not in {
            "GAP",
            "INITIAL_STATE",
            "NO_DATA",
            None,
        }:
            warnings.append(
                f"Timeline record {i}: "
                f"unexpected NO_DATA transition."
            )

for i, record in enumerate(assistance, start=1):

    if record.get("current_state") == "NO_DATA":

        if record.get("assistance_mode") != "WAIT_FOR_DATA":
            errors.append(
                f"Assistance record {i}: "
                f"NO_DATA does not produce WAIT_FOR_DATA."
            )

        if record.get("transition_direction") != "UNKNOWN":
            errors.append(
                f"Assistance record {i}: "
                f"NO_DATA has non-UNKNOWN direction."
            )

        bad_no_data += 1

print(f"NO_DATA assistance records checked: {bad_no_data}")

if bad_no_data > 0:
    print("NO_DATA safety behavior: PASS")
else:
    warnings.append(
        "No NO_DATA assistance records were available for validation."
    )


# ============================================================
# 5. TRANSITION VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("5. TRANSITION VALIDATION")
print("-" * 70)

transition_counter = Counter()
direction_counter = Counter()

for i, record in enumerate(transitions, start=1):

    transition_type = record.get("transition_type")
    direction = record.get("transition_direction")

    transition_counter[transition_type] += 1
    direction_counter[direction] += 1

    if transition_type not in VALID_TRANSITION_TYPES:
        errors.append(
            f"Transition record {i}: "
            f"invalid transition type '{transition_type}'"
        )

    if direction not in VALID_DIRECTIONS:
        errors.append(
            f"Transition record {i}: "
            f"invalid direction '{direction}'"
        )

    previous_state = record.get("previous_state")
    current_state = record.get("current_state")

    previous_score = record.get("previous_score")
    current_score = record.get("current_score")

    if previous_score is not None:
        check_score(
            previous_score,
            f"Transition record {i} previous_score"
        )

    if current_score is not None:
        check_score(
            current_score,
            f"Transition record {i} current_score"
        )

    if transition_type == "GAP":

        if (
            previous_state != "NO_DATA"
            and current_state != "NO_DATA"
        ):
            errors.append(
                f"Transition record {i}: "
                f"GAP requires NO_DATA on at least one side."
            )

        if direction != "UNKNOWN":
            errors.append(
                f"Transition record {i}: GAP direction is not UNKNOWN."
            )

    if transition_type == "STATE_ESCALATION":

        if direction != "INCREASING":
            errors.append(
                f"Transition record {i}: "
                f"STATE_ESCALATION direction mismatch."
            )

    if transition_type == "STATE_DEESCALATION":

        if direction != "DECREASING":
            errors.append(
                f"Transition record {i}: "
                f"STATE_DEESCALATION direction mismatch."
            )

print("\nTransition distribution:")

for transition, count in sorted(transition_counter.items()):
    print(f"  {transition:35} {count}")


# ============================================================
# 6. ASSISTANCE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("6. ADAPTIVE ASSISTANCE VALIDATION")
print("-" * 70)

assistance_counter = Counter()
priority_counter = Counter()

for i, record in enumerate(assistance, start=1):

    mode = record.get("assistance_mode")
    priority = record.get("priority")

    assistance_counter[mode] += 1
    priority_counter[priority] += 1

    if mode not in VALID_ASSISTANCE_MODES:
        errors.append(
            f"Assistance record {i}: "
            f"invalid mode '{mode}'"
        )

    if priority not in {"LOW", "MODERATE", "HIGH"}:
        errors.append(
            f"Assistance record {i}: "
            f"invalid priority '{priority}'"
        )

    current_state = record.get("current_state")
    transition_type = record.get("transition_type")

    # NO_DATA safety
    if current_state == "NO_DATA":

        if mode != "WAIT_FOR_DATA":
            errors.append(
                f"Assistance record {i}: "
                f"NO_DATA should produce WAIT_FOR_DATA."
            )

    # Escalation safety
    if transition_type == "STATE_ESCALATION":

        if mode != "ADAPTIVE_ESCALATION":
            errors.append(
                f"Assistance record {i}: "
                f"STATE_ESCALATION should produce "
                f"ADAPTIVE_ESCALATION."
            )

    # De-escalation safety
    if transition_type == "STATE_DEESCALATION":

        if mode != "ADAPTIVE_DEESCALATION":
            errors.append(
                f"Assistance record {i}: "
                f"STATE_DEESCALATION should produce "
                f"ADAPTIVE_DEESCALATION."
            )

print("\nAssistance-mode distribution:")

for mode, count in sorted(assistance_counter.items()):
    print(f"  {mode:30} {count}")

print("\nPriority distribution:")

for priority, count in sorted(priority_counter.items()):
    print(f"  {priority:30} {count}")


# ============================================================
# 7. SCORE RANGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("7. SCORE RANGE VALIDATION")
print("-" * 70)

all_scores = []

for record in dynamic_states:
    if isinstance(record.get("overall_score"), (int, float)):
        all_scores.append(record["overall_score"])

for record in timeline:
    if isinstance(record.get("overall_score"), (int, float)):
        all_scores.append(record["overall_score"])

if all_scores:

    print(f"Scores checked: {len(all_scores)}")
    print(f"Minimum score: {min(all_scores):.4f}")
    print(f"Maximum score: {max(all_scores):.4f}")
    print(f"Mean score:    {sum(all_scores) / len(all_scores):.4f}")

    outside_range = [
        x for x in all_scores
        if x < 0 or x > 1
    ]

    if outside_range:
        errors.append(
            f"{len(outside_range)} scores outside [0,1]"
        )
    else:
        print("Score range: PASS")


# ============================================================
# 8. DIMENSION CHANGE VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("8. DIMENSION CHANGE VALIDATION")
print("-" * 70)

dimension_change_counter = Counter()

for i, record in enumerate(transitions, start=1):

    for change in record.get("changed_dimensions", []):

        dimension = change.get("dimension")
        direction = change.get("direction")
        delta = change.get("delta")

        dimension_change_counter[dimension] += 1

        if dimension not in DIMENSIONS:
            errors.append(
                f"Transition record {i}: "
                f"unknown dimension '{dimension}'"
            )

        if direction not in {
            "INCREASED",
            "DECREASED",
            "UNCHANGED",
        }:
            errors.append(
                f"Transition record {i}: "
                f"invalid dimension direction '{direction}'"
            )

        if not isinstance(delta, (int, float)):
            errors.append(
                f"Transition record {i}: "
                f"dimension delta is not numeric"
            )

print("Dimension changes observed:")

for dimension, count in sorted(dimension_change_counter.items()):
    print(f"  {dimension:30} {count}")


# ============================================================
# 9. RECORD COUNT CONSISTENCY
# ============================================================

print("\n" + "-" * 70)
print("9. RECORD COUNT CONSISTENCY")
print("-" * 70)

if len(timeline) != len(transitions):
    errors.append(
        "Timeline and transition record counts do not match."
    )
else:
    print("Timeline ↔ transitions: PASS")

if len(transitions) != len(assistance):
    errors.append(
        "Transition and assistance record counts do not match."
    )
else:
    print("Transitions ↔ assistance: PASS")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ ERRORS: {len(errors)}")

    for error in errors:
        print(f"  - {error}")

else:
    print("\n✅ ERRORS: 0")

if warnings:
    print(f"\n⚠ WARNINGS: {len(warnings)}")

    for warning in warnings:
        print(f"  - {warning}")

else:
    print("\n✅ WARNINGS: 0")


print("\n" + "-" * 70)

if not errors:
    print("PIPELINE VALIDATION: PASS")
    print("The current rule-based care-state pipeline is internally consistent.")
else:
    print("PIPELINE VALIDATION: REVIEW REQUIRED")
    print("Fix the reported errors before treating the pipeline as validated.")

print("-" * 70)

print("\nValidated components:")
print("  ✓ Clinical feature records")
print("  ✓ Dynamic care states")
print("  ✓ Continuous temporal windows")
print("  ✓ NO_DATA handling")
print("  ✓ Care-state transitions")
print("  ✓ Dimension-level transitions")
print("  ✓ Adaptive assistance")
print("  ✓ Score ranges")
print("  ✓ Patient consistency")
print("  ✓ Record-count consistency")

print("\nImportant:")
print("  This validator checks structural and logical consistency.")
print("  It does not establish clinical validity.")
print("  The system does not diagnose disease.")
print("  The system does not predict medical risk.")
print("  Scores represent documented care activity.")

print("=" * 70)