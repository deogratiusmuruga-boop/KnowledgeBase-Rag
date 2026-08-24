"""
ElderDocAI - Adaptive Context Builder

Combines the five validated pipeline outputs into a single compact
per-window adaptive-context dataset consumed by the later
context-aware RAG layer.

Pipeline position:
    FHIR -> clinical features -> dynamic care states
    -> continuous temporal timeline -> care-state transitions
    -> adaptive assistance -> ADAPTIVE CONTEXT -> context-aware RAG

Design:
    - One adaptive-context record per temporal window (9,723).
    - Join key: (patient_id, window_start, window_end).
    - NO_DATA windows keep overall_score = null (never 0.0).
    - context_status: ACTIVE / INITIAL / DATA_GAP.
    - Transitions and assistance are REUSED, never recomputed.

Important:
    This output describes documented care activity and temporal
    changes. It is not a diagnosis and does not predict medical risk.
"""

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "elderdocai" / "processed"

FILES = {
    "clinical_features": PROCESSED / "clinical_features.json",
    "dynamic_care_states": PROCESSED / "dynamic_care_states.json",
    "timeline": PROCESSED / "care_state_timeline.json",
    "transitions": PROCESSED / "care_state_transitions.json",
    "assistance": PROCESSED / "adaptive_assistance.json",
}

OUTPUT_JSON = PROCESSED / "adaptive_context.json"
OUTPUT_CSV = PROCESSED / "adaptive_context.csv"

# Compact patient-level aggregate fields carried from clinical_features.json.
PROFILE_FIELDS = [
    "first_event_date",
    "last_event_date",
    "timeline_days",
    "events_per_year",
    "total_event_count",
    "dated_event_count",
    "undated_event_count",
    "recent_event_count_365d",
    "recent_medication_event_count_365d",
    "recent_observation_event_count_365d",
    "recent_condition_event_count_365d",
    "recent_encounter_count_365d",
    "unique_condition_count",
    "condition_count_profile",
    "unique_medication_count",
    "medication_profile_count",
    "unique_observation_count",
    "observation_count_profile",
    "encounter_event_count",
    "procedure_event_count",
    "diagnostic_report_event_count",
    "immunization_event_count",
    "care_plan_event_count",
    "numeric_observation_mean",
    "numeric_observation_median",
]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _compact_dimensions(dimensions):
    """Reduce per-window/per-patient dimensions to name/level/score only."""
    if not isinstance(dimensions, dict):
        return []
    compact = []
    for name in sorted(dimensions.keys()):
        entry = dimensions.get(name) or {}
        compact.append({
            "name": entry.get("name", name),
            "level": entry.get("level"),
            "score": entry.get("score"),
        })
    return compact


def _compact_changed_dimensions(changes):
    """Preserve dimension-level transition deltas without extra fields."""
    compact = []
    for change in changes or []:
        compact.append({
            "dimension": change.get("dimension"),
            "direction": change.get("direction"),
            "delta": change.get("delta"),
            "previous_score": change.get("previous_score"),
            "current_score": change.get("current_score"),
        })
    return compact


def _context_status(state, transition_type):
    """Derive a compact status from the validated state/transition data."""
    if state == "NO_DATA":
        return "DATA_GAP"
    if transition_type == "INITIAL_STATE":
        return "INITIAL"
    return "ACTIVE"


def _build_record(rec, clinical, dynamic, transitions_by_key, assistance_by_key):
    patient_id = rec["patient_id"]
    window_start = rec["window_start"]
    window_end = rec["window_end"]
    key = (patient_id, window_start, window_end)

    state = rec.get("care_state")
    dyn = dynamic.get(patient_id)
    clin = clinical.get(patient_id)
    trans = transitions_by_key.get(key)
    assist = assistance_by_key.get(key)

    record = {
        "patient_id": patient_id,
        "window_start": window_start,
        "window_end": window_end,
        "year": rec.get("year"),
        "context_status": _context_status(
            state,
            (trans or {}).get("transition_type"),
        ),
        "patient_profile": {
            field: clin.get(field)
            for field in PROFILE_FIELDS
        } if clin else {},
        "patient_care_state": {
            "state": dyn.get("care_state"),
            "overall_score": dyn.get("overall_score"),
            "dimensions": _compact_dimensions(dyn.get("dimensions")),
        } if dyn else {},
        "care_state": {
            "state": state,
            "overall_score": rec.get("overall_score"),
            "has_documented_activity": rec.get("has_documented_activity"),
            "event_summary": rec.get("event_summary"),
            "dimensions": _compact_dimensions(rec.get("dimensions")),
            "interpretation": rec.get("interpretation"),
        },
        # Top-level dimension-level deltas (compact form) as described in the
        # adaptive-context specification. Mirrors transition.changed_dimensions
        # so the RAG layer can consume it without descending into transition.
        "changed_dimensions": _compact_changed_dimensions(
            (trans or {}).get("changed_dimensions")
        ) if trans else [],
        "transition": {
            "type": trans.get("transition_type"),
            "direction": trans.get("transition_direction"),
            "magnitude": trans.get("transition_magnitude"),
            "score_delta": trans.get("score_delta"),
            "previous_state": trans.get("previous_state"),
            "current_state": trans.get("current_state"),
            "previous_score": trans.get("previous_score"),
            "current_score": trans.get("current_score"),
            "changed_dimensions": _compact_changed_dimensions(
                trans.get("changed_dimensions")
            ),
            "supporting_evidence": trans.get("supporting_evidence"),
        } if trans else {},
        "adaptive_assistance": {
            "mode": assist.get("assistance_mode"),
            "priority": assist.get("priority"),
            "reason_codes": assist.get("reason_codes") or [],
            "reasons": assist.get("reasons") or [],
            "interpretation": assist.get("interpretation"),
        } if assist else {},
    }
    return record, trans is None, assist is None


def build():
    print("=" * 70)
    print("ELDERDOCAI - ADAPTIVE CONTEXT BUILDER")
    print("=" * 70)
    print("\nLoading validated pipeline outputs...")

    data = {
        name: _load_json(path)
        for name, path in FILES.items()
    }

    clinical = {r["patient_id"]: r for r in data["clinical_features"]}
    dynamic = {r["patient_id"]: r for r in data["dynamic_care_states"]}
    timeline = data["timeline"]
    transitions_by_key = {
        (r["patient_id"], r["window_start"], r["window_end"]): r
        for r in data["transitions"]
    }
    assistance_by_key = {
        (r["patient_id"], r["window_start"], r["window_end"]): r
        for r in data["assistance"]
    }

    records = []
    missing_transition = 0
    missing_assistance = 0

    for rec in timeline:
        record, no_trans, no_assist = _build_record(
            rec,
            clinical,
            dynamic,
            transitions_by_key,
            assistance_by_key,
        )
        records.append(record)
        missing_transition += int(no_trans)
        missing_assistance += int(no_assist)

    if missing_transition or missing_assistance:
        print(
            f"\n⚠ Join warnings: "
            f"{missing_transition} missing transition / "
            f"{missing_assistance} missing assistance record(s)."
        )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # CSV (flat, one row per window)
    # --------------------------------------------------------
    csv_fields = [
        "patient_id",
        "window_start",
        "window_end",
        "year",
        "context_status",
        "care_state",
        "care_state_score",
        "transition_type",
        "transition_direction",
        "transition_magnitude",
        "score_delta",
        "assistance_mode",
        "priority",
        "total_event_count",
        "unique_condition_count",
        "unique_medication_count",
        "unique_observation_count",
        "events_per_year",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            profile = record.get("patient_profile") or {}
            writer.writerow({
                "patient_id": record["patient_id"],
                "window_start": record["window_start"],
                "window_end": record["window_end"],
                "year": record.get("year"),
                "context_status": record["context_status"],
                "care_state": (record.get("care_state") or {}).get("state"),
                "care_state_score": (record.get("care_state") or {}).get("overall_score"),
                "transition_type": (record.get("transition") or {}).get("type"),
                "transition_direction": (record.get("transition") or {}).get("direction"),
                "transition_magnitude": (record.get("transition") or {}).get("magnitude"),
                "score_delta": (record.get("transition") or {}).get("score_delta"),
                "assistance_mode": (record.get("adaptive_assistance") or {}).get("mode"),
                "priority": (record.get("adaptive_assistance") or {}).get("priority"),
                "total_event_count": profile.get("total_event_count"),
                "unique_condition_count": profile.get("unique_condition_count"),
                "unique_medication_count": profile.get("unique_medication_count"),
                "unique_observation_count": profile.get("unique_observation_count"),
                "events_per_year": profile.get("events_per_year"),
            })

    from collections import Counter
    status_counts = Counter(r["context_status"] for r in records)
    patients = len(set(r["patient_id"] for r in records))

    print("\nSaving JSON...")
    print(f"  {OUTPUT_JSON}")
    print("Saving CSV...")
    print(f"  {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("ADAPTIVE CONTEXT BUILDER - SUMMARY")
    print("=" * 70)
    print(f"Patients processed:        {patients}")
    print(f"Adaptive-context records:  {len(records)}")
    print()
    print("context_status distribution:")
    for status in ("ACTIVE", "INITIAL", "DATA_GAP"):
        print(f"  {status:<12} {status_counts.get(status, 0)}")
    print()

    print("Important:")
    print("  The adaptive context describes documented care activity and temporal")
    print("  changes. It does not diagnose disease and does not predict medical risk.")
    print("=" * 70)


if __name__ == "__main__":
    build()