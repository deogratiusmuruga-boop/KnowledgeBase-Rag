"""
ElderDocAI - Adaptive Context Validator

Validates elderdocai/processed/adaptive_context.json and .csv against
the five validated pipeline outputs.

Checks are structural / logical only. This validator does not
establish clinical validity.
"""

import csv
import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "elderdocai" / "processed"

CTX_JSON = PROCESSED / "adaptive_context.json"
CTX_CSV = PROCESSED / "adaptive_context.csv"

SOURCE_FILES = {
    "clinical_features": PROCESSED / "clinical_features.json",
    "dynamic_care_states": PROCESSED / "dynamic_care_states.json",
    "timeline": PROCESSED / "care_state_timeline.json",
    "transitions": PROCESSED / "care_state_transitions.json",
    "assistance": PROCESSED / "adaptive_assistance.json",
}

EXPECTED_JSON_RECORDS = 9723
EXPECTED_PATIENTS = 178

ERRORS = []
WARNINGS = []


def record(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{'ok' if ok else 'XX'}] {label} -> {status}" + (f"  ({detail})" if detail else ""))
    if not ok:
        ERRORS.append(f"{label}: {detail}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    print("=" * 70)
    print("ELDERDOCAI - ADAPTIVE CONTEXT VALIDATOR")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1-2. FILES EXIST
    # ----------------------------------------------------------
    record("1. adaptive_context.json exists", CTX_JSON.exists(), str(CTX_JSON))
    record("2. adaptive_context.csv exists", CTX_CSV.exists(), str(CTX_CSV))
    if not CTX_JSON.exists() or not CTX_CSV.exists():
        print(f"\nXX ERRORS: {len(ERRORS)}")
        print("PIPELINE VALIDATION: REVIEW REQUIRED")
        return

    print("\nLoading adaptive context and source datasets...")
    ctx = load_json(CTX_JSON)

    with open(CTX_CSV, "r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    sources = {
        name: load_json(path)
        for name, path in SOURCE_FILES.items()
    }

    # ----------------------------------------------------------
    # 3-5. COUNTS
    # ----------------------------------------------------------
    record("3. JSON record count = 9723", len(ctx) == EXPECTED_JSON_RECORDS, f"{len(ctx)}")
    record("4. CSV record count = 9723", len(csv_rows) == EXPECTED_JSON_RECORDS, f"{len(csv_rows)}")
    json_patients = {r["patient_id"] for r in ctx}
    record("5. unique patient count = 178", len(json_patients) == EXPECTED_PATIENTS, f"{len(json_patients)}")

    # ----------------------------------------------------------
    # 6. PATIENT ID MATCH vs all five sources
    # ----------------------------------------------------------
    for name, recs in sources.items():
        pset = {r["patient_id"] for r in recs}
        record(
            f"6. patient IDs match {name}",
            json_patients == pset,
            f"ctx={len(json_patients)} src={len(pset)}",
        )

    # ----------------------------------------------------------
    # 7. KEY UNIQUENESS
    # ----------------------------------------------------------
    key_counter = Counter(
        (r["patient_id"], r["window_start"], r["window_end"]) for r in ctx
    )
    dup_keys = [k for k, c in key_counter.items() if c > 1]
    record("7. no duplicate (patient_id, window_start, window_end) keys",
           not dup_keys, f"duplicates={len(dup_keys)}")

    # ----------------------------------------------------------
    # Source lookup maps
    # ----------------------------------------------------------
    timeline_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r for r in sources["timeline"]}
    transitions_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r for r in sources["transitions"]}
    assistance_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r for r in sources["assistance"]}

    # ----------------------------------------------------------
    # 8. EVERY KEY IN TIMELINE
    # ----------------------------------------------------------
    missing_timeline = [k for k in key_counter if k not in timeline_by_key]
    record("8. every record corresponds to a timeline record",
           not missing_timeline, f"missing={len(missing_timeline)}")

    counters = {
        "state_mismatch": 0, "score_mismatch": 0, "transition_mismatch": 0,
        "assistance_mismatch": 0, "non_numeric_active": 0, "out_of_range": 0,
        "no_data_not_null": 0, "no_data_status": 0, "no_data_mode": 0,
        "gap_direction": 0, "initial_direction": 0, "escalation_direction": 0,
        "deescalation_direction": 0, "changed_dims_mismatch": 0,
    }


    _validate_rows(
        ctx,
        sources,
        key_counter,
        timeline_by_key,
        transitions_by_key,
        assistance_by_key,
        counters,
    )

    _report(counters, ctx)


def _validate_rows(ctx, sources, key_counter, timeline_by_key, transitions_by_key,
                   assistance_by_key, counters):
    """Row-by-row consistency checks (checks 9-20)."""
    for r in ctx:
        pid = r["patient_id"]
        key = (pid, r["window_start"], r["window_end"])
        t_rec = timeline_by_key.get(key)
        tr_rec = transitions_by_key.get(key)
        a_rec = assistance_by_key.get(key)
        cs = r.get("care_state") or {}
        tr = r.get("transition") or {}
        aa = r.get("adaptive_assistance") or {}

        if t_rec is not None:
            if cs.get("state") != t_rec.get("care_state"):
                counters["state_mismatch"] += 1
            if cs.get("overall_score") != t_rec.get("overall_score"):
                counters["score_mismatch"] += 1
        if tr_rec is not None:
            if (
                tr.get("type") != tr_rec.get("transition_type")
                or tr.get("direction") != tr_rec.get("transition_direction")
                or tr.get("magnitude") != tr_rec.get("transition_magnitude")
                or tr.get("score_delta") != tr_rec.get("score_delta")
            ):
                counters["transition_mismatch"] += 1
            # New (11b): top-level changed_dimensions mirrors the compact
            # representation of the source transition's dimension-level deltas.
            src_dims = [
                {
                    "dimension": d.get("dimension"),
                    "direction": d.get("direction"),
                    "delta": d.get("delta"),
                    "previous_score": d.get("previous_score"),
                    "current_score": d.get("current_score"),
                }
                for d in (tr_rec.get("changed_dimensions") or [])
            ]
            ctx_dims = [d for d in (r.get("changed_dimensions") or [])]
            if ctx_dims != src_dims:
                counters["changed_dims_mismatch"] += 1
        if a_rec is not None:
            if (
                aa.get("mode") != a_rec.get("assistance_mode")
                or aa.get("priority") != a_rec.get("priority")
            ):
                counters["assistance_mismatch"] += 1

        state = cs.get("state")
        score = cs.get("overall_score")
        if state == "NO_DATA":
            if score is not None:
                counters["no_data_not_null"] += 1
            if r.get("context_status") != "DATA_GAP":
                counters["no_data_status"] += 1
            if aa.get("mode") != "WAIT_FOR_DATA":
                counters["no_data_mode"] += 1
        else:
            if not isinstance(score, (int, float)):
                counters["non_numeric_active"] += 1
            elif score < 0 or score > 1:
                counters["out_of_range"] += 1
            expected_status = (
                "INITIAL"
                if tr.get("type") == "INITIAL_STATE"
                else "ACTIVE"
            )
            if r.get("context_status") != expected_status:
                counters["no_data_status"] += 1

        ttype = tr.get("type")
        direction = tr.get("direction")
        if ttype == "GAP" and direction != "UNKNOWN":
            counters["gap_direction"] += 1
        if ttype == "INITIAL_STATE" and direction != "INITIAL":
            counters["initial_direction"] += 1
        if ttype == "STATE_ESCALATION" and direction != "INCREASING":
            counters["escalation_direction"] += 1
        if ttype == "STATE_DEESCALATION" and direction != "DECREASING":
            counters["deescalation_direction"] += 1

    record("9. care-state values match timeline",
           counters["state_mismatch"] == 0, f"mismatch={counters['state_mismatch']}")
    record("9b. overall_score matches timeline",
           counters["score_mismatch"] == 0, f"mismatch={counters['score_mismatch']}")
    record("10. transition values match transitions.json",
           counters["transition_mismatch"] == 0, f"mismatch={counters['transition_mismatch']}")
    record("11. adaptive-assistance values match assistance.json",
           counters["assistance_mismatch"] == 0, f"mismatch={counters['assistance_mismatch']}")
    record("11b. top-level changed_dimensions match source transitions",
           counters["changed_dims_mismatch"] == 0, f"mismatch={counters['changed_dims_mismatch']}")
    record("12. non-NO_DATA scores are numeric",
           counters["non_numeric_active"] == 0, f"non_numeric={counters['non_numeric_active']}")
    record("13. non-NO_DATA scores within [0, 1]",
           counters["out_of_range"] == 0, f"out_of_range={counters['out_of_range']}")
    record("14. NO_DATA scores remain null",
           counters["no_data_not_null"] == 0, f"non_null={counters['no_data_not_null']}")
    record("15. NO_DATA -> DATA_GAP / active states -> ACTIVE",
           counters["no_data_status"] == 0, f"mismatch={counters['no_data_status']}")
    record("16. NO_DATA -> WAIT_FOR_DATA",
           counters["no_data_mode"] == 0, f"mismatch={counters['no_data_mode']}")
    record("17. GAP -> UNKNOWN direction",
           counters["gap_direction"] == 0, f"mismatch={counters['gap_direction']}")
    record("18. INITIAL_STATE -> INITIAL direction",
           counters["initial_direction"] == 0, f"mismatch={counters['initial_direction']}")
    record("19. STATE_ESCALATION -> INCREASING direction",
           counters["escalation_direction"] == 0, f"mismatch={counters['escalation_direction']}")
    record("20. STATE_DEESCALATION -> DECREASING direction",
           counters["deescalation_direction"] == 0, f"mismatch={counters['deescalation_direction']}")


def _report(counters, ctx):
    """context_status distribution + final pass/review."""
    status_counts = Counter(r["context_status"] for r in ctx)
    print("\nAdaptive-context status distribution:")
    for status in ("ACTIVE", "INITIAL", "DATA_GAP"):
        print(f"  {status:<12} {status_counts.get(status, 0)}")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nXX ERRORS: {len(ERRORS)}")
    for error in ERRORS:
        print(f"  - {error}")
    print(f"\n** WARNINGS: {len(WARNINGS)}")
    for warning in WARNINGS:
        print(f"  - {warning}")

    print("\n" + "-" * 70)
    if ERRORS:
        print("PIPELINE VALIDATION: REVIEW REQUIRED")
        print("Fix the reported errors before treating the pipeline as validated.")
    else:
        print("PIPELINE VALIDATION: PASS")
        print("The adaptive-context layer is internally consistent with the upstream pipeline.")
    print("-" * 70)


if __name__ == "__main__":
    main()