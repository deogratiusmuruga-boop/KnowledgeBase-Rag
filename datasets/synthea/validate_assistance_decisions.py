"""
ElderDocAI - Assistance Decision Validator

Validates elderdocai/processed/assistance_decisions.json and .csv
against the validated adaptive-context dataset.

Checks are structural / logical only. This validator does not
establish clinical validity.
"""

import csv
import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "elderdocai" / "processed"

DEC_JSON = PROCESSED / "assistance_decisions.json"
DEC_CSV = PROCESSED / "assistance_decisions.csv"
CTX_JSON = PROCESSED / "adaptive_context.json"

EXPECTED_JSON_RECORDS = 9723
EXPECTED_PATIENTS = 178

ERRORS = []
WARNINGS = []

REQUIRED_FIELDS = [
    "patient_id",
    "window_start",
    "window_end",
    "context_status",
    "care_state",
    "overall_score",
    "transition_type",
    "transition_direction",
    "score_delta",
    "adaptive_assistance_mode",
    "priority",
    "assistance_strategy",
    "decision_reason_codes",
    "decision_reasons",
    "changed_dimensions",
    "recommended_behavior",
    "safety_constraints",
    "interpretation",
]

# (transition_type) -> expected strategy
EXPECTED_STRATEGY = {
    "INITIAL_STATE": "ONBOARDING_SUPPORT",
    "GAP": "DATA_COLLECTION_SUPPORT",
    "STATE_ESCALATION": "ENHANCED_CONTEXT_SUPPORT",
    "STATE_DEESCALATION": "ADAPTIVE_DEESCALATION_SUPPORT",
    "INCREASING_ACTIVITY": "MONITORING_SUPPORT",
    "DECREASING_ACTIVITY": "FOLLOW_UP_SUPPORT",
}

# (assistance_mode) -> compatible strategy for NO_CHANGE
NO_CHANGE_MODE_STRATEGY = {
    "LIGHT_SUPPORT": "LIGHT_SUPPORT",
    "CONTEXTUAL_SUPPORT": "CONTEXTUAL_SUPPORT",
    "ENHANCED_SUPPORT": "ENHANCED_SUPPORT",
}

# (assistance_mode) -> expected strategy (full compatibility check)
MODE_STRATEGY = {
    "INITIAL_CONTEXT": "ONBOARDING_SUPPORT",
    "WAIT_FOR_DATA": "DATA_COLLECTION_SUPPORT",
    "ADAPTIVE_ESCALATION": "ENHANCED_CONTEXT_SUPPORT",
    "ADAPTIVE_DEESCALATION": "ADAPTIVE_DEESCALATION_SUPPORT",
    "MONITORING_SUPPORT": "MONITORING_SUPPORT",
    "FOLLOW_UP_SUPPORT": "FOLLOW_UP_SUPPORT",
    "LIGHT_SUPPORT": "LIGHT_SUPPORT",
    "CONTEXTUAL_SUPPORT": "CONTEXTUAL_SUPPORT",
    "ENHANCED_SUPPORT": "ENHANCED_SUPPORT",
}

# Unsupported medical-claim phrases (safety validation).
UNSAFE_PHRASES = [
    "has disease",
    "is deteriorating",
    "is medically unstable",
    "is unstable",
    "at high medical risk",
    "high medical risk",
    "disease is worsening",
    "is worsening",
    "will deteriorate",
    "mortality risk",
    "death risk",
    "likely to develop",
]
DISCLAIMER_MARKERS = [
    "does not",
    "do not",
    "not a diagnosis",
    "no medical risk",
    "not medical",
    "must not",
    "should not",
    "never",
]
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
    print("ELDERDOCAI - ASSISTANCE DECISION VALIDATOR")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1-2. FILES EXIST
    # ----------------------------------------------------------
    record("1. assistance_decisions.json exists", DEC_JSON.exists(), str(DEC_JSON))
    record("2. assistance_decisions.csv exists", DEC_CSV.exists(), str(DEC_CSV))
    record("2b. adaptive_context.json exists", CTX_JSON.exists(), str(CTX_JSON))
    if not DEC_JSON.exists() or not DEC_CSV.exists() or not CTX_JSON.exists():
        print(f"\nXX ERRORS: {len(ERRORS)}")
        print("ASSISTANCE DECISION VALIDATION: REVIEW REQUIRED")
        return

    # ----------------------------------------------------------
    # 3. LOAD
    # ----------------------------------------------------------
    print("\nLoading assistance decisions and adaptive context...")
    try:
        dec = load_json(DEC_JSON)
        record("3. decisions JSON is valid JSON", True)
    except Exception as exc:
        record("3. decisions JSON is valid JSON", False, str(exc))
        print(f"\nXX ERRORS: {len(ERRORS)}")
        print("ASSISTANCE DECISION VALIDATION: REVIEW REQUIRED")
        return

    with open(DEC_CSV, "r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    ctx = load_json(CTX_JSON)

    # ----------------------------------------------------------
    # 4-6. RECORD COUNTS
    # ----------------------------------------------------------
    record("4. JSON record count = 9723", len(dec) == EXPECTED_JSON_RECORDS, f"{len(dec)}")
    record("5. CSV record count = 9723", len(csv_rows) == EXPECTED_JSON_RECORDS, f"{len(csv_rows)}")
    dec_patients = {r["patient_id"] for r in dec}
    record("6. unique patient count = 178", len(dec_patients) == EXPECTED_PATIENTS, f"{len(dec_patients)}")
    ctx_patients = {r["patient_id"] for r in ctx}
    record("6b. patient IDs match adaptive_context", dec_patients == ctx_patients,
           f"dec={len(dec_patients)} ctx={len(ctx_patients)}")

    # ----------------------------------------------------------
    # 7-8. KEY UNIQUENESS / COVERAGE
    # ----------------------------------------------------------
    key_counter = Counter((r["patient_id"], r["window_start"], r["window_end"]) for r in dec)
    dup_keys = [k for k, c in key_counter.items() if c > 1]
    record("7. no duplicate (patient_id, window_start, window_end) keys",
           not dup_keys, f"duplicates={len(dup_keys)}")

    ctx_keys = {(r["patient_id"], r["window_start"], r["window_end"]) for r in ctx}
    missing_ctx = [k for k in key_counter if k not in ctx_keys]
    record("8. every decision corresponds to an adaptive-context record",
           not missing_ctx, f"missing={len(missing_ctx)}")

    ctx_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r for r in ctx}


    # ----------------------------------------------------------
    # 9. STRUCTURE
    # ----------------------------------------------------------
    struct_bad = 0
    for r in dec:
        if not isinstance(r, dict):
            struct_bad += 1
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            struct_bad += 1
            continue
        for field in ("patient_id", "window_start", "window_end", "context_status",
                      "care_state", "transition_type", "transition_direction",
                      "adaptive_assistance_mode", "priority", "assistance_strategy",
                      "interpretation"):
            if not isinstance(r[field], str):
                struct_bad += 1
                break
        for field in ("decision_reason_codes", "decision_reasons",
                      "changed_dimensions", "recommended_behavior",
                      "safety_constraints"):
            if not isinstance(r[field], list):
                struct_bad += 1
                break
    record("9. required fields present with correct types", struct_bad == 0, f"bad={struct_bad}")

    # CSV/JSON consistency
    csv_keys = {(r["patient_id"], r["window_start"], r["window_end"]) for r in csv_rows}
    json_keys = set(key_counter.keys())
    record("9b. JSON/CSV keys match", csv_keys == json_keys,
           f"csv={len(csv_keys)} json={len(json_keys)}")

    # ----------------------------------------------------------
    # Row-wise decision checks (10-17)
    # ----------------------------------------------------------
    counters = {
        "context_mismatch": 0,
        "transition_mismatch": 0,
        "mode_mismatch": 0,
        "no_data_strategy": 0,
        "no_data_stable": 0,
        "initial_strategy": 0,
        "gap_direction": 0,
        "safety_unsafe": 0,
        "priority_mismatch": 0,
        "score_mismatch": 0,
    }

    _validate_rows(dec, ctx_by_key, counters)

    record("10. context_status -> strategy consistency",
           counters["context_mismatch"] == 0, f"mismatch={counters['context_mismatch']}")
    record("11. transition_type -> strategy consistency",
           counters["transition_mismatch"] == 0, f"mismatch={counters['transition_mismatch']}")
    record("12. assistance mode -> strategy compatibility",
           counters["mode_mismatch"] == 0, f"mismatch={counters['mode_mismatch']}")
    record("13. priority preserved from adaptive context",
           counters["priority_mismatch"] == 0, f"mismatch={counters['priority_mismatch']}")
    record("13b. overall_score preserved from adaptive context",
           counters["score_mismatch"] == 0, f"mismatch={counters['score_mismatch']}")
    record("14. NO_DATA -> DATA_COLLECTION_SUPPORT (not STABLE)",
           counters["no_data_strategy"] == 0 and counters["no_data_stable"] == 0,
           f"wrong_strategy={counters['no_data_strategy']} stable={counters['no_data_stable']}")
    record("15. INITIAL -> ONBOARDING_SUPPORT",
           counters["initial_strategy"] == 0, f"mismatch={counters['initial_strategy']}")
    record("16. GAP records keep UNKNOWN direction",
           counters["gap_direction"] == 0, f"mismatch={counters['gap_direction']}")
    record("17. no unsupported medical claims in generated text",
           counters["safety_unsafe"] == 0, f"unsafe={counters['safety_unsafe']}")

    # ----------------------------------------------------------
    # 18. SAFETY CONSTRAINTS PRESENT
    # ----------------------------------------------------------
    missing_constraints = 0
    for r in dec:
        sc = r.get("safety_constraints") or []
        if not all(c in sc for c in ("NO_DIAGNOSIS", "NO_MEDICAL_RISK_PREDICTION",
                                     "NO_DISEASE_PROGRESSION_INFERENCE",
                                     "DOCUMENTED_ACTIVITY_ONLY")):
            missing_constraints += 1
        if r.get("assistance_strategy") == "DATA_COLLECTION_SUPPORT" \
                and "NO_DATA_IS_NOT_STABILITY" not in sc:
            missing_constraints += 1
    record("18. safety constraints present (incl. NO_DATA_IS_NOT_STABILITY for DATA_GAP)",
           missing_constraints == 0, f"missing={missing_constraints}")

    _report(counters, dec)


def _validate_rows(dec, ctx_by_key, counters):
    for r in dec:
        key = (r["patient_id"], r["window_start"], r["window_end"])
        ctx_rec = ctx_by_key.get(key)
        if ctx_rec is None:
            continue

        ctx_status = ctx_rec["context_status"]
        ctx_cs = ctx_rec.get("care_state") or {}
        ctx_tr = ctx_rec.get("transition") or {}
        ctx_aa = ctx_rec.get("adaptive_assistance") or {}

        strategy = r["assistance_strategy"]

        # 10. context_status -> strategy
        expected = None
        if ctx_status == "INITIAL":
            expected = "ONBOARDING_SUPPORT"
        elif ctx_status == "DATA_GAP":
            expected = "DATA_COLLECTION_SUPPORT"
        elif ctx_status == "ACTIVE":
            expected = MODE_STRATEGY.get(ctx_aa.get("mode"))
        if expected is not None and strategy != expected:
            counters["context_mismatch"] += 1

        # 11. transition_type -> strategy
        ttype = ctx_tr.get("type")
        if ttype in EXPECTED_STRATEGY and strategy != EXPECTED_STRATEGY[ttype]:
            counters["transition_mismatch"] += 1
        if ttype == "NO_CHANGE":
            expected_nc = NO_CHANGE_MODE_STRATEGY.get(ctx_aa.get("mode"))
            if expected_nc is not None and strategy != expected_nc:
                counters["transition_mismatch"] += 1

        # 12. assistance mode -> strategy compatibility
        mode = ctx_aa.get("mode")
        expected_mode = MODE_STRATEGY.get(mode)
        if expected_mode is not None and strategy != expected_mode:
            counters["mode_mismatch"] += 1

        # 13. priority preserved
        if r["priority"] != ctx_aa.get("priority"):
            counters["priority_mismatch"] += 1
        # 13b. overall_score preserved
        if r["overall_score"] != ctx_cs.get("overall_score"):
            counters["score_mismatch"] += 1

        # 14. NO_DATA safety
        if ctx_cs.get("state") == "NO_DATA":
            if strategy != "DATA_COLLECTION_SUPPORT":
                counters["no_data_strategy"] += 1
            if "STABLE" in str(strategy):
                counters["no_data_stable"] += 1
        elif ctx_status == "DATA_GAP":
            if strategy != "DATA_COLLECTION_SUPPORT":
                counters["no_data_strategy"] += 1

        # 15. INITIAL safety
        if ctx_status == "INITIAL" and strategy != "ONBOARDING_SUPPORT":
            counters["initial_strategy"] += 1

        # 16. GAP direction preserved
        if ttype == "GAP" and r["transition_direction"] != "UNKNOWN":
            counters["gap_direction"] += 1

        # 17. safety text scan
        text_blob = " | ".join(
            str(x) for x in (
                r.get("decision_reasons") or []
            ) + (r.get("recommended_behavior") or [])
        ) + " " + str(r.get("interpretation") or "")
        lower = text_blob.lower()
        for phrase in UNSAFE_PHRASES:
            if phrase.lower() in lower:
                if not any(marker in lower for marker in DISCLAIMER_MARKERS):
                    counters["safety_unsafe"] += 1
                    break


def _report(counters, dec):
    strategy_counts = Counter(r["assistance_strategy"] for r in dec)
    priority_counts = Counter(r["priority"] for r in dec)
    context_counts = Counter(r["context_status"] for r in dec)

    print("\nassistance_strategy distribution:")
    for strategy in sorted(strategy_counts):
        print(f"  {strategy:<32} {strategy_counts[strategy]}")
    print("\npriority distribution:")
    for priority in ("LOW", "MODERATE", "HIGH"):
        print(f"  {priority:<12} {priority_counts.get(priority, 0)}")
    print("\ncontext_status distribution:")
    for status in ("ACTIVE", "INITIAL", "DATA_GAP"):
        print(f"  {status:<12} {context_counts.get(status, 0)}")

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
        print("ASSISTANCE DECISION VALIDATION: REVIEW REQUIRED")
        print("Fix the reported errors before treating the layer as validated.")
    else:
        print("ASSISTANCE DECISION VALIDATION: PASS")
        print("The assistance-decision layer is internally consistent with the adaptive context.")
    print("-" * 70)


if __name__ == "__main__":
    main()
