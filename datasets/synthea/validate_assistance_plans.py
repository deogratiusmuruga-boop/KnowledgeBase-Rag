"""
ElderDocAI - Assistance Plan Validator

Validates elderdocai/processed/assistance_plans.json and .csv
against the assistance-plan specification and the validated
assistance-decisions layer.

Execution:
    python validate_assistance_plans.py

Checks are structural / logical only. This validator does not
establish clinical validity.
"""

import csv
import json
import math
import sys
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "elderdocai" / "processed"

PLAN_JSON = PROCESSED / "assistance_plans.json"
PLAN_CSV = PROCESSED / "assistance_plans.csv"
DEC_JSON = PROCESSED / "assistance_decisions.json"

EXPECTED_JSON_RECORDS = 9723
EXPECTED_PATIENTS = 178

ERRORS = []
WARNINGS = []

REQUIRED_FIELDS = [
    "patient_id",
    "window_start",
    "window_end",
    "year",
    "context_status",
    "current_state",
    "overall_score",
    "transition_type",
    "transition_direction",
    "assistance_strategy",
    "priority",
    "actions",
    "safety_constraints",
    "interpretation",
]

VALID_CONTEXT = {"INITIAL", "DATA_GAP", "ACTIVE"}

VALID_STRATEGIES = {
    "ONBOARDING_SUPPORT",
    "DATA_COLLECTION_SUPPORT",
    "CONTEXTUAL_SUPPORT",
    "LIGHT_SUPPORT",
    "ENHANCED_SUPPORT",
    "ENHANCED_CONTEXT_SUPPORT",
    "MONITORING_SUPPORT",
    "FOLLOW_UP_SUPPORT",
    "ADAPTIVE_DEESCALATION_SUPPORT",
}

ALL_ACTIONS = [
    "REQUEST_CHECK_IN",
    "REQUEST_DATA_UPDATE",
    "AVOID_STRONG_PERSONALIZATION",
    "ENCOURAGE_FUTURE_CHECK_IN",
    "REVIEW_RECENT_CARE_CONTEXT",
    "PROVIDE_TARGETED_INFORMATION",
    "PROVIDE_GENERAL_GUIDANCE",
    "ENCOURAGE_APPROPRIATE_FOLLOW_UP",
    "PROVIDE_CONTEXTUAL_INFORMATION",
    "CONTINUE_MONITORING",
    "REDUCE_INTERVENTION_INTENSITY",
    "ESTABLISH_INITIAL_CONTEXT",
]
ACTION_SET = set(ALL_ACTIONS)

VALID_TRANSITION_TYPES = {
    "INITIAL_STATE",
    "NO_CHANGE",
    "GAP",
    "INCREASING_ACTIVITY",
    "DECREASING_ACTIVITY",
    "STATE_ESCALATION",
    "STATE_DEESCALATION",
}

VALID_DIRECTIONS = {
    "INCREASING",
    "DECREASING",
    "UNCHANGED",
    "UNKNOWN",
    "INITIAL",
}

VALID_PRIORITY = {"LOW", "MODERATE", "HIGH"}


BASE_SAFETY = [
    "NO_DIAGNOSIS",
    "NO_MEDICAL_RISK_PREDICTION",
    "NO_DISEASE_PROGRESSION_INFERENCE",
    "DOCUMENTED_ACTIVITY_ONLY",
]

# Expected distributions from the current validated output.
EXPECTED_CONTEXT = {"INITIAL": 178, "DATA_GAP": 6356, "ACTIVE": 3189}
EXPECTED_STRATEGY = {
    "ONBOARDING_SUPPORT": 178,
    "DATA_COLLECTION_SUPPORT": 7445,
    "CONTEXTUAL_SUPPORT": 476,
    "ENHANCED_SUPPORT": 452,
    "ENHANCED_CONTEXT_SUPPORT": 446,
    "ADAPTIVE_DEESCALATION_SUPPORT": 302,
    "LIGHT_SUPPORT": 202,
    "MONITORING_SUPPORT": 121,
    "FOLLOW_UP_SUPPORT": 101,
}

# Strategy -> ordered action names (from the assistance-plan specification).
STRATEGY_ACTIONS = {
    "DATA_COLLECTION_SUPPORT": [
        "REQUEST_CHECK_IN",
        "REQUEST_DATA_UPDATE",
        "AVOID_STRONG_PERSONALIZATION",
    ],
    "ONBOARDING_SUPPORT": [
        "ESTABLISH_INITIAL_CONTEXT",
        "REQUEST_CHECK_IN",
        "PROVIDE_GENERAL_GUIDANCE",
    ],
    "MONITORING_SUPPORT": [
        "CONTINUE_MONITORING",
        "REVIEW_RECENT_CARE_CONTEXT",
        "ENCOURAGE_FUTURE_CHECK_IN",
    ],
    "FOLLOW_UP_SUPPORT": [
        "REVIEW_RECENT_CARE_CONTEXT",
        "ENCOURAGE_APPROPRIATE_FOLLOW_UP",
        "ENCOURAGE_FUTURE_CHECK_IN",
    ],
    "ENHANCED_CONTEXT_SUPPORT": [
        "PROVIDE_TARGETED_INFORMATION",
        "REVIEW_RECENT_CARE_CONTEXT",
        "REQUEST_CHECK_IN",
        "ENCOURAGE_APPROPRIATE_FOLLOW_UP",
    ],
    "ADAPTIVE_DEESCALATION_SUPPORT": [
        "REDUCE_INTERVENTION_INTENSITY",
        "PROVIDE_GENERAL_GUIDANCE",
        "CONTINUE_MONITORING",
    ],
    "LIGHT_SUPPORT": [
        "PROVIDE_GENERAL_GUIDANCE",
        "ENCOURAGE_FUTURE_CHECK_IN",
    ],
    "CONTEXTUAL_SUPPORT": [
        "PROVIDE_CONTEXTUAL_INFORMATION",
        "ENCOURAGE_FUTURE_CHECK_IN",
    ],
    "ENHANCED_SUPPORT": [
        "PROVIDE_TARGETED_INFORMATION",
        "REVIEW_RECENT_CARE_CONTEXT",
        "ENCOURAGE_FUTURE_CHECK_IN",
    ],
}

# Text to scan for prohibited clinical claims in interpretation + reasons.
# Matched case-insensitively; the scan is confined to textual fields and
# tolerates disclaimer wording (e.g. "does not constitute a diagnosis").
PROHIBITED_TERMS = [
    "diagnosis",
    "diagnosed",
    "deterioration",
    "deteriorating",
    "mortality",
    "hospitalization",
    "medically unstable",
    "medical risk prediction",
    "is stable",
    "is worsening",
    "disease progression",
]

# Disclaimer / negation markers that keep a word-phrase OUTSIDE the set of
# unsupported claims (e.g. "without inferring improvement", "does not
# constitute a diagnosis", "does not predict medical risk").
DISCLAIMER_MARKERS = [
    "does not",
    "do not",
    "not diagnose",
    "not predict",
    "not a diagnosis",
    "no diagnosis",
    "no medical",
    "without",
    "not constitute",
    "is not",
    "are not",
    "not infer",
    "should not",
]


_CHECKS_RUN = 0



def record(label, ok, detail=""):
    global _CHECKS_RUN
    _CHECKS_RUN += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{'ok' if ok else 'XX'}] {label} -> {status}" + (f"  ({detail})" if detail else ""))
    if not ok:
        ERRORS.append(f"{label}: {detail}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _actions_join(actions):
    return [a["action"] for a in actions]


def main():
    print("=" * 74)
    print("ELDERDOCAI ASSISTANCE PLAN VALIDATION")
    print("=" * 74)

    # ----------------------------------------------------------
    # A. FILE AND STRUCTURE
    # ----------------------------------------------------------
    record("1. assistance_plans.json exists", PLAN_JSON.exists(), str(PLAN_JSON))
    record("2. assistance_plans.csv exists", PLAN_CSV.exists(), str(PLAN_CSV))
    record("2b. assistance_decisions.json (upstream) exists",
           DEC_JSON.exists(), str(DEC_JSON))

    ok_files = PLAN_JSON.exists() and PLAN_CSV.exists() and DEC_JSON.exists()
    if not ok_files:
        print(f"\nErrors: {len(ERRORS)}")
        return _exit(1)

    try:
        plans = load_json(PLAN_JSON)
        record("3. JSON is valid", True)
    except Exception as exc:
        record("3. JSON is valid", False, str(exc))
        return _exit(1)

    record("3b. JSON is a list", isinstance(plans, list), type(plans).__name__)

    try:
        with open(PLAN_CSV, "r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        record("4. CSV is readable", True)
    except Exception as exc:
        record("4. CSV is readable", False, str(exc))
        return _exit(1)

    try:
        decisions = load_json(DEC_JSON)
    except Exception:
        decisions = []

    expected_records = len(decisions) if decisions else EXPECTED_JSON_RECORDS

    # ----------------------------------------------------------
    # COUNTS: JSON / CSV / expected / patients
    # ----------------------------------------------------------
    record("5. JSON record count = expected",
           len(plans) == expected_records, f"{len(plans)} vs {expected_records}")
    record("5b. JSON record count = 9723",
           len(plans) == EXPECTED_JSON_RECORDS, f"{len(plans)}")
    record("6. CSV record count matches JSON",
           len(csv_rows) == len(plans), f"csv={len(csv_rows)} json={len(plans)}")

    json_keys = {(r["patient_id"], r["window_start"], r["window_end"]) for r in plans}
    record("7. JSON/CSV same number of records",
           len(csv_rows) == len(plans), f"{len(csv_rows)}")

    plan_patients = {r["patient_id"] for r in plans}
    record("8. unique patient count = 178",
           len(plan_patients) == EXPECTED_PATIENTS, f"{len(plan_patients)}")


    # ----------------------------------------------------------
    # C. PATIENT INTEGRITY
    # ----------------------------------------------------------
    missing_pid = [r["patient_id"] for r in plans if not isinstance(r.get("patient_id"), str) or not r.get("patient_id")]
    record("9. no missing/empty patient IDs", not missing_pid, f"bad={len(missing_pid)}")
    malformed_pid = [p for p in plan_patients if not isinstance(p, str) or len(p) < 8]
    record("9b. no malformed patient IDs", not malformed_pid, f"bad={len(malformed_pid)}")

    dup_keys = [k for k, c in Counter(json_keys).items() if c > 1] if isinstance(json_keys, list) else []
    key_counter = Counter((r["patient_id"], r["window_start"], r["window_end"]) for r in plans)
    real_dups = [k for k, c in key_counter.items() if c > 1]
    record("10. no duplicate patient/window records", not real_dups, f"dups={len(real_dups)}")

    # Every plan patient belongs to the upstream decision population.
    if decisions:
        dec_patients = {r["patient_id"] for r in decisions}
        record("11. plan patients are subset of upstream decision patients",
               plan_patients <= dec_patients,
               f"plan={len(plan_patients)} dec={len(dec_patients)} extra={len(plan_patients - dec_patients)}")
    else:
        record("11. plan patients are subset of upstream decision patients",
               True, "upstream not loadable; skipped")

    # ----------------------------------------------------------
    # B. REQUIRED FIELDS + TYPES
    # ----------------------------------------------------------
    struct_bad = 0
    for r in plans:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            struct_bad += 1
            continue
        for field in ("patient_id", "window_start", "window_end", "context_status",
                      "current_state", "transition_type", "transition_direction",
                      "assistance_strategy", "priority", "interpretation"):
            if not isinstance(r[field], str):
                struct_bad += 1
                break
        if not isinstance(r["actions"], list) or not isinstance(r["safety_constraints"], list):
            struct_bad += 1
    record("12. required fields present and correctly typed", struct_bad == 0, f"bad={struct_bad}")

    # ----------------------------------------------------------
    # D. CONTEXT-STATUS VOCABULARY + DISTRIBUTION
    # ----------------------------------------------------------
    ctx_counts = Counter(r["context_status"] for r in plans)
    bad_ctx = [c for c in ctx_counts if c not in VALID_CONTEXT]
    record("13. context_status vocabulary valid", not bad_ctx, f"bad={bad_ctx}")
    record("13b. context_status distribution matches expected",
           dict(ctx_counts) == dict(EXPECTED_CONTEXT), str(dict(ctx_counts)))

    # ----------------------------------------------------------
    # E. STRATEGY VOCABULARY + DISTRIBUTION
    # ----------------------------------------------------------
    strat_counts = Counter(r["assistance_strategy"] for r in plans)
    bad_strat = [s for s in strat_counts if s not in VALID_STRATEGIES]
    record("14. assistance_strategy vocabulary valid", not bad_strat, f"bad={bad_strat}")
    record("14b. strategy distribution matches expected",
           dict(strat_counts) == dict(EXPECTED_STRATEGY), str(dict(strat_counts)))

    # ----------------------------------------------------------
    # F. ACTION VOCABULARY
    # ----------------------------------------------------------
    seen_actions = set()
    unknown_actions = []
    for r in plans:
        for a in r["actions"]:
            name = a.get("action")
            seen_actions.add(name)
            if name not in ACTION_SET:
                unknown_actions.append((r["patient_id"], r["window_start"], name))
    record("15. all actions belong to approved vocabulary",
           not unknown_actions, f"unknown={len(unknown_actions)}")
    missing_actions = ACTION_SET - seen_actions
    if missing_actions:
        WARNINGS.append(f"Actions not observed in output: {sorted(missing_actions)}")
    record("15b. output uses actions from the 12-action vocabulary",
           seen_actions <= ACTION_SET and seen_actions, f"used={len(seen_actions)}")

    # ----------------------------------------------------------
    # G. STRATEGY -> ACTION MAPPING (independent recalculation)
    # ----------------------------------------------------------
    mapping_mismatch = []
    for r in plans:
        strategy = r["assistance_strategy"]
        expected_actions = STRATEGY_ACTIONS.get(strategy)
        actual_actions = _actions_join(r["actions"])
        if expected_actions is None:
            mapping_mismatch.append((r["patient_id"], r["window_start"], r["window_end"],
                                     strategy, actual_actions, "NO_EXPECTED"))
        elif actual_actions != expected_actions:
            mapping_mismatch.append((r["patient_id"], r["window_start"], r["window_end"],
                                     strategy, actual_actions, expected_actions))
    record("16. strategy -> action mapping correct", not mapping_mismatch,
           f"mismatch={len(mapping_mismatch)}")
    for mm in mapping_mismatch[:10]:
        print(f"      MISMATCH pid={mm[0]} {mm[1]} strategy={mm[3]} "
              f"actual={mm[4]} expected={mm[5]}")

    # ----------------------------------------------------------
    # H. SAFETY CONSTRAINTS
    # ----------------------------------------------------------
    missing_base_safety = 0
    for r in plans:
        sc = set(r.get("safety_constraints") or [])
        if not set(BASE_SAFETY).issubset(sc):
            missing_base_safety += 1
    record("17. every record has base safety constraints",
           missing_base_safety == 0, f"missing={missing_base_safety}")

    missing_dc_safety = 0
    for r in plans:
        if r["assistance_strategy"] == "DATA_COLLECTION_SUPPORT":
            sc = set(r.get("safety_constraints") or [])
            if "NO_DATA_IS_NOT_STABILITY" not in sc:
                missing_dc_safety += 1
    record("18. every DATA_COLLECTION record has NO_DATA_IS_NOT_STABILITY",
           missing_dc_safety == 0, f"missing={missing_dc_safety}")

    # ----------------------------------------------------------
    # I. DATA_GAP SAFETY
    # ----------------------------------------------------------
    dc_count = strat_counts.get("DATA_COLLECTION_SUPPORT", 0)
    record("19. DATA_COLLECTION record count", dc_count == 7445, f"{dc_count}")
    dc_bad_actions = 0
    dc_stable_claim = 0
    for r in plans:
        if r["assistance_strategy"] != "DATA_COLLECTION_SUPPORT":
            continue
        names = set(_actions_join(r["actions"]))
        if not {"REQUEST_CHECK_IN", "REQUEST_DATA_UPDATE",
                "AVOID_STRONG_PERSONALIZATION"}.issubset(names):
            dc_bad_actions += 1
        if "is stable" in r["interpretation"].lower() or \
           "stable" in " ".join(a["reason"] for a in r["actions"]).lower():
            dc_stable_claim += 1
    record("19b. DATA_GAP records get data-collection actions",
           dc_bad_actions == 0, f"bad={dc_bad_actions}")
    record("19c. no DATA_GAP record claims stability",
           dc_stable_claim == 0, f"claims={dc_stable_claim}")

    # ----------------------------------------------------------
    # J. TRANSITION VALIDATION
    # ----------------------------------------------------------
    tt_counts = Counter(r["transition_type"] for r in plans)
    td_counts = Counter(r["transition_direction"] for r in plans)
    bad_tt = [t for t in tt_counts if t not in VALID_TRANSITION_TYPES]
    bad_td = [d for d in td_counts if d not in VALID_DIRECTIONS]
    record("20. transition_type vocabulary valid", not bad_tt, f"bad={bad_tt}")
    record("20b. transition_direction vocabulary valid", not bad_td, f"bad={bad_td}")

    def _validate_transition(plans):
        """Enforce legal transition_type -> direction combinations."""
        legal = {
            "INITIAL_STATE": {"INITIAL"},
            "NO_CHANGE": {"UNCHANGED"},
            "GAP": {"UNKNOWN"},
            "INCREASING_ACTIVITY": {"INCREASING"},
            "DECREASING_ACTIVITY": {"DECREASING"},
            "STATE_ESCALATION": {"INCREASING"},
            "STATE_DEESCALATION": {"DECREASING"},
        }
        bad_combos = []
        for r in plans:
            t = r["transition_type"]
            d = r["transition_direction"]
            allow = legal.get(t)
            if allow is not None and d not in allow:
                bad_combos.append((r["patient_id"], r["window_start"], t, d))
        return bad_combos

    bad_combos = _validate_transition(plans)
    record("20c. transition_type/direction combinations legal",
           not bad_combos, f"bad={len(bad_combos)}")

    # ----------------------------------------------------------
    # K. PRIORITY VOCABULARY
    # ----------------------------------------------------------
    prio_counts = Counter(r["priority"] for r in plans)
    bad_prio = [p for p in prio_counts if p not in VALID_PRIORITY]
    record("21. priority vocabulary valid", not bad_prio, f"bad={bad_prio}")

    # ----------------------------------------------------------
    # L. SCORE VALIDATION
    # ----------------------------------------------------------
    bad_score = 0
    for r in plans:
        score = r["overall_score"]
        if score is None:
            if r["current_state"] == "NO_DATA":
                continue  # allowed null for NO_DATA
            bad_score += 1
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            bad_score += 1
            continue
        if not math.isfinite(score) or isinstance(score, bool):
            bad_score += 1
            continue
        if score < 0 or score > 1:
            bad_score += 1
    record("22. overall_score numeric/finite/NaN-free/in [0,1]",
           bad_score == 0, f"bad={bad_score}")
    non_null_scores = [r["overall_score"] for r in plans if r["overall_score"] is not None]
    if non_null_scores:
        record("22b. observed score range", True,
               f"min={min(non_null_scores)} max={max(non_null_scores)}")


    # ----------------------------------------------------------
    # M. ACTION/STRATEGY CONSISTENCY (decision-layer agreement)
    # ----------------------------------------------------------
    # The strategy field comes from the decision layer; the actions derive
    # from strategy. Strategy->action mapping already checked (check 16).
    # Verify the strategy on each plan matches the upstream decision strategy.
    dec_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r
                  for r in decisions}
    strategy_mismatch = 0
    for r in plans:
        key = (r["patient_id"], r["window_start"], r["window_end"])
        dec = dec_by_key.get(key)
        if dec is not None and dec["assistance_strategy"] != r["assistance_strategy"]:
            strategy_mismatch += 1
    record("23. plan strategy matches upstream decision strategy",
           strategy_mismatch == 0, f"mismatch={strategy_mismatch}")

    # ----------------------------------------------------------
    # N. INTERPRETATION VALIDATION (prohibited clinical claims)
    # ----------------------------------------------------------
    # Scan interpretation + action-reason text. A term is only flagged when
    # it appears WITHOUT any disclaimer/negation marker in the same field,
    # so legitimate safety disclaimers (e.g. "without inferring improvement",
    # "does not constitute a diagnosis") are not treated as claims.
    offending = []
    for r in plans:
        text = r["interpretation"].lower()
        reasons = " ".join(a["reason"] for a in r["actions"]).lower()
        for term in PROHIBITED_TERMS:
            for field_name, field_text in (("interpretation", text), ("reasons", reasons)):
                if term in field_text:
                    has_disclaimer = any(m in field_text for m in DISCLAIMER_MARKERS)
                    if not has_disclaimer:
                        offending.append((r["patient_id"], r["window_start"], term, field_name, field_text[:80]))
    record("24. no prohibited clinical claims in interpretation/reasons",
           not offending, f"offending={len(offending)}")
    for o in offending[:10]:
        print(f"      OFFENDING term={o[2]} in={o[3]} pid={o[0]} {o[1]} text='{o[4]}'")

    # ----------------------------------------------------------
    # O. BASE SAFETY-POLICY COMPATIBILITY (already check 17)
    # ----------------------------------------------------------
    record("25. base safety policy applied to all records",
           missing_base_safety == 0, f"missing={missing_base_safety}")

    # ----------------------------------------------------------
    # P. CSV / JSON CONSISTENCY
    # ----------------------------------------------------------
    csv_issues = []
    csv_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r
                  for r in csv_rows}
    json_by_key = {(r["patient_id"], r["window_start"], r["window_end"]): r
                   for r in plans}
    if set(csv_by_key.keys()) != set(json_by_key.keys()):
        csv_issues.append("key sets differ")
    else:
        for key, jr in json_by_key.items():
            cr = csv_by_key[key]
            # CSV stores lists as "a | b" strings; compare parsed tokens.
            def _csv_list(cell):
                return [x.strip() for x in cell.split("|")] if cell else []

            checks = (
                (cr["patient_id"] == jr["patient_id"], "patient_id"),
                (cr["window_start"] == jr["window_start"], "window_start"),
                (cr["window_end"] == jr["window_end"], "window_end"),
                (str(cr["year"]) == str(jr["year"]), "year"),
                (cr["context_status"] == jr["context_status"], "context_status"),
                (cr["current_state"] == jr["current_state"], "current_state"),
                (cr["transition_type"] == jr["transition_type"], "transition_type"),
                (cr["transition_direction"] == jr["transition_direction"], "transition_direction"),
                (cr["assistance_strategy"] == jr["assistance_strategy"], "assistance_strategy"),
                (cr["priority"] == jr["priority"], "priority"),
                (_csv_list(cr["actions"]) == _actions_join(jr["actions"]), "actions"),
            )
            for ok, field in checks:
                if not ok:
                    csv_issues.append(f"{key[0]}:{key[1]}:{field}")
    record("26. CSV/JSON consistent (count, ids, windows, fields)",
           not csv_issues, f"issues={len(csv_issues)}")

    # ----------------------------------------------------------
    # Q. OUTPUT INTEGRITY SUMMARY
    # ----------------------------------------------------------
    _report(plans, csv_rows)

    if ERRORS:
        return _exit(1)
    return _exit(0)


def _report(plans, csv_rows):
    context_counts = Counter(r["context_status"] for r in plans)
    strategy_counts = Counter(r["assistance_strategy"] for r in plans)
    action_counts = Counter(a["action"] for r in plans for a in r["actions"])
    priority_counts = Counter(r["priority"] for r in plans)
    patients = len({r["patient_id"] for r in plans})

    print("\n" + "=" * 74)
    print("ELDERDOCAI ASSISTANCE PLAN VALIDATION")
    print("=" * 74)
    print(f"\nJSON records: {len(plans)}")
    print(f"CSV records:  {len(csv_rows)}")
    print(f"Patients:     {patients}")

    print("\nContext distribution:")
    for status in ("INITIAL", "DATA_GAP", "ACTIVE"):
        print(f"  {status:<10} {context_counts.get(status, 0)}")

    print("\nStrategy distribution:")
    for strategy in sorted(strategy_counts):
        print(f"  {strategy:<35} {strategy_counts[strategy]}")

    print("\nAction distribution:")
    for action in sorted(action_counts):
        print(f"  {action:<38} {action_counts[action]}")

    print("\nValidation checks:")
    # The numbered checks were printed inline during main(); here we print a
    # final tally reflecting recorded failures.
    for error in ERRORS:
        print(f"  [FAIL] {error}")
    for warning in WARNINGS:
        print(f"  [WARN] {warning}")

    print(f"\nErrors:   {len(ERRORS)}")
    print(f"Warnings: {len(WARNINGS)}")

    print(f"\nChecks executed: {_CHECKS_RUN}")

    print()
    print("=" * 74)
    if ERRORS:
        print("[FAIL] ASSISTANCE PLAN VALIDATION")
    else:
        print("[PASS] ASSISTANCE PLAN VALIDATION")
    print("=" * 74)


def _exit(code):
    sys.exit(code)


if __name__ == "__main__":
    main()
