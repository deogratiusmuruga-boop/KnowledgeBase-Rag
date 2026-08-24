"""
ElderDocAI - Assistance Decision Builder

Converts the validated adaptive-context dataset into an explicit
assistance strategy describing HOW ElderDocAI should adapt its behavior.

Pipeline position:
    adaptive context -> ASSISTANCE DECISION -> assistance strategy
    -> response behavior

Design:
    - One decision record per adaptive-context window (9,723).
    - Primary input: elderdocai/processed/adaptive_context.json.
    - Upstream values (care state, score, transition, assistance mode,
      priority, context status) are REUSED, never recomputed.
    - Deterministic rule-based decision only. No ML. No medical risk
      prediction. No diagnostic classification.

Safety scope:
    The decision layer describes documented care activity and temporal
    changes. NO_DATA does not mean STABLE. No diagnosis, no disease
    progression, no deterioration prediction.
"""

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "elderdocai" / "processed"

INPUT_JSON = PROCESSED / "adaptive_context.json"
OUTPUT_JSON = PROCESSED / "assistance_decisions.json"
OUTPUT_CSV = PROCESSED / "assistance_decisions.csv"

# ---------------------------------------------------------------------------
# Strategy catalog
# ---------------------------------------------------------------------------
# Each entry defines the canonical strategy name, a safe reason phrase,
# recommended behaviors, and the safety-scoped interpretation.
# The interpretation footer is identical across strategies (deterministic).

BASE_SAFETY = [
    "NO_DIAGNOSIS",
    "NO_MEDICAL_RISK_PREDICTION",
    "NO_DISEASE_PROGRESSION_INFERENCE",
    "DOCUMENTED_ACTIVITY_ONLY",
]

STRATEGY_CATALOG = {
    "ONBOARDING_SUPPORT": {
        "reason": "First observed care-state window; establish initial assistance context.",
        "behaviors": [
            "establish initial context",
            "encourage an initial check-in",
            "introduce available assistance options",
            "avoid assumptions about medical status",
        ],
    },
    "DATA_COLLECTION_SUPPORT": {
        "reason": "Current documented information is missing; encourage a data update.",
        "behaviors": [
            "recognize that current information is missing",
            "encourage a new check-in or data update",
            "avoid aggressive personalization",
            "avoid medical conclusions",
        ],
    },
    "ENHANCED_CONTEXT_SUPPORT": {
        "reason": "Documented care activity moved to a higher activity state.",
        "behaviors": [
            "increase contextual awareness",
            "provide more targeted assistance",
            "consider recently increased documented activity",
            "encourage appropriate follow-up/check-in",
        ],
    },
    "ADAPTIVE_DEESCALATION_SUPPORT": {
        "reason": "Documented care activity moved to a lower activity state.",
        "behaviors": [
            "reduce unnecessary intervention intensity",
            "maintain appropriate support",
            "avoid interpreting reduced activity as medical improvement",
            "continue observing future check-ins",
        ],
    },
    "MONITORING_SUPPORT": {
        "reason": "Documented care activity increased within the same state.",
        "behaviors": [
            "acknowledge increased documented care activity",
            "maintain contextual awareness",
            "provide relevant support",
            "continue observing future changes",
        ],
    },
    "FOLLOW_UP_SUPPORT": {
        "reason": "Documented care activity decreased within the same state.",
        "behaviors": [
            "encourage appropriate follow-up/check-in",
            "avoid interpreting reduced activity as medical improvement",
            "maintain awareness of possible data gaps",
        ],
    },
    "LIGHT_SUPPORT": {
        "reason": "Documented care activity remained low without a major state change.",
        "behaviors": [
            "maintain light support",
            "continue observing documented care activity",
        ],
    },
    "CONTEXTUAL_SUPPORT": {
        "reason": "Documented care activity remained moderate without a major state change.",
        "behaviors": [
            "maintain contextual support",
            "continue observing documented care activity",
        ],
    },
    "ENHANCED_SUPPORT": {
        "reason": "Documented care activity remained high without a major state change.",
        "behaviors": [
            "maintain enhanced support",
            "continue observing documented care activity",
        ],
    },
}

INTERPRETATION = (
    "Rule-based assistance decision derived from documented care activity "
    "and temporal changes. This output does not constitute a diagnosis and "
    "does not predict medical risk or disease progression."
)

# Transition type -> decision strategy (NO_CHANGE resolved separately).
TRANSITION_STRATEGY = {
    "INITIAL_STATE": "ONBOARDING_SUPPORT",
    "GAP": "DATA_COLLECTION_SUPPORT",
    "STATE_ESCALATION": "ENHANCED_CONTEXT_SUPPORT",
    "STATE_DEESCALATION": "ADAPTIVE_DEESCALATION_SUPPORT",
    "INCREASING_ACTIVITY": "MONITORING_SUPPORT",
    "DECREASING_ACTIVITY": "FOLLOW_UP_SUPPORT",
}

# Assistance mode -> compatible decision strategy (NO_CHANGE tier modes).
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

def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _resolve_strategy(record):
    """Deterministic strategy selection with INITIAL/DATA_GAP precedence."""
    ctx_status = record["context_status"]
    cs = record.get("care_state") or {}
    state = cs.get("state")
    tr = record.get("transition") or {}
    ttype = tr.get("type")
    aa = record.get("adaptive_assistance") or {}
    mode = aa.get("mode")

    if ctx_status == "INITIAL" or ttype == "INITIAL_STATE":
        return "ONBOARDING_SUPPORT"
    if ctx_status == "DATA_GAP" or state == "NO_DATA" or ttype == "GAP":
        return "DATA_COLLECTION_SUPPORT"
    if ttype == "STATE_ESCALATION":
        return "ENHANCED_CONTEXT_SUPPORT"
    if ttype == "STATE_DEESCALATION":
        return "ADAPTIVE_DEESCALATION_SUPPORT"
    if ttype == "INCREASING_ACTIVITY":
        return "MONITORING_SUPPORT"
    if ttype == "DECREASING_ACTIVITY":
        return "FOLLOW_UP_SUPPORT"
    if ttype == "NO_CHANGE":
        # Upstream NO_CHANGE modes: LIGHT / CONTEXTUAL / ENHANCED support.
        # The strategy inherits the validated assistance mode (score tier).
        return MODE_STRATEGY.get(mode, "CONTEXTUAL_SUPPORT")
    # Unexpected transition type -> deterministic safe fallback.
    return MODE_STRATEGY.get(mode, "CONTEXTUAL_SUPPORT")


def _dimension_reasons(changed_dimensions):
    """Safe, documented-activity-only reasons from dimension-level deltas."""
    reasons = []
    for change in changed_dimensions or []:
        dimension = change.get("dimension")
        direction = change.get("direction")
        delta = change.get("delta")
        if not dimension:
            continue
        if delta is None:
            reasons.append(
                f"Documented activity signal {dimension} "
                f"{str(direction).lower()} during this period."
            )
        else:
            reasons.append(
                f"Documented activity signal {dimension} "
                f"{str(direction).lower()} (delta {delta:+.4f})."
            )
    return reasons


def _transition_reason(record):
    """Safe transition description using only documented states/scores."""
    tr = record.get("transition") or {}
    prev_state = tr.get("previous_state")
    current_state = tr.get("current_state")
    prev_score = tr.get("previous_score")
    current_score = tr.get("current_score")
    if prev_state and current_state and prev_state != current_state:
        return (
            f"Documented care state moved from {prev_state} to {current_state} "
            f"(overall score {prev_score} -> {current_score})."
        )
    if current_state:
        return f"Documented care state remained {current_state} in this window."
    return "Documented care-state information for this window."


def _build_record(record):
    pid = record["patient_id"]
    window_start = record["window_start"]
    window_end = record["window_end"]

    cs = record.get("care_state") or {}
    tr = record.get("transition") or {}
    aa = record.get("adaptive_assistance") or {}

    state = cs.get("state")
    score = cs.get("overall_score")
    ttype = tr.get("type")
    tdir = tr.get("direction")
    delta = tr.get("score_delta")
    mode = aa.get("mode")
    priority = aa.get("priority")
    changed_dimensions = record.get("changed_dimensions") or []

    strategy = _resolve_strategy(record)
    catalog = STRATEGY_CATALOG[strategy]

    reason_codes = _dedupe(
        [ttype, strategy, mode] + list(aa.get("reason_codes") or [])
    )
    reasons = _dedupe(
        [catalog["reason"], _transition_reason(record)]
        + _dimension_reasons(changed_dimensions)
        + list(aa.get("reasons") or [])
    )

    safety_constraints = list(BASE_SAFETY)
    if strategy == "DATA_COLLECTION_SUPPORT":
        safety_constraints.append("NO_DATA_IS_NOT_STABILITY")

    return {
        "patient_id": pid,
        "window_start": window_start,
        "window_end": window_end,
        "year": record.get("year"),
        "context_status": record["context_status"],
        "care_state": state,
        "overall_score": score,
        "transition_type": ttype,
        "transition_direction": tdir,
        "score_delta": delta,
        "adaptive_assistance_mode": mode,
        "priority": priority,
        "assistance_strategy": strategy,
        "decision_reason_codes": reason_codes,
        "decision_reasons": reasons,
        "changed_dimensions": changed_dimensions,
        "recommended_behavior": list(catalog["behaviors"]),
        "safety_constraints": safety_constraints,
        "interpretation": INTERPRETATION,
    }

def build():
    print("=" * 70)
    print("ELDERDOCAI - ASSISTANCE DECISION BUILDER")
    print("=" * 70)

    print("\nLoading adaptive context...")
    context = _load_json(INPUT_JSON)
    print(f"  adaptive_context records: {len(context)}")

    records = [_build_record(record) for record in context]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    csv_fields = [
        "patient_id",
        "window_start",
        "window_end",
        "year",
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

    def _join(items):
        return " | ".join(str(item) for item in items)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "patient_id": record["patient_id"],
                    "window_start": record["window_start"],
                    "window_end": record["window_end"],
                    "year": record["year"],
                    "context_status": record["context_status"],
                    "care_state": record["care_state"],
                    "overall_score": record["overall_score"],
                    "transition_type": record["transition_type"],
                    "transition_direction": record["transition_direction"],
                    "score_delta": record["score_delta"],
                    "adaptive_assistance_mode": record["adaptive_assistance_mode"],
                    "priority": record["priority"],
                    "assistance_strategy": record["assistance_strategy"],
                    "decision_reason_codes": _join(record["decision_reason_codes"]),
                    "decision_reasons": _join(record["decision_reasons"]),
                    "changed_dimensions": _join(record["changed_dimensions"]),
                    "recommended_behavior": _join(record["recommended_behavior"]),
                    "safety_constraints": _join(record["safety_constraints"]),
                    "interpretation": record["interpretation"],
                }
            )

    from collections import Counter

    strategy_counts = Counter(r["assistance_strategy"] for r in records)
    priority_counts = Counter(r["priority"] for r in records)
    context_counts = Counter(r["context_status"] for r in records)
    patients = len(set(r["patient_id"] for r in records))

    print("\nSaving JSON...")
    print(f"  {OUTPUT_JSON}")
    print("Saving CSV...")
    print(f"  {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("ASSISTANCE DECISION BUILDER - SUMMARY")
    print("=" * 70)
    print(f"Patients processed:        {patients}")
    print(f"Decision records:          {len(records)}")
    print()
    print("assistance_strategy distribution:")
    for strategy in sorted(strategy_counts):
        print(f"  {strategy:<32} {strategy_counts[strategy]}")
    print()
    print("priority distribution:")
    for priority in ("LOW", "MODERATE", "HIGH"):
        print(f"  {priority:<12} {priority_counts.get(priority, 0)}")
    print()
    print("context_status distribution:")
    for status in ("ACTIVE", "INITIAL", "DATA_GAP"):
        print(f"  {status:<12} {context_counts.get(status, 0)}")
    print()

    print("Important:")
    print("  The assistance decision layer describes documented care activity")
    print("  and temporal changes. NO_DATA does not mean STABLE. It does not")
    print("  diagnose disease and does not predict medical risk.")
    print("=" * 70)


if __name__ == "__main__":
    build()
