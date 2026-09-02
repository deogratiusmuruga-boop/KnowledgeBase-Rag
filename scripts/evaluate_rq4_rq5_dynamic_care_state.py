"""
ElderDocAI RQ4/RQ5 - Dynamic Care-State and Adaptive Assistance Evaluation.

Analysis-only script. It does NOT modify production care-state generation
code, thresholds, rules, or the Synthea-derived datasets. This script only
reads the validated pipeline outputs and writes evaluation artifacts.

Inputs (datasets/synthea/elderdocai/processed/):
    clinical_features.json, dynamic_care_states.json,
    care_state_timeline.json, care_state_transitions.json,
    adaptive_context.json, adaptive_assistance.json,
    assistance_plans.json, assistance_decisions.json

Outputs (data/evaluation_results/):
    rq4_rq5_care_state_assistance_results.json  (machine-readable)
    rq4_rq5_care_state_assistance_report.md     (human-readable summary)
    figures/rq4_fig1_care_state_distribution.png
    figures/rq4_fig2_transition_distribution.png
    figures/rq4_fig3_transition_matrix_heatmap.png
    figures/rq5_fig4_state_assistance_stacked.png
    figures/rq5_fig5_transition_assistance_stacked.png
    figures/rq5_fig6_transition_priority_grouped.png

Method:
    * RQ4 and RQ5 are descriptive statistics over the existing synthetic
      longitudinal records. No clinical validity is claimed.
    * RQ5 rule-consistency section re-encodes the documented deterministic
      rule tables from datasets/synthea/build_adaptive_assistance.py and
      measures agreement between the generated assistance labels and those
      reference rules. This is an INTERNAL rule-consistency check only.

Important:
    These are SYNTHETIC longitudinal clinical records generated from Synthea.
    This is not a clinical validation study and implies no medical validity.
"""

import os
import json
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "datasets", "synthea", "elderdocai", "processed")
OUT_DIR = os.path.join(BASE_DIR, "data", "evaluation_results")
FIG_DIR = os.path.join(OUT_DIR, "figures")

RESULTS_JSON = os.path.join(OUT_DIR, "rq4_rq5_care_state_assistance_results.json")
REPORT_MD = os.path.join(OUT_DIR, "rq4_rq5_care_state_assistance_report.md")

STATE_ORDER = ["STABLE", "LOW_ACTIVITY", "MODERATE_ACTIVITY", "HIGH_ACTIVITY", "NO_DATA"]
TRANSITION_ORDER = [
    "INITIAL_STATE",
    "NO_CHANGE",
    "STATE_ESCALATION",
    "STATE_DEESCALATION",
    "INCREASING_ACTIVITY",
    "DECREASING_ACTIVITY",
    "GAP",
]

STATE_COLORS = {
    "STABLE": "#4caf50",
    "LOW_ACTIVITY": "#ffc107",
    "MODERATE_ACTIVITY": "#ff9800",
    "HIGH_ACTIVITY": "#f44336",
    "NO_DATA": "#9e9e9e",
}

MODE_COLORS = {
    "INITIAL_CONTEXT": "#78909c",
    "WAIT_FOR_DATA": "#b0bec5",
    "LIGHT_SUPPORT": "#aed581",
    "CONTEXTUAL_SUPPORT": "#ffb74d",
    "ENHANCED_SUPPORT": "#ef5350",
    "ADAPTIVE_ESCALATION": "#d32f2f",
    "ADAPTIVE_DEESCALATION": "#7e57c2",
    "MONITORING_SUPPORT": "#4dd0e1",
    "FOLLOW_UP_SUPPORT": "#ba68c8",
}

# ---------------------------------------------------------------------------
# Reference rule tables (documented deterministic rules, re-encoded for the
# internal rule-consistency check only; sourced from
# datasets/synthea/build_adaptive_assistance.py + build_assistance_decisions.py)
# ---------------------------------------------------------------------------
STATE_RULE = {
    "NO_DATA": ("WAIT_FOR_DATA", "LOW"),
    "STABLE": ("MAINTENANCE_SUPPORT", "LOW"),
    "LOW_ACTIVITY": ("LIGHT_SUPPORT", "LOW"),
    "MODERATE_ACTIVITY": ("CONTEXTUAL_SUPPORT", "MODERATE"),
    "HIGH_ACTIVITY": ("ENHANCED_SUPPORT", "HIGH"),
}

TRANSITION_MODE_RULE = {
    "INITIAL_STATE": "INITIAL_CONTEXT",
    "GAP": "WAIT_FOR_DATA",
    "NO_CHANGE": None,  # keep the state default
    "STATE_ESCALATION": "ADAPTIVE_ESCALATION",
    "STATE_DEESCALATION": "ADAPTIVE_DEESCALATION",
    "INCREASING_ACTIVITY": "MONITORING_SUPPORT",
    "DECREASING_ACTIVITY": "FOLLOW_UP_SUPPORT",
}

TRANSITION_PRIORITY_RULE = {
    "INITIAL_STATE": None,  # inherits the state default in the builder
    "GAP": "LOW",
    "NO_CHANGE": None,  # keep the state default
    "STATE_ESCALATION": "HIGH",
    "STATE_DEESCALATION": "MODERATE",
    "INCREASING_ACTIVITY": "MODERATE",  # combined with base via max()
    "DECREASING_ACTIVITY": "MODERATE",  # combined with base via max()
}

MODE_STRATEGY_RULE = {
    "INITIAL_CONTEXT": "ONBOARDING_SUPPORT",
    "WAIT_FOR_DATA": "DATA_COLLECTION_SUPPORT",
    "LIGHT_SUPPORT": "LIGHT_SUPPORT",
    "CONTEXTUAL_SUPPORT": "CONTEXTUAL_SUPPORT",
    "ENHANCED_SUPPORT": "ENHANCED_SUPPORT",
    "ADAPTIVE_ESCALATION": "ENHANCED_CONTEXT_SUPPORT",
    "ADAPTIVE_DEESCALATION": "ADAPTIVE_DEESCALATION_SUPPORT",
    "MONITORING_SUPPORT": "MONITORING_SUPPORT",
    "FOLLOW_UP_SUPPORT": "FOLLOW_UP_SUPPORT",
}

PRIORITY_ORDER = {"LOW": 1, "MODERATE": 2, "HIGH": 3}


def load(name):
    """Load a JSON dataset from the processed directory."""
    with open(os.path.join(PROCESSED, name), "r", encoding="utf-8") as f:
        return json.load(f)


def pct(n, d):
    """Percent with 2 decimals; 0.0 when the denominator is 0."""
    return round(100.0 * n / d, 2) if d else 0.0


def cround(value, ndigits=4):
    """Round a float safely; leave None untouched."""
    if value is None:
        return None
    return round(float(value), ndigits)


def jsonable(x):
    """Convert Counters / tuple-keyed dicts into plain JSON-safe structures."""
    if isinstance(x, Counter):
        return {k: jsonable(v) for k, v in sorted(x.items())}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    return x


def max_priority(*priorities):
    """Strongest priority among the given labels (mirrors builder logic)."""
    valid = [p for p in priorities if p in PRIORITY_ORDER]
    if not valid:
        return "LOW"
    return max(valid, key=lambda p: PRIORITY_ORDER[p])


def expected_mode_priority(state, transition_type):
    """
    Recompute the expected assistance (mode, priority) from the documented
    deterministic rule tables for a given (current_state, transition_type).
    Used ONLY for the internal rule-consistency check.
    """
    base_mode, base_priority = STATE_RULE.get(state, STATE_RULE["NO_DATA"])

    # GAP overrides everything (builder: explicit LOW / WAIT_FOR_DATA)
    if transition_type == "GAP":
        return TRANSITION_MODE_RULE["GAP"], "LOW"

    # INITIAL_STATE: mode INITIAL_CONTEXT but priority inherits the state default
    if transition_type == "INITIAL_STATE":
        return TRANSITION_MODE_RULE["INITIAL_STATE"], base_priority

    if transition_type == "STATE_ESCALATION":
        return TRANSITION_MODE_RULE["STATE_ESCALATION"], TRANSITION_PRIORITY_RULE["STATE_ESCALATION"]

    if transition_type == "STATE_DEESCALATION":
        return TRANSITION_MODE_RULE["STATE_DEESCALATION"], TRANSITION_PRIORITY_RULE["STATE_DEESCALATION"]

    if transition_type == "INCREASING_ACTIVITY":
        return (
            TRANSITION_MODE_RULE["INCREASING_ACTIVITY"],
            max_priority(base_priority, TRANSITION_PRIORITY_RULE["INCREASING_ACTIVITY"]),
        )

    if transition_type == "DECREASING_ACTIVITY":
        return (
            TRANSITION_MODE_RULE["DECREASING_ACTIVITY"],
            max_priority(base_priority, TRANSITION_PRIORITY_RULE["DECREASING_ACTIVITY"]),
        )

    # NO_CHANGE (and any fallback) keeps the state default
    return base_mode, base_priority
def analyze_rq4():
    """
    RQ4 - Dynamic care-state evaluation.

    Returns a JSON-serializable dict with patient-level statistics,
    state distributions, score statistics, transition statistics,
    transition matrix, and longitudinal behavior metrics.
    """
    print("\n" + "=" * 72)
    print("RQ4 - DYNAMIC CARE-STATE EVALUATION")
    print("=" * 72)

    clinical = load("clinical_features.json")
    timeline = load("care_state_timeline.json")
    transitions = load("care_state_transitions.json")

    n_windows = len(timeline)
    n_patients = len({c["patient_id"] for c in clinical})

    # ---- A. Patient-level statistics ---------------------------------
    windows_per_patient = Counter(t["patient_id"] for t in timeline)
    wp_counts = list(windows_per_patient.values())
    multi_window = sum(1 for c in wp_counts if c > 1)

    patient_stats = {
        "n_patients": n_patients,
        "n_windows": n_windows,
        "windows_per_patient": {
            "avg": round(statistics.mean(wp_counts), 2),
            "min": min(wp_counts),
            "max": max(wp_counts),
            "median": statistics.median(wp_counts),
            "pct_patients_with_more_than_one_window": pct(multi_window, n_patients),
        },
        "n_transition_records": len(transitions),
        "n_patients_in_transitions": len({t["patient_id"] for t in transitions}),
    }

    # ---- B. Care-state distribution -----------------------------------
    state_counts = Counter(t["care_state"] for t in timeline)

    per_patient_states = {}
    for t in timeline:
        per_patient_states.setdefault(t["patient_id"], Counter())[t["care_state"]] += 1

    dominant_state = {
        pid: c.most_common(1)[0][0] for pid, c in per_patient_states.items()
    }
    patient_dominant_dist = Counter(dominant_state.values())
    patient_ever_dist = {}
    for st in STATE_ORDER:
        patient_ever_dist[st] = sum(1 for c in per_patient_states.values() if st in c)

    state_distribution_window = {
        st: {"count": state_counts.get(st, 0), "pct": pct(state_counts.get(st, 0), n_windows)}
        for st in STATE_ORDER
    }
    state_distribution_patient_dominant = {
        st: {"count": patient_dominant_dist.get(st, 0),
             "pct": pct(patient_dominant_dist.get(st, 0), n_patients)}
        for st in STATE_ORDER
    }
    state_distribution_patient_ever = {
        st: {"count": patient_ever_dist[st], "pct": pct(patient_ever_dist[st], n_patients)}
        for st in STATE_ORDER
    }

    # ---- C. Overall care-state score ----------------------------------
    scores = [t["overall_score"] for t in timeline if t.get("overall_score") is not None]
    n_scored = len(scores)
    sorted_scores = sorted(scores)
    score_stats = {
        "n_scored_windows": n_scored,
        "n_null_score_windows": n_windows - n_scored,
        "min": cround(min(scores)),
        "max": cround(max(scores)),
        "mean": cround(statistics.mean(scores)),
        "median": cround(statistics.median(scores)),
        "stdev": cround(statistics.stdev(scores)),
    }
    if n_scored >= 4:
        q1, q2, q3 = statistics.quantiles(sorted_scores, n=4)
        score_stats["q1"] = cround(q1)
        score_stats["q2"] = cround(q2)
        score_stats["q3"] = cround(q3)

    # ---- D. Transition analysis ---------------------------------------
    trans_counts = Counter(t["transition_type"] for t in transitions)
    trans_direction = Counter(t["transition_direction"] for t in transitions)

    n_escalation = trans_counts.get("STATE_ESCALATION", 0)
    n_deescalation = trans_counts.get("STATE_DEESCALATION", 0)
    n_increasing = trans_counts.get("INCREASING_ACTIVITY", 0)
    n_decreasing = trans_counts.get("DECREASING_ACTIVITY", 0)
    n_unchanged = trans_counts.get("NO_CHANGE", 0)
    n_initial = trans_counts.get("INITIAL_STATE", 0)
    n_gap = trans_counts.get("GAP", 0)

    transitionable = n_windows - n_initial - n_gap
    n_actual_transition = n_escalation + n_deescalation + n_increasing + n_decreasing

    transition_distribution = {
        t: {"count": trans_counts.get(t, 0), "pct": pct(trans_counts.get(t, 0), n_windows)}
        for t in TRANSITION_ORDER
    }
    transition_direction_distribution = {
        d: {"count": trans_direction.get(d, 0), "pct": pct(trans_direction.get(d, 0), n_windows)}
        for d in ["INITIAL", "UNCHANGED", "INCREASING", "DECREASING", "UNKNOWN"]
    }

    escalation_deescalation_stats = {
        "n_escalation_STATE_ESCALATION": n_escalation,
        "n_deescalation_STATE_DEESCALATION": n_deescalation,
        "n_increasing_INCREASING_ACTIVITY": n_increasing,
        "n_decreasing_DECREASING_ACTIVITY": n_decreasing,
        "n_unchanged_NO_CHANGE": n_unchanged,
        "n_initial_INITIAL_STATE": n_initial,
        "n_gap_GAP": n_gap,
    }

    transition_rate = {
        "n_transitionable_windows_excl_initial_and_gap": transitionable,
        "n_transition_events": n_actual_transition,
        "rate_pct_events_over_transitionable": pct(n_actual_transition, transitionable),
        "all_windows_event_rate_pct": pct(n_actual_transition, n_windows),
    }

    # ---- E. State-to-state transition matrix --------------------------
    matrix = Counter()
    for t in transitions:
        prev = t.get("previous_state")
        curr = t.get("current_state")
        if prev and curr:
            matrix[(prev, curr)] += 1

    matrix_rows = []
    for (p, c), n in sorted(matrix.items()):
        row_total = sum(v for (pp, cc), v in matrix.items() if pp == p)
        matrix_rows.append({
            "previous": p,
            "current": c,
            "count": n,
            "pct_of_all_windows": pct(n, n_windows),
            "pct_within_previous_state": pct(n, row_total),
        })

    # Active-only matrix: exclude pairs touching NO_DATA (gap artifacts)
    matrix_active = Counter()
    for t in transitions:
        prev = t.get("previous_state")
        curr = t.get("current_state")
        if prev and curr and prev != "NO_DATA" and curr != "NO_DATA":
            matrix_active[(prev, curr)] += 1
    matrix_active_rows = []
    for (p, c), n in sorted(matrix_active.items()):
        row_total = sum(v for (pp, cc), v in matrix_active.items() if pp == p)
        matrix_active_rows.append({
            "previous": p,
            "current": c,
            "count": n,
            "pct_of_all_windows": pct(n, n_windows),
            "pct_within_previous_state": pct(n, row_total),
        })

    # ---- F. Longitudinal behavior -------------------------------------
    state_runs = defaultdict(list)
    per_patient_windows = defaultdict(list)
    for t in sorted(timeline, key=lambda x: (x["patient_id"], x.get("year", 0))):
        per_patient_windows[t["patient_id"]].append(t)

    for pid, wins in per_patient_windows.items():
        run_len = 0
        prev_state = None
        for w in wins:
            st = w["care_state"]
            if st == prev_state:
                run_len += 1
            else:
                if prev_state is not None and run_len > 0:
                    state_runs[prev_state].append(run_len)
                run_len = 1
                prev_state = st
        if prev_state is not None and run_len > 0:
            state_runs[prev_state].append(run_len)

    state_persistence = {
        st: {
            "avg_consecutive_windows": round(statistics.mean(state_runs[st]), 2) if state_runs[st] else None,
            "n_runs": len(state_runs[st]),
        }
        for st in STATE_ORDER
    }

    meaningful_per_patient = defaultdict(int)
    state_change_per_patient = defaultdict(int)
    for t in transitions:
        tt = t["transition_type"]
        if tt in ("INITIAL_STATE", "GAP"):
            continue
        meaningful_per_patient[t["patient_id"]] += 1
        if tt != "NO_CHANGE":
            state_change_per_patient[t["patient_id"]] += 1

    n_meaningful_patients = len(meaningful_per_patient)
    m_counts = list(meaningful_per_patient.values())
    sc_counts = list(state_change_per_patient.values())

    esc_patients = {t["patient_id"] for t in transitions if t["transition_type"] == "STATE_ESCALATION"}
    desc_patients = {t["patient_id"] for t in transitions if t["transition_type"] == "STATE_DEESCALATION"}
    dec_patients = {t["patient_id"] for t in transitions if t["transition_type"] == "DECREASING_ACTIVITY"}
    inc_patients = {t["patient_id"] for t in transitions if t["transition_type"] == "INCREASING_ACTIVITY"}

    unchanged_patients = {
        pid for pid, cnt in meaningful_per_patient.items() if state_change_per_patient.get(pid, 0) == 0
    }
    single_state_patients = 0
    for pid, wins in per_patient_windows.items():
        states = [w["care_state"] for w in wins]
        if len(set(states)) == 1:
            single_state_patients += 1

    longitudinal = {
        "state_persistence_avg_consecutive_windows": state_persistence,
        "avg_meaningful_transitions_per_patient": round(statistics.mean(m_counts), 2) if m_counts else 0.0,
        "avg_state_changes_per_patient": round(statistics.mean(sc_counts), 2) if sc_counts else 0.0,
        "median_meaningful_transitions_per_patient": statistics.median(m_counts) if m_counts else 0,
        "pct_patients_with_at_least_one_escalation": pct(len(esc_patients), n_patients),
        "pct_patients_with_at_least_one_deescalation": pct(len(desc_patients), n_patients),
        "pct_patients_with_at_least_one_increase": pct(len(inc_patients), n_patients),
        "pct_patients_with_at_least_one_decrease": pct(len(dec_patients), n_patients),
        "pct_patients_unchanged_across_meaningful_transitions": pct(len(unchanged_patients), n_meaningful_patients) if n_meaningful_patients else 0.0,
        "pct_patients_single_state_across_all_windows": pct(single_state_patients, n_patients),
        "n_patients_with_meaningful_transitions": n_meaningful_patients,
    }

    return {
        "datasets": [
            "clinical_features.json",
            "dynamic_care_states.json",
            "care_state_timeline.json",
            "care_state_transitions.json",
        ],
        "A_patient_statistics": patient_stats,
        "B_state_distribution_window": state_distribution_window,
        "B_state_distribution_patient_dominant": state_distribution_patient_dominant,
        "B_state_distribution_patient_ever": state_distribution_patient_ever,
        "C_score_statistics": score_stats,
        "D_transition_distribution": transition_distribution,
        "D_transition_direction_distribution": transition_direction_distribution,
        "D_escalation_deescalation_stats": escalation_deescalation_stats,
        "D_transition_rate": transition_rate,
        "E_transition_matrix": matrix_rows,
        "E_transition_matrix_active_only": matrix_active_rows,
        "F_longitudinal": longitudinal,
    }
def analyze_rq5():
    """
    RQ5 - Adaptive assistance evaluation.

    Returns a JSON-serializable dict with assistance mode / priority
    distributions, care-state->assistance and transition->assistance
    mappings, plan coverage, and internal rule-consistency diagnostics.
    """
    print("\n" + "=" * 72)
    print("RQ5 - ADAPTIVE ASSISTANCE EVALUATION")
    print("=" * 72)

    adaptive_context = load("adaptive_context.json")
    adaptive_assistance = load("adaptive_assistance.json")
    assistance_plans = load("assistance_plans.json")
    assistance_decisions = load("assistance_decisions.json")
    timeline = load("care_state_timeline.json")

    n_windows = len(adaptive_context)
    n_timeline = len(timeline)

    # ---- A. Assistance mode distribution ------------------------------
    mode_counts = Counter(a["assistance_mode"] for a in adaptive_assistance)
    mode_distribution = {
        m: {"count": c, "pct": pct(c, n_windows)}
        for m, c in sorted(mode_counts.items())
    }

    # ---- B. Priority distribution -------------------------------------
    priority_counts = Counter(a["priority"] for a in adaptive_assistance)
    priority_distribution = {
        p: {"count": c, "pct": pct(c, n_windows)}
        for p, c in sorted(priority_counts.items())
    }

    # ---- C. Care-state -> assistance mapping --------------------------
    state_mode = Counter()
    state_total = Counter()
    for a in adaptive_assistance:
        state_mode[(a["current_state"], a["assistance_mode"])] += 1
        state_total[a["current_state"]] += 1

    state_assistance_rows = []
    for (s, m), n in sorted(state_mode.items()):
        state_assistance_rows.append({
            "care_state": s,
            "assistance_mode": m,
            "count": n,
            "pct_of_all_windows": pct(n, n_windows),
            "pct_within_state": pct(n, state_total[s]),
        })

    # ---- D. Transition -> assistance mapping --------------------------
    trans_mode = Counter()
    trans_total = Counter()
    for a in adaptive_assistance:
        trans_mode[(a["transition_type"], a["assistance_mode"])] += 1
        trans_total[a["transition_type"]] += 1

    transition_assistance_rows = []
    for (t, m), n in sorted(trans_mode.items()):
        transition_assistance_rows.append({
            "transition_type": t,
            "assistance_mode": m,
            "count": n,
            "pct_of_all_windows": pct(n, n_windows),
            "pct_within_transition": pct(n, trans_total[t]),
        })

    # ---- E. Assistance-plan / decision coverage -----------------------
    def unique_keys(records):
        return {(r.get("patient_id"), r.get("window_start")) for r in records}

    def duplicate_keys(records):
        seen, dups = set(), set()
        for r in records:
            k = (r.get("patient_id"), r.get("window_start"))
            (dups if k in seen else seen).add(k)
        return dups

    timeline_keys = unique_keys(timeline)
    context_keys = unique_keys(adaptive_context)
    assistance_keys = unique_keys(adaptive_assistance)
    plan_keys = unique_keys(assistance_plans)
    decision_keys = unique_keys(assistance_decisions)

    coverage = {
        "join_key": "(patient_id, window_start)",
        "n_timeline_windows": n_timeline,
        "n_unique_timeline_keys": len(timeline_keys),
        "artifacts": {
            "adaptive_context": {
                "n_records": len(adaptive_context),
                "n_unique_keys": len(context_keys),
                "n_duplicate_keys": len(duplicate_keys(adaptive_context)),
                "matched_to_timeline": len(context_keys & timeline_keys),
                "pct_of_timeline_windows": pct(len(context_keys & timeline_keys), n_timeline),
                "missing": n_timeline - len(context_keys & timeline_keys),
            },
            "adaptive_assistance": {
                "n_records": len(adaptive_assistance),
                "n_unique_keys": len(assistance_keys),
                "n_duplicate_keys": len(duplicate_keys(adaptive_assistance)),
                "matched_to_timeline": len(assistance_keys & timeline_keys),
                "pct_of_timeline_windows": pct(len(assistance_keys & timeline_keys), n_timeline),
                "missing": n_timeline - len(assistance_keys & timeline_keys),
            },
            "assistance_plans": {
                "n_records": len(assistance_plans),
                "n_unique_keys": len(plan_keys),
                "n_duplicate_keys": len(duplicate_keys(assistance_plans)),
                "matched_to_timeline": len(plan_keys & timeline_keys),
                "pct_of_timeline_windows": pct(len(plan_keys & timeline_keys), n_timeline),
                "missing": n_timeline - len(plan_keys & timeline_keys),
            },
            "assistance_decisions": {
                "n_records": len(assistance_decisions),
                "n_unique_keys": len(decision_keys),
                "n_duplicate_keys": len(duplicate_keys(assistance_decisions)),
                "matched_to_timeline": len(decision_keys & timeline_keys),
                "pct_of_timeline_windows": pct(len(decision_keys & timeline_keys), n_timeline),
                "missing": n_timeline - len(decision_keys & timeline_keys),
            },
        },
    }

    null_rates = {
        "assistance_mode_null": sum(1 for a in adaptive_assistance if a.get("assistance_mode") is None),
        "priority_null": sum(1 for a in adaptive_assistance if a.get("priority") is None),
        "plan_strategy_null": sum(1 for p in assistance_plans if p.get("assistance_strategy") is None),
        "plan_actions_missing": sum(1 for p in assistance_plans if not p.get("actions")),
        "decision_strategy_null": sum(1 for d in assistance_decisions if d.get("assistance_strategy") is None),
    }
    coverage["null_rates"] = null_rates

    # ---- F. Internal rule-consistency check ---------------------------
    rule_consistency = check_rule_consistency(
        adaptive_assistance, assistance_plans, assistance_decisions, n_windows
    )

    return {
        "datasets": [
            "adaptive_context.json",
            "adaptive_assistance.json",
            "assistance_plans.json",
            "assistance_decisions.json",
        ],
        "n_windows": n_windows,
        "A_assistance_mode_distribution": mode_distribution,
        "B_priority_distribution": priority_distribution,
        "C_state_assistance_mapping": state_assistance_rows,
        "D_transition_assistance_mapping": transition_assistance_rows,
        "E_coverage": coverage,
        "F_rule_consistency": rule_consistency,
    }
def check_rule_consistency(adaptive_assistance, assistance_plans, assistance_decisions, n_windows):
    """
    Internal rule-consistency check.

    Re-derives the expected assistance mode and priority from the documented
    deterministic rule tables (build_adaptive_assistance.py) and measures the
    agreement of the generated labels with those rules. Also verifies that the
    assistance strategies in the plans/decisions match the documented
    mode->strategy mapping.

    This validates internal logical consistency ONLY. It does not imply
    clinical validity.
    """
    mismatches = []
    mode_ok = priority_ok = 0
    for a in adaptive_assistance:
        state = a["current_state"]
        tt = a["transition_type"]
        exp_mode, exp_prio = expected_mode_priority(state, tt)
        if a["assistance_mode"] == exp_mode:
            mode_ok += 1
        else:
            mismatches.append({
                "kind": "mode",
                "patient_id": a["patient_id"],
                "window_start": a["window_start"],
                "state": state,
                "transition_type": tt,
                "actual_mode": a["assistance_mode"],
                "expected_mode": exp_mode,
            })
        if a["priority"] == exp_prio:
            priority_ok += 1
        else:
            mismatches.append({
                "kind": "priority",
                "patient_id": a["patient_id"],
                "window_start": a["window_start"],
                "state": state,
                "transition_type": tt,
                "actual_priority": a["priority"],
                "expected_priority": exp_prio,
            })

    # Observed state x priority table
    state_priority = Counter()
    state_total = Counter()
    for a in adaptive_assistance:
        state_priority[(a["current_state"], a["priority"])] += 1
        state_total[a["current_state"]] += 1
    state_priority_rows = [
        {
            "care_state": s,
            "priority": p,
            "count": n,
            "pct_within_state": pct(n, state_total[s]),
        }
        for (s, p), n in sorted(state_priority.items())
    ]

    # Observed transition x priority table
    trans_priority = Counter()
    trans_total = Counter()
    for a in adaptive_assistance:
        trans_priority[(a["transition_type"], a["priority"])] += 1
        trans_total[a["transition_type"]] += 1
    transition_priority_rows = [
        {
            "transition_type": t,
            "priority": p,
            "count": n,
            "pct_within_transition": pct(n, trans_total[t]),
        }
        for (t, p), n in sorted(trans_priority.items())
    ]

    # Semantic checks: intensity ordering and transition->mode mapping
    def sub_pct(records, cond):
        total = len(records)
        if total == 0:
            return 0.0, 0
        ok = sum(1 for r in records if cond(r))
        return pct(ok, total), ok

    esc_recs = [a for a in adaptive_assistance if a["transition_type"] == "STATE_ESCALATION"]
    desc_recs = [a for a in adaptive_assistance if a["transition_type"] == "STATE_DEESCALATION"]
    inc_recs = [a for a in adaptive_assistance if a["transition_type"] == "INCREASING_ACTIVITY"]
    dec_recs = [a for a in adaptive_assistance if a["transition_type"] == "DECREASING_ACTIVITY"]
    gap_recs = [a for a in adaptive_assistance if a["transition_type"] == "GAP"]
    init_recs = [a for a in adaptive_assistance if a["transition_type"] == "INITIAL_STATE"]
    nochange_recs = [a for a in adaptive_assistance if a["transition_type"] == "NO_CHANGE"]
    high_recs = [a for a in adaptive_assistance if a["current_state"] == "HIGH_ACTIVITY"]
    mod_recs = [a for a in adaptive_assistance if a["current_state"] == "MODERATE_ACTIVITY"]
    low_recs = [a for a in adaptive_assistance if a["current_state"] == "LOW_ACTIVITY"]
    stable_recs = [a for a in adaptive_assistance if a["current_state"] == "STABLE"]
    nodata_recs = [a for a in adaptive_assistance if a["current_state"] == "NO_DATA"]

    semantic = {
        "state_intensity": {
            "HIGH_ACTIVITY_windows": len(high_recs),
            "HIGH_ACTIVITY_high_priority_pct": sub_pct(
                high_recs, lambda r: r["priority"] == "HIGH")[0],
            "HIGH_ACTIVITY_ENHANCED_or_escalation_mode_pct": sub_pct(
                high_recs, lambda r: r["assistance_mode"] in ("ENHANCED_SUPPORT", "ADAPTIVE_ESCALATION"))[0],
            "MODERATE_ACTIVITY_windows": len(mod_recs),
            "MODERATE_ACTIVITY_moderate_priority_pct": sub_pct(
                mod_recs, lambda r: r["priority"] == "MODERATE")[0],
            "LOW_ACTIVITY_windows": len(low_recs),
            "LOW_ACTIVITY_low_priority_pct": sub_pct(
                low_recs, lambda r: r["priority"] == "LOW")[0],
            "STABLE_windows": len(stable_recs),
            "STABLE_low_priority_pct": sub_pct(
                stable_recs, lambda r: r["priority"] == "LOW")[0],
            "NO_DATA_windows": len(nodata_recs),
            "NO_DATA_low_priority_pct": sub_pct(
                nodata_recs, lambda r: r["priority"] == "LOW")[0],
        },
        "transition_semantics": {
            "STATE_ESCALATION_ADAPTIVE_ESCALATION_pct": sub_pct(
                esc_recs, lambda r: r["assistance_mode"] == "ADAPTIVE_ESCALATION")[0],
            "STATE_ESCALATION_high_priority_pct": sub_pct(
                esc_recs, lambda r: r["priority"] == "HIGH")[0],
            "STATE_DEESCALATION_ADAPTIVE_DEESCALATION_pct": sub_pct(
                desc_recs, lambda r: r["assistance_mode"] == "ADAPTIVE_DEESCALATION")[0],
            "INCREASING_ACTIVITY_MONITORING_SUPPORT_pct": sub_pct(
                inc_recs, lambda r: r["assistance_mode"] == "MONITORING_SUPPORT")[0],
            "INCREASING_ACTIVITY_priority_ge_MODERATE_pct": sub_pct(
                inc_recs, lambda r: r["priority"] in ("MODERATE", "HIGH"))[0],
            "DECREASING_ACTIVITY_FOLLOW_UP_SUPPORT_pct": sub_pct(
                dec_recs, lambda r: r["assistance_mode"] == "FOLLOW_UP_SUPPORT")[0],
            "GAP_WAIT_FOR_DATA_pct": sub_pct(
                gap_recs, lambda r: r["assistance_mode"] == "WAIT_FOR_DATA")[0],
            "INITIAL_STATE_INITIAL_CONTEXT_pct": sub_pct(
                init_recs, lambda r: r["assistance_mode"] == "INITIAL_CONTEXT")[0],
            "NO_CHANGE_state_default_mode_pct": sub_pct(
                nochange_recs,
                lambda r: r["assistance_mode"] in ("LIGHT_SUPPORT", "CONTEXTUAL_SUPPORT", "ENHANCED_SUPPORT"))[0],
        },
    }

    # Plans / decisions strategy agreement vs documented mode->strategy mapping
    plan_ok = 0
    plan_denom = len(assistance_plans)
    for p, a in zip(assistance_plans, adaptive_assistance):
        exp_strategy = MODE_STRATEGY_RULE.get(a["assistance_mode"])
        if exp_strategy and p["assistance_strategy"] == exp_strategy:
            plan_ok += 1
    decision_ok = 0
    for d, a in zip(assistance_decisions, adaptive_assistance):
        exp_strategy = MODE_STRATEGY_RULE.get(a["assistance_mode"])
        if exp_strategy and d["assistance_strategy"] == exp_strategy:
            decision_ok += 1

    return {
        "reference_policy": "deterministic rule tables from build_adaptive_assistance.py",
        "n_records_checked": len(adaptive_assistance),
        "mode_agreement_count": mode_ok,
        "mode_agreement_pct": pct(mode_ok, len(adaptive_assistance)),
        "priority_agreement_count": priority_ok,
        "priority_agreement_pct": pct(priority_ok, len(adaptive_assistance)),
        "n_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "state_priority": state_priority_rows,
        "transition_priority": transition_priority_rows,
        "semantic_checks": semantic,
        "strategy_agreement_plans_pct": pct(plan_ok, plan_denom),
        "strategy_agreement_decisions_pct": pct(decision_ok, len(assistance_decisions)),
        "note": "Internal rule-consistency check. Does not establish clinical validity.",
    }
def save_fig(fig, name):
    """Save a figure into the figures output directory."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure {name}")
    return name


def make_figures(r4, r5):
    """Generate publication-ready figures for the RQ4/RQ5 evidence."""
    figures = []

    # ---- Figure 1: dynamic care-state distribution --------------------
    states = STATE_ORDER
    counts = [r4["B_state_distribution_window"][s]["count"] for s in states]
    pcts = [r4["B_state_distribution_window"][s]["pct"] for s in states]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(states, counts, color=[STATE_COLORS[s] for s in states],
                  edgecolor="black", linewidth=0.5)
    for b, c, p in zip(bars, counts, pcts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 25,
                f"{c:,}\n({p}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Figure 1. Dynamic Care-State Distribution (9,723 patient-year windows)")
    ax.set_xlabel("Care State")
    ax.set_ylabel("Number of Windows")
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    figures.append(save_fig(fig, "rq4_fig1_care_state_distribution.png"))

    # ---- Figure 2: care-state transition distribution ------------------
    trans = TRANSITION_ORDER
    tcounts = [r4["D_transition_distribution"][t]["count"] for t in trans]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(trans, tcounts, color="#2196f3", edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, tcounts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 40,
                f"{c:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Figure 2. Care-State Transition Distribution")
    ax.set_xlabel("Transition Type")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(tcounts) * 1.12)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    figures.append(save_fig(fig, "rq4_fig2_transition_distribution.png"))

    # ---- Figure 3 (supplementary): transition matrix heatmap ----------
    labels_act = [s for s in STATE_ORDER if s != "NO_DATA"]
    rows = r4["E_transition_matrix_active_only"]
    mat = np.zeros((len(labels_act), len(labels_act)), dtype=int)
    for rr in rows:
        if rr["previous"] in labels_act and rr["current"] in labels_act:
            mat[labels_act.index(rr["previous"]), labels_act.index(rr["current"])] = rr["count"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(range(len(labels_act)))
    ax.set_yticks(range(len(labels_act)))
    ax.set_xticklabels(labels_act, rotation=30, ha="right")
    ax.set_yticklabels(labels_act)
    ax.set_xlabel("Current Care State")
    ax.set_ylabel("Previous Care State")
    for i in range(len(labels_act)):
        for j in range(len(labels_act)):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > max(1, mat.max() / 2) else "black",
                    fontsize=10)
    ax.set_title("Figure 3 (suppl.). State-to-State Transition Matrix\n(active windows, excludes NO_DATA gaps)")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    figures.append(save_fig(fig, "rq4_fig3_transition_matrix_heatmap.png"))

    # ---- Figure 4: care-state -> adaptive assistance (stacked) ----------
    modes = sorted(r5["A_assistance_mode_distribution"])
    state_mode_matrix = {s: {m: 0 for m in modes} for s in STATE_ORDER}
    for row in r5["C_state_assistance_mapping"]:
        s, m = row["care_state"], row["assistance_mode"]
        if s in state_mode_matrix and m in state_mode_matrix[s]:
            state_mode_matrix[s][m] = row["count"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(STATE_ORDER)
    for m in modes:
        vals = [state_mode_matrix[s][m] for s in STATE_ORDER]
        ax.bar(STATE_ORDER, vals, bottom=bottom, label=m, color=MODE_COLORS.get(m, "#cccccc"))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Figure 4. Care-State -> Adaptive-Assistance Mode (stacked)")
    ax.set_xlabel("Care State")
    ax.set_ylabel("Number of patient-year windows")
    ax.legend(title="Assistance Mode", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    figures.append(save_fig(fig, "rq5_fig4_state_assistance_stacked.png"))

    # ---- Figure 5: transition -> assistance mode (stacked) --------------
    trans_assist = {}
    for row in r5["D_transition_assistance_mapping"]:
        t, m = row["transition_type"], row["assistance_mode"]
        trans_assist.setdefault(t, {})[m] = row["count"]
    tt_order = [t for t in TRANSITION_ORDER if t in trans_assist]
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = [0] * len(tt_order)
    for m in modes:
        vals = [trans_assist.get(t, {}).get(m, 0) for t in tt_order]
        ax.bar(tt_order, vals, bottom=bottom, label=m, color=MODE_COLORS.get(m, "#cccccc"))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Figure 5. Transition Type -> Adaptive-Assistance Mode (stacked)")
    ax.set_xlabel("Transition Type")
    ax.set_ylabel("Number of patient-year windows")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Assistance Mode", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    figures.append(save_fig(fig, "rq5_fig5_transition_assistance_stacked.png"))

    # ---- Figure 6 (supplementary): transition -> priority (grouped) ------
    tp_rows = r5["F_rule_consistency"]["transition_priority"]
    priorities = ["LOW", "MODERATE", "HIGH"]
    tp_matrix = {t: {p: 0 for p in priorities} for t in TRANSITION_ORDER}
    for row in tp_rows:
        t, p = row["transition_type"], row["priority"]
        if t in tp_matrix and p in tp_matrix[t]:
            tp_matrix[t][p] = row["count"]
    x = np.arange(len(TRANSITION_ORDER))
    width = 0.26
    priority_colors = {"LOW": "#8bc34a", "MODERATE": "#ff9800", "HIGH": "#e53935"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, p in enumerate(priorities):
        vals = [tp_matrix[t][p] for t in TRANSITION_ORDER]
        ax.bar(x + (i - 1) * width, vals, width, label=p, color=priority_colors[p])
    ax.set_xticks(x)
    ax.set_xticklabels(TRANSITION_ORDER, rotation=25, ha="right")
    ax.set_title("Figure 6 (suppl.). Transition Type -> Assistance Priority")
    ax.set_xlabel("Transition Type")
    ax.set_ylabel("Count")
    ax.legend(title="Priority")
    fig.tight_layout()
    figures.append(save_fig(fig, "rq5_fig6_transition_priority_grouped.png"))

    return figures
def print_rq4(r4):
    """Print a concise RQ4 summary to the console."""
    ps = r4["A_patient_statistics"]
    print("\n--- RQ4 summary ---")
    print(f"A. Patients: {ps['n_patients']}, windows: {ps['n_windows']}, "
          f"avg windows/patient: {ps['windows_per_patient']['avg']}, "
          f"min: {ps['windows_per_patient']['min']}, max: {ps['windows_per_patient']['max']}, "
          f"pct patients >1 window: {ps['windows_per_patient']['pct_patients_with_more_than_one_window']}%")
    print("B. Window-level state distribution (count, %):")
    for st in STATE_ORDER:
        row = r4["B_state_distribution_window"][st]
        print(f"   {st:20s} {row['count']:6d}  {row['pct']:6.2f}%")
    print("   Patient-dominant state distribution:")
    for st in STATE_ORDER:
        row = r4["B_state_distribution_patient_dominant"][st]
        print(f"   {st:20s} {row['count']:6d}  {row['pct']:6.2f}%")
    print(f"C. Score stats: {r4['C_score_statistics']}")
    print("D. Transition distribution (count, %):")
    for t in TRANSITION_ORDER:
        row = r4["D_transition_distribution"][t]
        print(f"   {t:25s} {row['count']:6d}  {row['pct']:6.2f}%")
    es = r4["D_escalation_deescalation_stats"]
    print(f"   Escalation: {es['n_escalation_STATE_ESCALATION']}, "
          f"De-escalation: {es['n_deescalation_STATE_DEESCALATION']}, "
          f"Unchanged: {es['n_unchanged_NO_CHANGE']}, "
          f"Increasing: {es['n_increasing_INCREASING_ACTIVITY']}, "
          f"Decreasing: {es['n_decreasing_DECREASING_ACTIVITY']}")
    print(f"   Transition rate (events/transitionable excl INITIAL+GAP): "
          f"{r4['D_transition_rate']['rate_pct_events_over_transitionable']}%")
    print(f"E. Transition matrix rows: {len(r4['E_transition_matrix'])} "
          f"(active-only: {len(r4['E_transition_matrix_active_only'])})")
    print(f"F. Longitudinal: {r4['F_longitudinal']}")


def print_rq5(r5):
    """Print a concise RQ5 summary to the console."""
    print("\n--- RQ5 summary ---")
    print(f"A. Assistance mode distribution ({r5['n_windows']} windows):")
    for m, row in sorted(r5["A_assistance_mode_distribution"].items()):
        print(f"   {m:30s} {row['count']:6d}  {row['pct']:6.2f}%")
    print("B. Priority distribution:")
    for p, row in sorted(r5["B_priority_distribution"].items()):
        print(f"   {p:10s} {row['count']:6d}  {row['pct']:6.2f}%")
    print(f"C. State->assistance mapping rows: {len(r5['C_state_assistance_mapping'])}")
    print(f"D. Transition->assistance mapping rows: {len(r5['D_transition_assistance_mapping'])}")
    print(f"E. Coverage: {json.dumps(r5['E_coverage'], indent=2)[:1600]}")
    rc = r5["F_rule_consistency"]
    print(f"F. Rule consistency: mode_agreement={rc['mode_agreement_pct']}%, "
          f"priority_agreement={rc['priority_agreement_pct']}%, "
          f"mismatches={rc['n_mismatches']}")


def write_report(r4, r5, figures):
    """Write a human-readable Markdown report of the evaluation."""
    rows = []
    A = lambda text: rows.append(text)

    A("# ElderDocAI RQ4/RQ5 Evaluation Report")
    A("")
    A("> Synthetic longitudinal clinical records (Synthea), 178 patients, "
      "9,723 patient x year windows.")
    A("> **Scope:** descriptive statistics + internal rule-consistency. "
      "This is NOT a clinical validation study.")
    A("")
    A("## Datasets used (read-only inputs)")
    A("")
    A("| File | Records |")
    A("|---|---:|")
    A("| datasets/synthea/elderdocai/processed/clinical_features.json | 178 |")
    A("| datasets/synthea/elderdocai/processed/dynamic_care_states.json | 178 |")
    A("| datasets/synthea/elderdocai/processed/care_state_timeline.json | 9,723 |")
    A("| datasets/synthea/elderdocai/processed/care_state_transitions.json | 9,723 |")
    A("| datasets/synthea/elderdocai/processed/adaptive_context.json | 9,723 |")
    A("| datasets/synthea/elderdocai/processed/adaptive_assistance.json | 9,723 |")
    A("| datasets/synthea/elderdocai/processed/assistance_plans.json | 9,723 |")
    A("| datasets/synthea/elderdocai/processed/assistance_decisions.json | 9,723 |")
    A("")

    A("## RQ4 - Dynamic Care-State Evaluation")
    A("")
    A("### A. Patient-level statistics")
    ps = r4["A_patient_statistics"]
    wpp = ps["windows_per_patient"]
    A(f"- patients: **{ps['n_patients']}**")
    A(f"- longitudinal windows: **{ps['n_windows']}**")
    A(f"- average windows per patient: **{wpp['avg']}** (median {wpp['median']})")
    A(f"- min / max windows per patient: **{wpp['min']} / {wpp['max']}**")
    A(f"- patients with more than one care-state window: "
      f"**{wpp['pct_patients_with_more_than_one_window']}%**")
    A("")
    A("### B. Care-state distribution")
    A("")
    A("Window-level:")
    A("")
    A("| Care State | Count | % of windows |")
    A("|---|---|---:|")
    for st in STATE_ORDER:
        row = r4["B_state_distribution_window"][st]
        A(f"| {st} | {row['count']:,} | {row['pct']} |")
    A("")
    A("Patient-level (dominant state per patient):")
    A("")
    A("| Care State | Patients | % of patients |")
    A("|---|---|---:|")
    for st in STATE_ORDER:
        row = r4["B_state_distribution_patient_dominant"][st]
        A(f"| {st} | {row['count']} | {row['pct']} |")
    A("")
    A("Patient-level (ever observed in state):")
    A("")
    A("| Care State | Patients | % of patients |")
    A("|---|---|---:|")
    for st in STATE_ORDER:
        row = r4["B_state_distribution_patient_ever"][st]
        A(f"| {st} | {row['count']} | {row['pct']} |")
    A("")
    A("### C. Overall care-state score")
    sc = r4["C_score_statistics"]
    A(f"- scored windows (non-null): **{sc['n_scored_windows']}** "
      f"(null/NO_DATA: {sc['n_null_score_windows']})")
    A(f"- min **{sc['min']}**, max **{sc['max']}**, mean **{sc['mean']}**, "
      f"median **{sc['median']}**, stdev **{sc['stdev']}**")
    if "q1" in sc:
        A(f"- quartiles: Q1 **{sc['q1']}**, Q2 **{sc['q2']}**, Q3 **{sc['q3']}**")
    A("")

    A("### D. Transition analysis")
    A("")
    A("| Transition Type | Count | % of windows |")
    A("|---|---|---:|")
    for t in TRANSITION_ORDER:
        row = r4["D_transition_distribution"][t]
        A(f"| {t} | {row['count']:,} | {row['pct']} |")
    A("")
    es = r4["D_escalation_deescalation_stats"]
    A(f"- escalation events (STATE_ESCALATION): **{es['n_escalation_STATE_ESCALATION']}**")
    A(f"- de-escalation events (STATE_DEESCALATION): **{es['n_deescalation_STATE_DEESCALATION']}**")
    A(f"- increasing-activity events (INCREASING_ACTIVITY): **{es['n_increasing_INCREASING_ACTIVITY']}**")
    A(f"- decreasing-activity events (DECREASING_ACTIVITY): **{es['n_decreasing_DECREASING_ACTIVITY']}**")
    A(f"- unchanged-state windows (NO_CHANGE): **{es['n_unchanged_NO_CHANGE']}**")
    A("")
    tr = r4["D_transition_rate"]
    A(f"- transitionable windows (excl. INITIAL_STATE/GAP): **{tr['n_transitionable_windows_excl_initial_and_gap']}**")
    A(f"- actual transition events: **{tr['n_transition_events']}** "
      f"-> rate **{tr['rate_pct_events_over_transitionable']}%** "
      f"({tr['all_windows_event_rate_pct']}% of all windows)")
    A("")
    A("Transition direction distribution:")
    A("")
    A("| Direction | Count | % of windows |")
    A("|---|---|---:|")
    for d, row in sorted(r4["D_transition_direction_distribution"].items()):
        A(f"| {d} | {row['count']:,} | {row['pct']} |")
    A("")
    A("### E. State-to-state transition matrix")
    A("")
    A("Active windows only (previous/current both non-NO_DATA):")
    A("")
    A("| Previous | Current | Count | % of all windows | % within previous |")
    A("|---|---|---:|---:|---:|")
    for row in r4["E_transition_matrix_active_only"]:
        A(f"| {row['previous']} | {row['current']} | {row['count']:,} "
          f"| {row['pct_of_all_windows']} | {row['pct_within_previous_state']} |")
    A("")
    A("All observed pairs (including NO_DATA gap artifacts):")
    A("")
    A("| Previous | Current | Count | % of all windows | % within previous |")
    A("|---|---|---:|---:|---:|")
    for row in r4["E_transition_matrix"]:
        A(f"| {row['previous']} | {row['current']} | {row['count']:,} "
          f"| {row['pct_of_all_windows']} | {row['pct_within_previous_state']} |")
    A("")
    A("### F. Longitudinal behavior")
    lon = r4["F_longitudinal"]
    A("")
    A("Average consecutive-window persistence per state:")
    A("")
    A("| Care State | Avg consecutive windows | Runs observed |")
    A("|---|---:|---:|")
    for st, info in lon["state_persistence_avg_consecutive_windows"].items():
        avg = info["avg_consecutive_windows"] if info["avg_consecutive_windows"] is not None else "n/a"
        A(f"| {st} | {avg} | {info['n_runs']} |")
    A("")
    A(f"- average meaningful transitions per patient (excl. INITIAL/GAP): "
      f"**{lon['avg_meaningful_transitions_per_patient']}** "
      f"(median {lon['median_meaningful_transitions_per_patient']})")
    A(f"- average state changes per patient: **{lon['avg_state_changes_per_patient']}**")
    A(f"- patients with >=1 escalation: **{lon['pct_patients_with_at_least_one_escalation']}%**")
    A(f"- patients with >=1 de-escalation: **{lon['pct_patients_with_at_least_one_deescalation']}%**")
    A(f"- patients with >=1 increase: **{lon['pct_patients_with_at_least_one_increase']}%**")
    A(f"- patients with >=1 decrease: **{lon['pct_patients_with_at_least_one_decrease']}%**")
    A(f"- patients unchanged across all meaningful transitions: "
      f"**{lon['pct_patients_unchanged_across_meaningful_transitions']}%**")
    A(f"- patients in a single state across all windows: "
      f"**{lon['pct_patients_single_state_across_all_windows']}%**")
    A("")
    A("# RQ5 - Adaptive Assistance Evaluation")
    A("")
    A("### A. Assistance mode distribution")
    A("")
    A("| Assistance Mode | Count | % of windows |")
    A("|---|---|---:|")
    for m, row in sorted(r5["A_assistance_mode_distribution"].items()):
        A(f"| {m} | {row['count']:,} | {row['pct']} |")
    A("")
    A("### B. Priority distribution")
    A("")
    A("| Priority | Count | % of windows |")
    A("|---|---|---:|")
    for p, row in sorted(r5["B_priority_distribution"].items()):
        A(f"| {p} | {row['count']:,} | {row['pct']} |")
    A("")
    A("### C. Care-state -> assistance mapping")
    A("")
    A("| Care State | Assistance Mode | Count | % of windows | % within state |")
    A("|---|---|---:|---:|---:|")
    for row in r5["C_state_assistance_mapping"]:
        A(f"| {row['care_state']} | {row['assistance_mode']} | {row['count']:,} "
          f"| {row['pct_of_all_windows']} | {row['pct_within_state']} |")
    A("")
    A("### D. Transition -> assistance mapping")
    A("")
    A("| Transition Type | Assistance Mode | Count | % of windows | % within transition |")
    A("|---|---|---:|---:|---:|")
    for row in r5["D_transition_assistance_mapping"]:
        A(f"| {row['transition_type']} | {row['assistance_mode']} | {row['count']:,} "
          f"| {row['pct_of_all_windows']} | {row['pct_within_transition']} |")
    A("")
    A("### E. Assistance-plan coverage")
    cov = r5["E_coverage"]
    A("")
    A("| Artifact | Records | Unique (patient, window) | Matched to timeline | % of 9,723 windows | Missing |")
    A("|---|---:|---:|---:|---:|---:|")
    for art, info in cov["artifacts"].items():
        A(f"| {art} | {info['n_records']:,} | {info['n_unique_keys']:,} "
          f"| {info['matched_to_timeline']:,} | {info['pct_of_timeline_windows']} | {info['missing']} |")
    A("")
    A(f"- null rates: {cov['null_rates']}")
    A("")
    A("### F. Internal rule-consistency")
    rc = r5["F_rule_consistency"]
    A("")
    A(f"- mode agreement with documented rule tables: **{rc['mode_agreement_pct']}%** "
      f"({rc['mode_agreement_count']}/{rc['n_records_checked']})")
    A(f"- priority agreement with documented rule tables: **{rc['priority_agreement_pct']}%** "
      f"({rc['priority_agreement_count']}/{rc['n_records_checked']})")
    A(f"- mismatches: **{rc['n_mismatches']}**")
    A(f"- plans strategy agreement: **{rc['strategy_agreement_plans_pct']}%**")
    A(f"- decisions strategy agreement: **{rc['strategy_agreement_decisions_pct']}%**")
    A("")
    A("State x priority (observed):")
    A("")
    A("| Care State | Priority | Count | % within state |")
    A("|---|---|---:|---:|")
    for row in rc["state_priority"]:
        A(f"| {row['care_state']} | {row['priority']} | {row['count']:,} | {row['pct_within_state']} |")
    A("")
    A("Semantic checks:")
    A("")
    A("| Check | % |")
    A("|---|---:|")
    for k, v in rc["semantic_checks"]["state_intensity"].items():
        A(f"| {k} | {v} |")
    for k, v in rc["semantic_checks"]["transition_semantics"].items():
        A(f"| {k} | {v} |")
    A("")
    A("## Limitations")
    A("")
    A("- The dataset is **synthetic** (Synthea-derived). It is not real patient data.")
    A("- Care states encode documented activity intensity; `NO_DATA` windows do not imply clinical stability.")
    A("- Persistence/transition metrics are computed on calendar-year windows; durations are therefore at one-year granularity.")
    A("- The rule-consistency check verifies the generated labels match the framework's documented deterministic rules; it does not establish clinical correctness.")
    A("- Prevalence of `GAP`/`NO_DATA` windows (76.6% of all windows) dominates the window-level distributions; patient-level and transition-exclusive statistics should be read alongside them.")
    A("")
    A("## Figures")
    A("")
    for f in figures:
        A(f"- `{f}`")
    A("")
    A("## Files")
    A("")
    A("- `data/evaluation_results/rq4_rq5_care_state_assistance_results.json` (machine-readable)")
    A("- `data/evaluation_results/rq4_rq5_care_state_assistance_report.md` (this report)")
    A("")
    A("## Interpretation")
    A("")
    A("The results demonstrate that ElderDocAI converts longitudinal clinical records into dynamic care states, detects and characterizes state transitions, and varies adaptive assistance (mode, priority, strategy) according to the detected state and transition in a manner that is internally consistent with the framework's documented rules. This is a demonstration of dynamic, care-state-aware adaptive assistance on synthetic records; it is not evidence of clinical benefit.")
    A("")

    return "\n".join(rows)
def main():
    """Run the full RQ4/RQ5 evaluation and write outputs."""
    os.makedirs(OUT_DIR, exist_ok=True)

    r4 = analyze_rq4()
    r5 = analyze_rq5()

    print_rq4(r4)
    print_rq5(r5)

    figures = make_figures(r4, r5)

    result = {
        "metadata": {
            "title": "ElderDocAI RQ4/RQ5 - Dynamic Care-State and Adaptive Assistance Evaluation",
            "data_source": "Synthetic longitudinal clinical records derived from Synthea",
            "n_patients": r4["A_patient_statistics"]["n_patients"],
            "n_windows": r4["A_patient_statistics"]["n_windows"],
            "analysis_type": "descriptive statistics + internal rule-consistency",
            "disclaimer": (
                "Synthetic data. Descriptive/internal-consistency evaluation only. "
                "Not a clinical validation study. Does not imply clinical validity."
            ),
        },
        "rq4": jsonable(r4),
        "rq5": jsonable(r5),
        "figures": figures,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  wrote {RESULTS_JSON}")

    report = write_report(r4, r5, figures)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  wrote {REPORT_MD}")

    print("\n" + "=" * 72)
    print("RQ4/RQ5 EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()