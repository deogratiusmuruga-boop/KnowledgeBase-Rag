"""
ElderDocAI RQ3 - Reliability Gating Evaluation.

Analyzes the reliability-gating behavior of the production pipeline on the
16 gold QA questions, using the EXISTING reliability configuration
(config/reliability_config.json) and the EXISTING adaptive decision
controller thresholds.

Inputs:
- data/evaluation_results/gold_qa_results.json  (from RQ2 evaluator)
- config/reliability_config.json                (weights + thresholds)

Outputs:
- data/evaluation_results/reliability_gating_results.json
- concise terminal summary

This is a read-only analysis of already-collected results. It does NOT
re-run generation or modify production code.
"""

import os
import json

from scripts.reliability_config import load_reliability_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "data", "evaluation_results", "gold_qa_results.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "evaluation_results", "reliability_gating_results.json")


def main():
    config = load_reliability_config()
    weights = config["reliability_weights"]
    thresholds = config["decision_thresholds"]

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])

    print("=" * 70)
    print("ElderDocAI RQ3 - Reliability Gating Evaluation")
    print("=" * 70)
    print(f"Config weights     : {weights}")
    print(f"Config thresholds  : {thresholds}")

    # Recompute decisions from stored reliability using the SAME controller
    # to confirm the stored decisions are reproducible.
    from scripts.adaptive_decision_controller import make_reliability_decision

    rows = []
    for rec in results:
        rel = rec.get("reliability")
        if not rel:
            continue
        decision = make_reliability_decision(rel)
        rows.append({
            "id": rec.get("id"),
            "overall_reliability": rel.get("overall_reliability"),
            "authority": rel.get("authority"),
            "relevance": rel.get("relevance"),
            "support": rel.get("support"),
            "coverage": rel.get("coverage"),
            "consistency": rel.get("consistency"),
            "stored_decision": rec.get("adaptive_decision"),
            "recomputed_decision": decision.get("decision"),
            "decision_consistent": rec.get("adaptive_decision") == decision.get("decision"),
        })

    # ---- Summary ----
    n = len(rows)
    decisions = {}
    for r in rows:
        decisions[r["recomputed_decision"]] = decisions.get(r["recomputed_decision"], 0) + 1

    overalls = [r["overall_reliability"] for r in rows]
    avg_overall = sum(overalls) / len(overalls) if overalls else None
    min_overall = min(overalls) if overalls else None
    max_overall = max(overalls) if overalls else None

    # Threshold analysis
    accept_thr = thresholds["accept"]
    refine_thr = thresholds["refine"]
    reretrieve_thr = thresholds["re_retrieve"]

    n_accept = sum(1 for r in rows if r["overall_reliability"] >= accept_thr)
    n_refine = sum(1 for r in rows if refine_thr <= r["overall_reliability"] < accept_thr)
    n_reretrieve = sum(1 for r in rows if reretrieve_thr <= r["overall_reliability"] < refine_thr)
    n_reject = sum(1 for r in rows if r["overall_reliability"] < reretrieve_thr)

    # Dimension averages
    dims = ["authority", "relevance", "support", "coverage", "consistency"]
    dim_avg = {}
    for d in dims:
        vals = [r[d] for r in rows if r.get(d) is not None]
        dim_avg[d] = round(sum(vals) / len(vals), 4) if vals else None

    print("\n" + "=" * 70)
    print("RELIABILITY GATING SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated        : {n}")
    print(f"Overall reliability        : avg={avg_overall:.4f} min={min_overall:.4f} max={max_overall:.4f}")
    print(f"Decision distribution      : {decisions}")
    print(f"  ACCEPT (>= {accept_thr})      : {n_accept}")
    print(f"  REFINE (>= {refine_thr})      : {n_refine}")
    print(f"  RE-RETRIEVE (>= {reretrieve_thr}): {n_reretrieve}")
    print(f"  REJECT (< {reretrieve_thr})    : {n_reject}")
    print(f"Dimension averages         : {dim_avg}")
    print(f"Decision consistency       : {sum(1 for r in rows if r['decision_consistent'])}/{n}")

    print("\nPer-question reliability gating:")
    for r in rows:
        print(
            f"  {r['id']}: overall={r['overall_reliability']:.4f} "
            f"dec={r['recomputed_decision']} "
            f"(stored={r['stored_decision']}, consistent={r['decision_consistent']})"
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "summary": {
                "n": n,
                "avg_overall": avg_overall,
                "min_overall": min_overall,
                "max_overall": max_overall,
                "decision_distribution": decisions,
                "n_accept": n_accept,
                "n_refine": n_refine,
                "n_reretrieve": n_reretrieve,
                "n_reject": n_reject,
                "dimension_averages": dim_avg,
                "decision_consistency": sum(1 for r in rows if r["decision_consistent"]),
            },
            "rows": rows,
        }, f, indent=2)
    print(f"\nWrote JSON results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()