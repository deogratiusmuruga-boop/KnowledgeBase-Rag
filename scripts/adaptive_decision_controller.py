"""Adaptive decision policy for trustworthy RAG reliability scores."""

from reliability_config import load_reliability_config


DECISION_REASONS = {
    "ACCEPT": "Retrieved evidence meets the configured high-reliability threshold.",
    "REFINE": "Retrieved evidence is moderately reliable and should be refined before use.",
    "RE-RETRIEVE": "Retrieved evidence has limited reliability and should be retrieved again.",
    "REJECT": "Retrieved evidence is too unreliable to ground an answer.",
}


def make_reliability_decision(reliability, config=None):
    """Return the policy decision for a reliability score in the range [0, 1]."""
    reliability = float(reliability)
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("Reliability must be between 0 and 1.")

    config = config or load_reliability_config()
    thresholds = config["decision_thresholds"]

    if reliability >= thresholds["accept"]:
        decision = "ACCEPT"
    elif reliability >= thresholds["refine"]:
        decision = "REFINE"
    elif reliability >= thresholds["re_retrieve"]:
        decision = "RE-RETRIEVE"
    else:
        decision = "REJECT"

    return {
        "decision": decision,
        "reliability": reliability,
        "reason": DECISION_REASONS[decision],
    }
