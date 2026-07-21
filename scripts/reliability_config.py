"""Loading and validation for trustworthy-RAG reliability configuration."""

import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_FILE = os.path.join(BASE_DIR, "config", "reliability_config.json")
RELIABILITY_DIMENSIONS = (
    "authority",
    "relevance",
    "support",
    "coverage",
    "consistency",
)


def load_reliability_config(config_file=DEFAULT_CONFIG_FILE):
    """Load the externally configurable reliability weights and thresholds."""
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Reliability configuration must be a JSON object.")

    weights = config.get("reliability_weights")
    if not isinstance(weights, dict) or set(weights) != set(RELIABILITY_DIMENSIONS):
        raise ValueError(
            "reliability_weights must define authority, relevance, support, coverage, "
            "and consistency."
        )

    normalized_weights = {}
    for dimension in RELIABILITY_DIMENSIONS:
        try:
            weight = float(weights[dimension])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Weight for {dimension} must be numeric.") from error
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"Weight for {dimension} must be between 0 and 1.")
        normalized_weights[dimension] = weight

    if abs(sum(normalized_weights.values()) - 1.0) > 1e-9:
        raise ValueError("Reliability weights must sum to 1.0.")

    thresholds = config.get("decision_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("decision_thresholds must be a JSON object.")

    try:
        normalized_thresholds = {
            "accept": float(thresholds["accept"]),
            "refine": float(thresholds["refine"]),
            "re_retrieve": float(thresholds["re_retrieve"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "decision_thresholds must define numeric accept, refine, and re_retrieve values."
        ) from error

    if not (
        0.0 <= normalized_thresholds["re_retrieve"]
        < normalized_thresholds["refine"]
        < normalized_thresholds["accept"]
        <= 1.0
    ):
        raise ValueError(
            "Thresholds must satisfy 0 <= re_retrieve < refine < accept <= 1."
        )

    return {
        "reliability_weights": normalized_weights,
        "decision_thresholds": normalized_thresholds,
    }
