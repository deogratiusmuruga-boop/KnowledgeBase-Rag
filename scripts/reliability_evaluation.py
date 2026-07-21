"""Rule-based reliability scoring for retrieved RAG evidence."""

import re

from reliability_config import RELIABILITY_DIMENSIONS, load_reliability_config

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}

OPPOSING_RECOMMENDATIONS = (
    ("recommended", "not recommended"),
    ("should", "should not"),
    ("must", "must not"),
    ("always", "never"),
)


def _content_terms(text):
    """Return normalized, meaningful terms."""
    return {
        term
        for term in re.findall(r"[a-zA-Z]{3,}", text.casefold())
        if term not in STOP_WORDS
    }


def _clamp_score(value):
    return max(0.0, min(1.0, float(value)))


def _normalized_cosine_similarity(similarity_score):
    """Convert cosine similarity from [-1,1] to [0,1]."""
    return _clamp_score((float(similarity_score) + 1.0) / 2.0)


def _contains_positive_recommendation(text, recommendation):
    normalized = text.casefold()
    return (
        recommendation in normalized
        and f"not {recommendation}" not in normalized
    )


def _has_explicit_opposition(left_text, right_text):
    """Detect explicit contradictory recommendations."""
    for positive, negative in OPPOSING_RECOMMENDATIONS:

        if (
            _contains_positive_recommendation(left_text, positive)
            and negative in right_text.casefold()
        ) or (
            _contains_positive_recommendation(right_text, positive)
            and negative in left_text.casefold()
        ):
            return True

    return False


def _consistency_score(evidence_items):
    """Estimate consistency among retrieved evidence."""

    if len(evidence_items) < 2:
        return 1.0

    conflicts = 0
    comparisons = 0

    item_terms = [
        _content_terms(item["text"])
        for item in evidence_items
    ]

    for i in range(len(evidence_items)):
        for j in range(i + 1, len(evidence_items)):

            comparisons += 1

            shared_terms = item_terms[i] & item_terms[j]

            if (
                _has_explicit_opposition(
                    evidence_items[i]["text"],
                    evidence_items[j]["text"]
                )
                and len(shared_terms) >= 3
            ):
                conflicts += 1

    if comparisons == 0:
        return 1.0

    return 1.0 - (conflicts / comparisons)


def evaluate_reliability(query, evidence_items, weights=None):
    """Evaluate retrieval reliability."""

    required_fields = {
        "text",
        "similarity_score",
        "authority_score"
    }

    for position, item in enumerate(evidence_items, start=1):

        if (
            not isinstance(item, dict)
            or not required_fields <= item.keys()
        ):
            raise ValueError(
                f"Evidence item {position} is missing required fields."
            )

    if weights is None:
        weights = load_reliability_config()["reliability_weights"]

    if set(weights) != set(RELIABILITY_DIMENSIONS):
        raise ValueError(
            "Reliability weights must define every reliability dimension."
        )

    weights = {
        dimension: float(weights[dimension])
        for dimension in RELIABILITY_DIMENSIONS
    }

    if any(weight < 0 or weight > 1 for weight in weights.values()):
        raise ValueError(
            "Reliability weights must be between 0 and 1."
        )

    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            "Reliability weights must sum to 1.0."
        )

    if not evidence_items:

        return {
            "authority": 0.0,
            "relevance": 0.0,
            "support": 0.0,
            "coverage": 0.0,
            "consistency": 0.0,
            "overall_reliability": 0.0,
        }

    # =====================================================
    # Authority
    # =====================================================

    authority = sum(
        _clamp_score(item["authority_score"])
        for item in evidence_items
    ) / len(evidence_items)

    # =====================================================
    # Relevance
    # =====================================================

    relevance = sum(
        _normalized_cosine_similarity(
            item["similarity_score"]
        )
        for item in evidence_items
    ) / len(evidence_items)

    # =====================================================
    # Support & Coverage
    # =====================================================

    query_terms = _content_terms(query)

    evidence_terms = [
        _content_terms(item["text"])
        for item in evidence_items
    ]

    if query_terms:

        MIN_SHARED_TERMS = 2

        supporting_chunks = sum(
            len(query_terms & terms) >= MIN_SHARED_TERMS
            for terms in evidence_terms
        )

        support = supporting_chunks / len(evidence_items)

        covered_terms = set()

        for terms in evidence_terms:
            covered_terms.update(query_terms & terms)

        coverage = len(covered_terms) / len(query_terms)

    else:

        support = relevance
        coverage = relevance

    # =====================================================
    # Consistency
    # =====================================================

    consistency = _consistency_score(evidence_items)

    # =====================================================
    # Overall Reliability
    # =====================================================

    dimension_scores = {
        "authority": authority,
        "relevance": relevance,
        "support": support,
        "coverage": coverage,
        "consistency": consistency,
    }

    overall_reliability = sum(
        weights[dimension] * dimension_scores[dimension]
        for dimension in RELIABILITY_DIMENSIONS
    )

    return {
        "authority": _clamp_score(authority),
        "relevance": _clamp_score(relevance),
        "support": _clamp_score(support),
        "coverage": _clamp_score(coverage),
        "consistency": _clamp_score(consistency),
        "overall_reliability": _clamp_score(overall_reliability),
    }