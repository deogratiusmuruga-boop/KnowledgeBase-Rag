"""Reusable source-authority classification for healthcare RAG evidence."""

from dataclasses import dataclass


AUTHORITY_SCORES = {
    "WHO": 1.00,
    "NIA": 0.95,
    "NIH": 0.95,
    "CDC": 0.95,
    "Government publication": 0.90,
    "Other trusted medical": 0.85,
    "Unknown": 0.50,
}


@dataclass(frozen=True)
class SourceAuthority:
    """The rule-based authority classification for one evidence source."""

    document_category: str
    authority_score: float


def classify_source_authority(source_document, text=""):
    """Classify a source using publisher indicators, not document-specific names.

    The checks intentionally use organization and government publisher terms, making
    the mapping reusable as new documents are added to the knowledge base.
    """
    source_name = source_document.casefold()
    text_excerpt = text[:2000]

    def source_has_indicator(indicator):
        return indicator in source_name

    if (
        "World Health Organization" in text_excerpt
        or source_has_indicator("who_")
        or source_has_indicator("_who")
    ):
        category = "WHO"
    elif (
        "National Institute on Aging" in text_excerpt
        or source_has_indicator("nia_")
        or source_has_indicator("_nia")
    ):
        category = "NIA"
    elif "National Institutes of Health" in text_excerpt or source_has_indicator("nih_"):
        category = "NIH"
    elif "Centers for Disease Control" in text_excerpt or source_has_indicator("cdc_"):
        category = "CDC"
    elif any(
        indicator in source_name or indicator in text_excerpt.casefold()
        for indicator in (".gov", "government", "usda", "department of health")
    ):
        category = "Government publication"
    elif any(
        indicator in source_name or indicator in text_excerpt.casefold()
        for indicator in ("hospital", "medical center", "medical association", "health system")
    ):
        category = "Other trusted medical"
    else:
        category = "Unknown"

    return SourceAuthority(
        document_category=category,
        authority_score=AUTHORITY_SCORES[category],
    )
