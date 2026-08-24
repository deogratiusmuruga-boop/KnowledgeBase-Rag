"""
ElderDocAI Medication Reference Resolver
=========================================

Purpose
-------
Resolve MedicationAdministration medication UUID references
in the ElderDocAI patient timeline using the original Synthea
FHIR Medication resources.

Input
-----
elderdocai/processed/patient_timeline.csv

Output
------
elderdocai/processed/patient_timeline_resolved.csv

Important
---------
The original patient_timeline.csv is NEVER modified.

The resolver only changes the `code` column for medication
events whose code is an unresolved UUID/reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


# =============================================================
# PATH CONFIGURATION
# =============================================================

BASE_DIR = Path(__file__).resolve().parent

TIMELINE_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "patient_timeline.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "elderdocai"
    / "processed"
    / "patient_timeline_resolved.csv"
)


# =============================================================
# CONFIGURATION
# =============================================================

MEDICATION_RESOURCE_TYPE = "Medication"

MEDICATION_EVENT_TYPES = {
    "MedicationAdministration",
    "MedicationRequest",
}

UUID_PATTERN = re.compile(
    r"^(?:urn:uuid:)?"
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


# =============================================================
# GENERAL HELPERS
# =============================================================

def is_uuid_reference(value: Any) -> bool:
    """
    Return True if a value looks like a FHIR UUID reference.
    """

    if value is None:
        return False

    if pd.isna(value):
        return False

    value = str(value).strip()

    if not value:
        return False

    return bool(UUID_PATTERN.match(value))


def normalize_reference(reference: Any) -> Optional[str]:
    """
    Normalize a FHIR reference.

    Examples
    --------
    urn:uuid:abc
        -> abc

    Medication/abc
        -> abc

    abc
        -> abc
    """

    if reference is None:
        return None

    if isinstance(reference, float) and pd.isna(reference):
        return None

    reference = str(reference).strip()

    if not reference:
        return None

    if reference.startswith("urn:uuid:"):
        return reference[len("urn:uuid:"):]

    if "/" in reference:
        return reference.split("/")[-1]

    return reference


def extract_text_from_codeable_concept(
    codeable_concept: Any,
) -> Optional[str]:
    """
    Extract a human-readable medication name from a
    FHIR CodeableConcept.
    """

    if not isinstance(codeable_concept, dict):
        return None

    # Prefer coding.display
    coding = codeable_concept.get("coding")

    if isinstance(coding, list):
        for item in coding:
            if not isinstance(item, dict):
                continue

            display = item.get("display")

            if display:
                return str(display).strip()

            code = item.get("code")

            if code:
                return str(code).strip()

    # Fall back to CodeableConcept.text
    text = codeable_concept.get("text")

    if text:
        return str(text).strip()

    return None


def extract_medication_name(resource: Dict[str, Any]) -> Optional[str]:
    """
    Extract the best available human-readable medication name
    from a FHIR Medication resource.
    """

    # ---------------------------------------------------------
    # Medication.code
    # ---------------------------------------------------------

    code = resource.get("code")

    name = extract_text_from_codeable_concept(code)

    if name:
        return name

    # ---------------------------------------------------------
    # Medication.code.text
    # ---------------------------------------------------------

    if isinstance(code, dict):
        text = code.get("text")

        if text:
            return str(text).strip()

    # ---------------------------------------------------------
    # Ingredient fallback
    # ---------------------------------------------------------

    ingredients = resource.get("ingredient")

    if isinstance(ingredients, list):

        names = []

        for ingredient in ingredients:

            if not isinstance(ingredient, dict):
                continue

            item = ingredient.get("itemCodeableConcept")

            name = extract_text_from_codeable_concept(item)

            if name:
                names.append(name)

        if names:
            return "; ".join(names)

    return None


# =============================================================
# FHIR RESOURCE ITERATION
# =============================================================

def iter_fhir_resources(
    obj: Any,
) -> Iterable[Dict[str, Any]]:
    """
    Recursively walk an arbitrary JSON object and yield
    FHIR resource dictionaries.

    Supports:

    - Individual FHIR resources
    - Bundles
    - Lists
    - Nested structures
    """

    if isinstance(obj, dict):

        resource_type = obj.get("resourceType")

        if resource_type:
            yield obj

        # Bundle entries
        entries = obj.get("entry")

        if isinstance(entries, list):

            for entry in entries:

                if not isinstance(entry, dict):
                    continue

                resource = entry.get("resource")

                if isinstance(resource, dict):
                    yield from iter_fhir_resources(resource)

        # Generic recursive traversal
        for key, value in obj.items():

            if key == "entry":
                continue

            if isinstance(value, (dict, list)):
                yield from iter_fhir_resources(value)

    elif isinstance(obj, list):

        for item in obj:
            yield from iter_fhir_resources(item)


# =============================================================
# FHIR FILE DISCOVERY
# =============================================================

def discover_fhir_files() -> list[Path]:
    """
    Discover JSON FHIR files under the Synthea dataset directory.

    The ElderDocAI directories are excluded to avoid accidentally
    reading generated output files.
    """

    files = []

    excluded_parts = {
        "elderdocai",
        ".venv",
        "venv",
        "__pycache__",
    }

    for path in BASE_DIR.rglob("*.json"):

        relative_parts = set(path.relative_to(BASE_DIR).parts)

        if relative_parts.intersection(excluded_parts):
            continue

        files.append(path)

    return sorted(files)


# =============================================================
# MEDICATION INDEX
# =============================================================

def build_medication_index(
    fhir_files: list[Path],
) -> Dict[str, str]:
    """
    Build:

        Medication UUID/reference -> medication name

    from Synthea FHIR JSON files.
    """

    medication_index: Dict[str, str] = {}

    files_read = 0
    resources_found = 0

    print()
    print("=" * 70)
    print("BUILDING MEDICATION RESOURCE INDEX")
    print("=" * 70)

    for json_file in fhir_files:

        try:

            with json_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            files_read += 1

        except Exception as exc:

            print(
                f"[WARN] Could not read {json_file}: {exc}"
            )

            continue

        for resource in iter_fhir_resources(data):

            if resource.get("resourceType") != MEDICATION_RESOURCE_TYPE:
                continue

            resources_found += 1

            resource_id = resource.get("id")

            if not resource_id:
                continue

            medication_name = extract_medication_name(resource)

            if not medication_name:
                continue

            normalized_id = normalize_reference(resource_id)

            if normalized_id:
                medication_index[normalized_id] = medication_name

            # Some FHIR structures may expose fullUrl.
            full_url = resource.get("fullUrl")

            if full_url:

                normalized_full_url = normalize_reference(full_url)

                if normalized_full_url:
                    medication_index[
                        normalized_full_url
                    ] = medication_name

    print(
        f"FHIR JSON files read:       {files_read}"
    )

    print(
        f"Medication resources found: {resources_found}"
    )

    print(
        f"Medication references indexed: "
        f"{len(medication_index)}"
    )

    print("=" * 70)

    return medication_index


# =============================================================
# REFERENCE RESOLUTION
# =============================================================

def resolve_medication_references(
    df: pd.DataFrame,
    medication_index: Dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """
    Resolve medication UUID references in the timeline.

    Only medication events are modified.

    Returns
    -------
    resolved dataframe
    statistics dictionary
    """

    df = df.copy()

    medication_mask = df["resource_type"].isin(
        MEDICATION_EVENT_TYPES
    )

    medication_rows = df.loc[medication_mask]

    total_medication_events = len(medication_rows)

    uuid_mask = medication_rows["code"].apply(
        is_uuid_reference
    )

    uuid_rows = medication_rows.loc[uuid_mask]

    uuid_before = len(uuid_rows)

    resolved_count = 0
    unresolved_count = 0

    unresolved_references = {}

    for index in uuid_rows.index:

        original_reference = df.at[index, "code"]

        normalized_reference = normalize_reference(
            original_reference
        )

        medication_name = None

        if normalized_reference:
            medication_name = medication_index.get(
                normalized_reference
            )

        if medication_name:

            df.at[index, "code"] = medication_name

            resolved_count += 1

        else:

            unresolved_count += 1

            reference_text = str(original_reference)

            unresolved_references[reference_text] = (
                unresolved_references.get(
                    reference_text,
                    0,
                )
                + 1
            )

    statistics = {
        "total_medication_events": total_medication_events,
        "uuid_before": uuid_before,
        "resolved": resolved_count,
        "unresolved": unresolved_count,
        "unresolved_unique": len(unresolved_references),
        "unresolved_references": unresolved_references,
    }

    return df, statistics


# =============================================================
# SAVE OUTPUT
# =============================================================

def save_timeline(
    df: pd.DataFrame,
) -> None:
    """
    Save the resolved timeline.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("ELDERDOCAI MEDICATION REFERENCE RESOLVER")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if not TIMELINE_FILE.exists():

        raise FileNotFoundError(
            f"Timeline file not found:\n{TIMELINE_FILE}"
        )

    # ---------------------------------------------------------
    # Load timeline
    # ---------------------------------------------------------

    print()
    print("Loading patient timeline...")

    df = pd.read_csv(
        TIMELINE_FILE,
        dtype=str,
    )

    print(
        f"Timeline rows: {len(df):,}"
    )

    print(
        f"Timeline columns: {len(df.columns)}"
    )

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = {
        "patient_id",
        "resource_type",
        "event_date",
        "date_source",
        "code",
        "value",
        "unit",
        "status",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # ---------------------------------------------------------
    # Discover FHIR files
    # ---------------------------------------------------------

    print()
    print("Searching for Synthea FHIR files...")

    fhir_files = discover_fhir_files()

    print(
        f"FHIR JSON files discovered: "
        f"{len(fhir_files)}"
    )

    if not fhir_files:

        raise FileNotFoundError(
            "No Synthea FHIR JSON files were found."
        )

    # ---------------------------------------------------------
    # Build medication index
    # ---------------------------------------------------------

    medication_index = build_medication_index(
        fhir_files
    )

    if not medication_index:

        raise RuntimeError(
            "No Medication resources could be indexed."
        )

    # ---------------------------------------------------------
    # Resolve references
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESOLVING MEDICATION REFERENCES")
    print("=" * 70)

    resolved_df, stats = resolve_medication_references(
        df,
        medication_index,
    )

    print(
        f"Medication events:          "
        f"{stats['total_medication_events']:,}"
    )

    print(
        f"UUID references before:     "
        f"{stats['uuid_before']:,}"
    )

    print(
        f"Successfully resolved:      "
        f"{stats['resolved']:,}"
    )

    print(
        f"Still unresolved:           "
        f"{stats['unresolved']:,}"
    )

    print(
        f"Unique unresolved UUIDs:    "
        f"{stats['unresolved_unique']:,}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print()
    print("Saving resolved timeline...")

    save_timeline(
        resolved_df
    )

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    # ---------------------------------------------------------
    # Integrity check
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUT INTEGRITY CHECK")
    print("=" * 70)

    print(
        f"Original rows: {len(df):,}"
    )

    print(
        f"Output rows:   {len(resolved_df):,}"
    )

    if len(df) != len(resolved_df):

        raise RuntimeError(
            "Row count changed during medication resolution."
        )

    if list(df.columns) != list(resolved_df.columns):

        raise RuntimeError(
            "Column structure changed during medication resolution."
        )

    print(
        "[PASS] Row count preserved."
    )

    print(
        "[PASS] Column structure preserved."
    )

    # ---------------------------------------------------------
    # Remaining UUIDs
    # ---------------------------------------------------------

    medication_mask = resolved_df["resource_type"].isin(
        MEDICATION_EVENT_TYPES
    )

    remaining_uuid_count = (
        resolved_df.loc[
            medication_mask,
            "code",
        ]
        .apply(is_uuid_reference)
        .sum()
    )

    print(
        f"Remaining medication UUIDs: "
        f"{remaining_uuid_count:,}"
    )

    # ---------------------------------------------------------
    # Show unresolved references
    # ---------------------------------------------------------

    if stats["unresolved"] > 0:

        print()
        print("=" * 70)
        print("UNRESOLVED MEDICATION REFERENCES")
        print("=" * 70)

        sorted_unresolved = sorted(
            stats["unresolved_references"].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for reference, count in sorted_unresolved[:20]:

            print(
                f"{count:6d}  {reference}"
            )

        if len(sorted_unresolved) > 20:

            print(
                f"... and "
                f"{len(sorted_unresolved) - 20:,} "
                f"more unique references."
            )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("MEDICATION REFERENCE RESOLUTION COMPLETE")
    print("=" * 70)

    if remaining_uuid_count == 0:

        print(
            "[PASS] All medication UUID references were resolved."
        )

    else:

        print(
            "[WARN] Some medication UUID references remain unresolved."
        )

    print(
        f"Input:  {TIMELINE_FILE}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()