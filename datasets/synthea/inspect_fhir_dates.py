import json
from pathlib import Path
from collections import Counter

FHIR_DIR = Path("elderdocai/fhir")

date_fields = Counter()

for file_path in FHIR_DIR.glob("*.json"):

    with open(file_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    for entry in bundle.get("entry", []):

        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if not resource_type:
            continue

        for field in [
            "effectiveDateTime",
            "issued",
            "authoredOn",
            "performedDateTime",
            "occurrenceDateTime",
            "recordedDate",
            "onsetDateTime",
            "abatementDateTime",
            "start",
            "end"
        ]:

            if field in resource:
                date_fields[(resource_type, field)] += 1

print("=" * 65)
print("FHIR DATE/TIME FIELD SUMMARY")
print("=" * 65)

for (resource_type, field), count in date_fields.most_common():
    print(f"{resource_type:<30} {field:<25} {count:>8}")

print("=" * 65)