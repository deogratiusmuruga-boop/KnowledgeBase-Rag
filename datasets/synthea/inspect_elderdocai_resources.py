import json
from pathlib import Path
from collections import Counter

FHIR_DIR = Path("elderdocai/fhir")

resource_counts = Counter()
patient_count = 0

print("Scanning ElderDocAI elderly cohort...")
print()

for file_path in FHIR_DIR.glob("*.json"):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        patient_count += 1

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type:
                resource_counts[resource_type] += 1

    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

print("=" * 55)
print("ELDERDOCAI RESOURCE SUMMARY")
print("=" * 55)

for resource_type, count in resource_counts.most_common():
    print(f"{resource_type:<30} {count:>10}")

print("=" * 55)
print(f"Elderly patient files: {patient_count}")
print(f"Total resources:       {sum(resource_counts.values())}")
print("=" * 55)