import json
from pathlib import Path
from collections import Counter

FHIR_DIR = Path("output/fhir")

resource_counts = Counter()
patient_files = 0

print("Scanning Synthea FHIR dataset...")
print("")

for file_path in FHIR_DIR.glob("*.json"):

    # Skip hospital/practitioner information files
    if "Information" in file_path.name:
        continue

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        patient_files += 1

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type:
                resource_counts[resource_type] += 1

    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")

print("")
print("=" * 50)
print("FHIR RESOURCE SUMMARY")
print("=" * 50)

for resource_type, count in resource_counts.most_common():
    print(f"{resource_type:<30} {count:>10}")

print("=" * 50)
print(f"Patient FHIR files processed: {patient_files}")
print(f"Total FHIR resources: {sum(resource_counts.values())}")
print("=" * 50)