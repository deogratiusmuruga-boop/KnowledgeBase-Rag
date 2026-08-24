import json
import shutil
from pathlib import Path
from datetime import date

SOURCE_DIR = Path("output/fhir")
TARGET_DIR = Path("elderdocai/fhir")

MIN_AGE = 60

TARGET_DIR.mkdir(parents=True, exist_ok=True)

total_patients = 0
elderly_patients = 0

def calculate_age(birth_date):
    birth = date.fromisoformat(birth_date)
    today = date.today()

    age = today.year - birth.year

    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1

    return age


for file_path in SOURCE_DIR.glob("*.json"):

    # Ignore hospital/practitioner information files
    if "Information" in file_path.name:
        continue

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        patient = None

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})

            if resource.get("resourceType") == "Patient":
                patient = resource
                break

        if patient is None:
            continue

        total_patients += 1

        birth_date = patient.get("birthDate")

        if not birth_date:
            continue

        age = calculate_age(birth_date)

        if age >= MIN_AGE:

            elderly_patients += 1

            target_file = TARGET_DIR / file_path.name

            shutil.copy2(file_path, target_file)

            print(f"Included: {patient.get('id')} | Age: {age}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")


print()
print("=" * 60)
print("ELDERDOCAI COHORT CREATION")
print("=" * 60)
print(f"Total source patients:      {total_patients}")
print(f"Elderly patients (>=60):    {elderly_patients}")
print(f"Excluded patients (<60):    {total_patients - elderly_patients}")
print("=" * 60)