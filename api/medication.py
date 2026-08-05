from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models.medication import Medication
from api.dependencies import require_user


router = APIRouter(
    prefix="/medications",
    tags=["Medications"]
)



# Request Schema

class MedicationCreate(BaseModel):

    user_id: int

    medicine_name: str

    dosage: str | None = None

    time: str | None = None

    frequency: str | None = None


# Save Medication

@router.post("/")
def create_medication(
    medication: MedicationCreate,
    db: Session = Depends(get_db)
):
    require_user(db, medication.user_id)

    new_medication = Medication(

        user_id=medication.user_id,

        medicine_name=medication.medicine_name,

        dosage=medication.dosage,

        time=medication.time,

        frequency=medication.frequency

    )

    db.add(new_medication)

    db.commit()

    db.refresh(new_medication)

    return {

        "message": "Medication saved successfully",

        "medication_id": new_medication.id

    }


# Get User Medications
@router.get("/{user_id}")
def get_medications(
    user_id: int,
    db: Session = Depends(get_db)
):
    require_user(db, user_id)

    medications = db.query(Medication).filter(
        Medication.user_id == user_id
    ).all()

    return [

        {
            "id": med.id,
            "medicine_name": med.medicine_name,
            "dosage": med.dosage,
            "time": med.time,
            "frequency": med.frequency
        }

        for med in medications

    ]
