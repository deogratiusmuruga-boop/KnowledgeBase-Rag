from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models.medication import Medication
from models.appointment import Appointment
from api.dependencies import require_user

router = APIRouter(
    prefix="/reminders",
    tags=["Reminder Engine"]
)


@router.get("/{user_id}")
def get_reminders(
    user_id: int,
    db: Session = Depends(get_db)
):
    require_user(db, user_id)

    medications = db.query(Medication).filter(
        Medication.user_id == user_id
    ).all()

    appointments = db.query(Appointment).filter(
        Appointment.user_id == user_id
    ).all()

    reminders = []

    # -----------------------------
    # Medication reminders
    # -----------------------------
    for med in medications:

        reminders.append({

            "type": "medication",

            "title": med.medicine_name,

            "time": med.time,

            "details": med.dosage

        })

    
    # Appointment reminders
   
    for appt in appointments:

        reminders.append({

            "type": "appointment",

            "title": appt.title,

            "date": appt.appointment_date,

            "time": appt.appointment_time,

            "details": appt.location

        })

    return reminders
