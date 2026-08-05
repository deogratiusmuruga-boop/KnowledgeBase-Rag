from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models.appointment import Appointment
from api.dependencies import require_user


router = APIRouter(
    prefix="/appointments",
    tags=["Appointments"]
)


class AppointmentCreate(BaseModel):

    user_id: int

    title: str

    appointment_date: str

    appointment_time: str

    location: str | None = None

    notes: str | None = None


@router.post("/")
def create_appointment(
    appointment: AppointmentCreate,
    db: Session = Depends(get_db)
):
    require_user(db, appointment.user_id)

    new_appointment = Appointment(

        user_id=appointment.user_id,

        title=appointment.title,

        appointment_date=appointment.appointment_date,

        appointment_time=appointment.appointment_time,

        location=appointment.location,

        notes=appointment.notes

    )

    db.add(new_appointment)

    db.commit()

    db.refresh(new_appointment)

    return {

        "message": "Appointment saved successfully",

        "appointment_id": new_appointment.id

    }

@router.get("/")
def get_appointments(
    user_id: int,
    db: Session = Depends(get_db)
):
    require_user(db, user_id)

    appointments = (
        db.query(Appointment)
        .filter(Appointment.user_id == user_id)
        .all()
    )

    return appointments

@router.delete("/{appointment_id}")
def delete_appointment(
    appointment_id: int,
    db: Session = Depends(get_db)
):

    appointment = (
        db.query(Appointment)
        .filter(Appointment.id == appointment_id)
        .first()
    )

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    db.delete(appointment)

    db.commit()

    return {
        "message": "Appointment deleted successfully"
    }
