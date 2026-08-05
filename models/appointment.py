from sqlalchemy import Column, Integer, String, ForeignKey
from database import Base


class Appointment(Base):

    __tablename__ = "appointments"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("user_profiles.id"),
        nullable=False
    )

    title = Column(
        String,
        nullable=False
    )

    appointment_date = Column(
        String,
        nullable=False
    )

    appointment_time = Column(
        String,
        nullable=False
    )

    location = Column(
        String,
        nullable=True
    )

    notes = Column(
        String,
        nullable=True
    )