from sqlalchemy import Column, Integer, String, ForeignKey
from database import Base


class Medication(Base):

    __tablename__ = "medications"

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

    medicine_name = Column(
        String,
        nullable=False
    )

    dosage = Column(
        String,
        nullable=True
    )

    time = Column(
        String,
        nullable=True
    )

    frequency = Column(
        String,
        nullable=True
    )