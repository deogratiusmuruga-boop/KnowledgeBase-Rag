from sqlalchemy import Column, Integer, String
from database import Base


class UserProfile(Base):

    __tablename__ = "user_profiles"


    id = Column(
        Integer,
        primary_key=True,
        index=True
    )


    age = Column(
        Integer,
        nullable=True
    )


    location = Column(
        String,
        nullable=True
    )


    chronic_conditions = Column(
        String,
        nullable=True
    )


    medications = Column(
        String,
        nullable=True
    )


    preferred_language = Column(
        String,
        default="en"
    )


    speech_speed = Column(
        String,
        default="normal"
    )