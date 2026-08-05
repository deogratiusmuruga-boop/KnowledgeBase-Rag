from database import engine, Base

from models.user import UserProfile
from models.medication import Medication
from models.appointment import Appointment


print("Creating database tables...")


Base.metadata.create_all(
    bind=engine
)


print("Tables created successfully!")