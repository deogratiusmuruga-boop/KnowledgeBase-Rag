from database import engine, Base

from models.user import UserProfile


print("Creating database tables...")


Base.metadata.create_all(
    bind=engine
)


print("Tables created successfully!")