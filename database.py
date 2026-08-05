from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base


# ============================================================
# Database Configuration
# ============================================================

DATABASE_URL = "sqlite:///./carebuddy.db"


# ============================================================
# Database Engine
# ============================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)


@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """Enable SQLite's opt-in foreign-key enforcement for every connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ============================================================
# Database Session
# ============================================================

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ============================================================
# Base Class for Models
# ============================================================

Base = declarative_base()


# ============================================================
# Dependency for FastAPI
# ============================================================

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
