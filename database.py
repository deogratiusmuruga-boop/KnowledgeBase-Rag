from sqlalchemy import create_engine
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