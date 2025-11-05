from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import sqlite3

SQLALCHEMY_DATABASE_URL = "sqlite:///./interior_designer.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def _repair_if_corrupt():
    db_path = os.path.abspath("interior_designer.db")
    if not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("PRAGMA integrity_check;")
            row = cur.fetchone()
            if not row or row[0] != "ok":
                conn.close()
                os.remove(db_path)
                return
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except sqlite3.DatabaseError:
        try:
            os.remove(db_path)
        except Exception:
            pass

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    designs = relationship("DesignSelection", back_populates="owner", cascade="all, delete-orphan")

class DesignSelection(Base):
    __tablename__ = "design_selections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    room_type = Column(String)
    design_style = Column(String)
    background_color = Column(String)
    foreground_color = Column(String)
    additional_instructions = Column(Text)
    interior_recommendations = Column(Text)
    plants_info = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="designs")

# Ensure DB is healthy, then create tables
_repair_if_corrupt()
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

