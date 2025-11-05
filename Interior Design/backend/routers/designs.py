from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db, DesignSelection
from datetime import datetime

router = APIRouter()

class DesignSelectionCreate(BaseModel):
    room_type: str
    design_style: str
    background_color: str
    foreground_color: str
    additional_instructions: str
    interior_recommendations: str
    plants_info: str = ""

class DesignSelectionResponse(BaseModel):
    id: int
    room_type: str
    design_style: str
    background_color: str
    foreground_color: str
    additional_instructions: str
    interior_recommendations: str
    plants_info: str
    created_at: datetime
    
    class Config:
        from_attributes = True

@router.post("/save", response_model=DesignSelectionResponse)
def save_design_selection(
    design_data: DesignSelectionCreate,
    db: Session = Depends(get_db)
):
    db_design = DesignSelection(
        user_id=None,  # No user authentication required
        room_type=design_data.room_type,
        design_style=design_data.design_style,
        background_color=design_data.background_color,
        foreground_color=design_data.foreground_color,
        additional_instructions=design_data.additional_instructions,
        interior_recommendations=design_data.interior_recommendations,
        plants_info=design_data.plants_info,
    )
    db.add(db_design)
    db.commit()
    db.refresh(db_design)
    return db_design

