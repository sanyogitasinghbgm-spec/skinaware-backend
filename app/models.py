from pydantic import BaseModel, EmailStr
from typing import Optional, Dict

# ---------- USER ----------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# ---------- ANALYSIS ----------
class AnalysisResult(BaseModel):
    dryness: str
    redness: str
    uneven_tone: str
    routine: str

# ---------- FEEDBACK ----------
class FeedbackCreate(BaseModel):
    analysis_id: str
    rating: int
    comments: Optional[str] = None
