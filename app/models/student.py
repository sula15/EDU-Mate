"""
Pydantic models for student-related data structures
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class LearningStyleEnum(str, Enum):
    """Enum for learning styles"""
    DETAILED = "detailed"
    CONCISE = "concise"
    BULLETED = "bulleted"
    ELI5 = "eli5"
    VISUAL = "visual"
    QUIZ = "quiz"


class StudentRegistration(BaseModel):
    """Model for student registration"""
    student_id: str = Field(..., min_length=1, max_length=50, description="Unique student identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Student's full name")
    email: Optional[EmailStr] = Field(None, description="Student's email address")
    default_learning_style: LearningStyleEnum = Field(
        LearningStyleEnum.DETAILED, 
        description="Preferred learning style"
    )


class StudentLogin(BaseModel):
    """Model for student login"""
    student_id: str = Field(..., min_length=1, max_length=50, description="Student identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Student's name")


class StudentProfile(BaseModel):
    """Model for complete student profile"""
    student_id: str = Field(..., description="Unique student identifier")
    name: str = Field(..., description="Student's full name")
    email: Optional[str] = Field(None, description="Student's email address")
    default_learning_style: LearningStyleEnum = Field(..., description="Preferred learning style")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_active: datetime = Field(..., description="Last activity timestamp")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Additional preferences")
    modules_accessed: List[str] = Field(default_factory=list, description="List of accessed modules")
    total_interactions: int = Field(default=0, description="Total number of interactions")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StudentProfileUpdate(BaseModel):
    """Model for updating student profile"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Student's full name")
    email: Optional[EmailStr] = Field(None, description="Student's email address")
    default_learning_style: Optional[LearningStyleEnum] = Field(None, description="Preferred learning style")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Additional preferences")


class StudentInteractionCreate(BaseModel):
    """Model for creating a new student interaction"""
    student_id: str = Field(..., description="Student identifier")
    query: str = Field(..., min_length=1, description="Student's query")
    learning_style_id: Optional[LearningStyleEnum] = Field(None, description="Learning style for this interaction")
    module_id: Optional[str] = Field(None, description="Module ID related to this interaction")
    lecture_code: Optional[str] = Field(None, description="Lecture code related to this interaction")


class StudentInteraction(BaseModel):
    """Model for complete student interaction"""
    interaction_id: str = Field(..., description="Unique interaction identifier")
    student_id: str = Field(..., description="Student identifier")
    query: str = Field(..., description="Student's query")
    learning_style_id: LearningStyleEnum = Field(..., description="Learning style used")
    module_id: Optional[str] = Field(None, description="Module ID")
    lecture_code: Optional[str] = Field(None, description="Lecture code")
    lecture_number: Optional[int] = Field(None, description="Lecture number")
    start_time: datetime = Field(..., description="Interaction start time")
    end_time: Optional[datetime] = Field(None, description="Interaction end time")
    time_spent: Optional[float] = Field(None, description="Time spent in seconds")
    feedback: Optional[str] = Field(None, description="Student feedback")
    helpful: Optional[bool] = Field(None, description="Whether response was helpful")
    follow_up_queries: List[Dict[str, Any]] = Field(default_factory=list, description="Follow-up queries")
    retrieved_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved sources")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StudentInteractionUpdate(BaseModel):
    """Model for updating student interaction"""
    end_time: Optional[datetime] = Field(None, description="Interaction end time")
    time_spent: Optional[float] = Field(None, description="Time spent in seconds")
    feedback: Optional[str] = Field(None, description="Student feedback")
    helpful: Optional[bool] = Field(None, description="Whether response was helpful")
    retrieved_sources: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved sources")


class StudentFeedback(BaseModel):
    """Model for student feedback"""
    interaction_id: str = Field(..., description="Interaction identifier")
    feedback: Optional[str] = Field(None, description="Text feedback")
    helpful: bool = Field(..., description="Whether response was helpful")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")


class LearningStyleInfo(BaseModel):
    """Model for learning style information"""
    id: LearningStyleEnum = Field(..., description="Learning style ID")
    name: str = Field(..., description="Learning style name")
    description: str = Field(..., description="Learning style description")
    prompt_template: str = Field(..., description="Prompt template for this style")


class StudentAnalyticsSummary(BaseModel):
    """Model for student analytics summary"""
    student_id: str = Field(..., description="Student identifier")
    total_interactions: int = Field(..., description="Total number of interactions")
    total_study_time: float = Field(..., description="Total study time in minutes")
    avg_session_duration: float = Field(..., description="Average session duration in minutes")
    satisfaction_rate: Optional[float] = Field(None, description="Satisfaction rate (0-1)")
    preferred_learning_style: Optional[LearningStyleInfo] = Field(None, description="Most used learning style")
    modules_studied: List[str] = Field(default_factory=list, description="List of studied modules")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModuleAnalytics(BaseModel):
    """Model for module-specific analytics"""
    module_id: str = Field(..., description="Module identifier")
    total_interactions: int = Field(..., description="Total interactions in module")
    total_time_spent: float = Field(..., description="Total time spent in minutes")
    avg_time_per_query: float = Field(..., description="Average time per query in seconds")
    satisfaction_rate: Optional[float] = Field(None, description="Satisfaction rate for this module")
    preferred_styles: List[Dict[str, Any]] = Field(default_factory=list, description="Preferred learning styles")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StudentDetailedAnalytics(BaseModel):
    """Model for detailed student analytics"""
    student_id: str = Field(..., description="Student identifier")
    summary: StudentAnalyticsSummary = Field(..., description="Analytics summary")
    module_analytics: List[ModuleAnalytics] = Field(default_factory=list, description="Per-module analytics")
    learning_style_usage: Dict[LearningStyleEnum, int] = Field(
        default_factory=dict, 
        description="Usage count per learning style"
    )
    daily_activity: List[Dict[str, Any]] = Field(default_factory=list, description="Daily activity data")
    weekly_activity: List[Dict[str, Any]] = Field(default_factory=list, description="Weekly activity data")
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas for improvement")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Personalized recommendations")


class StudentResponse(BaseModel):
    """Standard response model for student operations"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    student_profile: Optional[StudentProfile] = Field(None, description="Student profile data")


class AnxietyAssessment(BaseModel):
    """Model for anxiety assessment data"""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    student_id: str = Field(..., description="Student identifier")
    input_text: Optional[str] = Field(None, description="Text input for assessment")
    has_audio: bool = Field(default=False, description="Whether audio was provided")
    anxiety_level: str = Field(..., description="Detected anxiety level")
    confidence_score: Optional[float] = Field(None, description="Confidence in prediction")
    recommendations: List[str] = Field(default_factory=list, description="Wellness recommendations")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnxietyAssessmentCreate(BaseModel):
    """Model for creating anxiety assessment"""
    student_id: str = Field(..., description="Student identifier")
    input_text: Optional[str] = Field(None, description="Text input for assessment")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")


class StudentSession(BaseModel):
    """Model for student session data"""
    student_id: str = Field(..., description="Student identifier")
    session_token: str = Field(..., description="Session token")
    learning_style: LearningStyleEnum = Field(..., description="Current learning style")
    current_module: Optional[str] = Field(None, description="Current module")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }