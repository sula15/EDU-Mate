"""
Pydantic models for anxiety detection and wellness features
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class AnxietyLevel(str, Enum):
    """Enum for anxiety levels"""
    NO_ANXIETY = "No Anxiety"
    MILD_ANXIETY = "Mild Anxiety"
    MODERATE_ANXIETY = "Moderate Anxiety"
    SEVERE_ANXIETY = "Severe Anxiety"


class InputType(str, Enum):
    """Enum for input types"""
    TEXT_ONLY = "text_only"
    AUDIO_ONLY = "audio_only"
    MULTIMODAL = "multimodal"


class WellnessRecommendationType(str, Enum):
    """Enum for wellness recommendation types"""
    BREATHING = "breathing"
    MINDFULNESS = "mindfulness"
    PHYSICAL = "physical"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    PROFESSIONAL = "professional"


class AnxietyAssessmentRequest(BaseModel):
    """Model for anxiety assessment requests"""
    student_id: str = Field(..., description="Student identifier")
    input_text: Optional[str] = Field(None, description="Text input for assessment")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_format: Optional[str] = Field(None, description="Audio format (wav, mp3, m4a)")
    session_context: Optional[Dict[str, Any]] = Field(None, description="Additional session context")


class AnxietyPrediction(BaseModel):
    """Model for anxiety prediction results"""
    anxiety_level: AnxietyLevel = Field(..., description="Predicted anxiety level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    input_type: InputType = Field(..., description="Type of input used for prediction")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(default="1.0", description="Model version used")


class WellnessRecommendation(BaseModel):
    """Model for wellness recommendations"""
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    type: WellnessRecommendationType = Field(..., description="Type of recommendation")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    instructions: List[str] = Field(..., description="Step-by-step instructions")
    duration_minutes: Optional[int] = Field(None, description="Recommended duration in minutes")
    difficulty_level: int = Field(default=1, ge=1, le=5, description="Difficulty level (1=easy, 5=hard)")
    effectiveness_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Effectiveness rating")
    tags: List[str] = Field(default_factory=list, description="Recommendation tags")


class AnxietyAssessment(BaseModel):
    """Model for complete anxiety assessment"""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    student_id: str = Field(..., description="Student identifier")
    prediction: AnxietyPrediction = Field(..., description="Anxiety prediction results")
    recommendations: List[WellnessRecommendation] = Field(..., description="Personalized recommendations")
    input_text: Optional[str] = Field(None, description="Original text input")
    has_audio: bool = Field(default=False, description="Whether audio was provided")
    transcript: Optional[str] = Field(None, description="Audio transcript if available")
    timestamp: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnxietyFeedback(BaseModel):
    """Model for anxiety assessment feedback"""
    assessment_id: str = Field(..., description="Assessment identifier")
    student_id: str = Field(..., description="Student identifier")
    accuracy_rating: int = Field(..., ge=1, le=5, description="Accuracy rating (1=very inaccurate, 5=very accurate)")
    helpfulness_rating: int = Field(..., ge=1, le=5, description="Helpfulness rating (1=not helpful, 5=very helpful)")
    used_recommendations: List[str] = Field(default_factory=list, description="IDs of recommendations that were tried")
    recommendation_effectiveness: Dict[str, int] = Field(
        default_factory=dict, 
        description="Effectiveness rating for each recommendation (1-5)"
    )
    additional_feedback: Optional[str] = Field(None, description="Additional text feedback")
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnxietyTrend(BaseModel):
    """Model for anxiety trend data"""
    student_id: str = Field(..., description="Student identifier")
    date: datetime = Field(..., description="Date of assessment")
    anxiety_level: AnxietyLevel = Field(..., description="Anxiety level on this date")
    confidence_score: float = Field(..., description="Prediction confidence")
    assessment_count: int = Field(default=1, description="Number of assessments on this date")
    avg_confidence: float = Field(..., description="Average confidence for the day")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnxietyAnalytics(BaseModel):
    """Model for anxiety analytics"""
    student_id: str = Field(..., description="Student identifier")
    total_assessments: int = Field(..., description="Total number of assessments")
    assessment_period_days: int = Field(..., description="Period covered in days")
    
    # Level distribution
    level_distribution: Dict[AnxietyLevel, int] = Field(..., description="Distribution of anxiety levels")
    level_percentages: Dict[AnxietyLevel, float] = Field(..., description="Percentage distribution")
    
    # Trends
    trend_data: List[AnxietyTrend] = Field(..., description="Daily trend data")
    overall_trend: str = Field(..., description="Overall trend (improving/stable/declining)")
    
    # Patterns
    peak_anxiety_times: List[str] = Field(default_factory=list, description="Times when anxiety is typically highest")
    low_anxiety_times: List[str] = Field(default_factory=list, description="Times when anxiety is typically lowest")
    
    # Recommendations effectiveness
    most_effective_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Most effective recommendation types"
    )
    
    # Risk indicators
    risk_level: str = Field(..., description="Overall risk level (low/medium/high)")
    intervention_recommended: bool = Field(..., description="Whether professional intervention is recommended")
    
    last_updated: datetime = Field(default_factory=datetime.now, description="Analytics last updated")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CrisisAlert(BaseModel):
    """Model for crisis alerts"""
    alert_id: str = Field(..., description="Unique alert identifier")
    student_id: str = Field(..., description="Student identifier")
    severity_level: int = Field(..., ge=1, le=5, description="Severity level (1=low, 5=critical)")
    anxiety_level: AnxietyLevel = Field(..., description="Detected anxiety level")
    confidence_score: float = Field(..., description="Prediction confidence")
    triggers: List[str] = Field(default_factory=list, description="Identified triggers")
    immediate_actions: List[str] = Field(..., description="Immediate recommended actions")
    professional_contact_info: Dict[str, str] = Field(
        default_factory=dict, 
        description="Professional contact information"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Alert creation timestamp")
    acknowledged: bool = Field(default=False, description="Whether alert has been acknowledged")
    resolved: bool = Field(default=False, description="Whether crisis has been resolved")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WellnessProgram(BaseModel):
    """Model for personalized wellness programs"""
    program_id: str = Field(..., description="Unique program identifier")
    student_id: str = Field(..., description="Student identifier")
    program_name: str = Field(..., description="Program name")
    description: str = Field(..., description="Program description")
    duration_days: int = Field(..., description="Program duration in days")
    daily_recommendations: List[WellnessRecommendation] = Field(
        ..., 
        description="Daily recommendations"
    )
    goals: List[str] = Field(..., description="Program goals")
    progress_metrics: List[str] = Field(..., description="Metrics to track progress")
    created_at: datetime = Field(default_factory=datetime.now, description="Program creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Program start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Program completion timestamp")
    is_active: bool = Field(default=True, description="Whether program is active")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WellnessProgressUpdate(BaseModel):
    """Model for wellness progress updates"""
    program_id: str = Field(..., description="Program identifier")
    student_id: str = Field(..., description="Student identifier")
    day_number: int = Field(..., ge=1, description="Day number in the program")
    completed_recommendations: List[str] = Field(
        ..., 
        description="IDs of completed recommendations"
    )
    mood_rating: Optional[int] = Field(None, ge=1, le=10, description="Mood rating (1-10)")
    anxiety_level: Optional[AnxietyLevel] = Field(None, description="Self-reported anxiety level")
    notes: Optional[str] = Field(None, description="Progress notes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnxietyResponse(BaseModel):
    """Standard response model for anxiety operations"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    assessment: Optional[AnxietyAssessment] = Field(None, description="Assessment results")
    analytics: Optional[AnxietyAnalytics] = Field(None, description="Analytics data")
    crisis_alert: Optional[CrisisAlert] = Field(None, description="Crisis alert if applicable")


class AnxietyModelInfo(BaseModel):
    """Model for anxiety detection model information"""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    accuracy: float = Field(..., description="Model accuracy on test set")
    precision: Dict[AnxietyLevel, float] = Field(..., description="Precision scores per class")
    recall: Dict[AnxietyLevel, float] = Field(..., description="Recall scores per class")
    f1_score: Dict[AnxietyLevel, float] = Field(..., description="F1 scores per class")
    training_date: datetime = Field(..., description="Model training date")
    last_evaluation: datetime = Field(..., description="Last evaluation date")
    supported_inputs: List[InputType] = Field(..., description="Supported input types")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemWellnessMetrics(BaseModel):
    """Model for system-wide wellness metrics"""
    total_assessments: int = Field(..., description="Total assessments conducted")
    total_students: int = Field(..., description="Total students assessed")
    avg_assessments_per_student: float = Field(..., description="Average assessments per student")
    anxiety_distribution: Dict[AnxietyLevel, int] = Field(..., description="System-wide anxiety distribution")
    high_risk_students: int = Field(..., description="Number of high-risk students")
    intervention_success_rate: float = Field(..., description="Intervention success rate")
    most_effective_interventions: List[Dict[str, Any]] = Field(
        ..., 
        description="Most effective intervention types"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }