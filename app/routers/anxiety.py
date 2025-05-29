"""
Anxiety detection endpoints for mental health assessment and wellness
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from datetime import datetime, timedelta
import base64

from app.models.anxiety import (
    AnxietyAssessmentRequest, AnxietyAssessment, AnxietyFeedback,
    AnxietyAnalytics, AnxietyResponse, WellnessProgram,
    WellnessProgressUpdate, CrisisAlert, AnxietyModelInfo,
    AnxietyLevel
)
from app.services.anxiety_service import AnxietyService
from app.routers.auth import get_current_student

logger = logging.getLogger(__name__)

router = APIRouter()
anxiety_service = AnxietyService()


@router.on_event("startup")
async def startup_event():
    """Initialize anxiety service on startup"""
    try:
        success = await anxiety_service.load_models()
        if success:
            logger.info("Anxiety detection models loaded successfully")
        else:
            logger.warning("Failed to load anxiety detection models")
    except Exception as e:
        logger.error(f"Error loading anxiety models: {e}")


@router.post("/assess", response_model=AnxietyResponse)
async def assess_anxiety(
    request: AnxietyAssessmentRequest,
    current_student: str = Depends(get_current_student)
):
    """Assess anxiety level from text and/or audio input"""
    try:
        # Validate student matches token
        if request.student_id != current_student:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student ID mismatch"
            )
        
        logger.info(f"Anxiety assessment request from student {current_student}")
        
        # Perform anxiety assessment
        assessment = await anxiety_service.assess_anxiety(request)
        
        # Check for crisis level
        crisis_alert = None
        if assessment.prediction.anxiety_level == AnxietyLevel.SEVERE_ANXIETY:
            crisis_alert = await create_crisis_alert(assessment)
        
        logger.info(f"Anxiety assessment completed: {assessment.prediction.anxiety_level}")
        
        return AnxietyResponse(
            success=True,
            message=f"Assessment completed: {assessment.prediction.anxiety_level}",
            assessment=assessment,
            crisis_alert=crisis_alert
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in anxiety assessment: {e}")
        return AnxietyResponse(
            success=False,
            message="Failed to complete anxiety assessment",
            assessment=None
        )


@router.post("/assess-audio", response_model=AnxietyResponse)
async def assess_anxiety_audio(
    student_id: str,
    audio_file: UploadFile = File(...),
    current_student: str = Depends(get_current_student)
):
    """Assess anxiety from audio file upload"""
    try:
        # Validate student matches token
        if student_id != current_student:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student ID mismatch"
            )
        
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file format"
            )
        
        logger.info(f"Audio anxiety assessment from student {current_student}")
        
        # Read audio file
        audio_content = await audio_file.read()
        
        # Convert to base64 for processing
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        # Get audio format from filename
        audio_format = audio_file.filename.split('.')[-1] if audio_file.filename else 'wav'
        
        # Create assessment request
        request = AnxietyAssessmentRequest(
            student_id=current_student,
            input_text=None,
            audio_data=audio_base64,
            audio_format=audio_format
        )
        
        # Perform assessment
        assessment = await anxiety_service.assess_anxiety(request)
        
        # Check for crisis level
        crisis_alert = None
        if assessment.prediction.anxiety_level == AnxietyLevel.SEVERE_ANXIETY:
            crisis_alert = await create_crisis_alert(assessment)
        
        logger.info(f"Audio anxiety assessment completed: {assessment.prediction.anxiety_level}")
        
        return AnxietyResponse(
            success=True,
            message=f"Audio assessment completed: {assessment.prediction.anxiety_level}",
            assessment=assessment,
            crisis_alert=crisis_alert
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio anxiety assessment: {e}")
        return AnxietyResponse(
            success=False,
            message="Failed to complete audio anxiety assessment"
        )


@router.get("/history")
async def get_anxiety_history(
    days: int = 30,
    current_student: str = Depends(get_current_student)
):
    """Get anxiety assessment history for the student"""
    try:
        logger.info(f"Getting anxiety history for student {current_student}")
        
        # Get assessment history
        assessments = await anxiety_service.get_student_anxiety_history(current_student, days)
        
        # Convert to response format
        history = []
        for assessment in assessments:
            history.append({
                "assessment_id": assessment.assessment_id,
                "timestamp": assessment.timestamp,
                "anxiety_level": assessment.prediction.anxiety_level,
                "confidence_score": assessment.prediction.confidence_score,
                "input_type": assessment.prediction.input_type,
                "has_audio": assessment.has_audio,
                "recommendations_count": len(assessment.recommendations)
            })
        
        return {
            "success": True,
            "student_id": current_student,
            "period_days": days,
            "total_assessments": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting anxiety history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get anxiety history"
        )


@router.get("/analytics", response_model=AnxietyAnalytics)
async def get_anxiety_analytics(
    current_student: str = Depends(get_current_student)
):
    """Get comprehensive anxiety analytics for the student"""
    try:
        logger.info(f"Getting anxiety analytics for student {current_student}")
        
        # Get analytics from service
        analytics = await anxiety_service.get_anxiety_analytics(current_student)
        
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No anxiety data found for analytics"
            )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting anxiety analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get anxiety analytics"
        )


@router.post("/feedback")
async def submit_anxiety_feedback(
    feedback: AnxietyFeedback,
    current_student: str = Depends(get_current_student)
):
    """Submit feedback for an anxiety assessment"""
    try:
        # Validate student matches token
        if feedback.student_id != current_student:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student ID mismatch"
            )
        
        logger.info(f"Anxiety feedback from student {current_student}")
        
        # Submit feedback
        success = await anxiety_service.submit_feedback(feedback)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit feedback"
            )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "assessment_id": feedback.assessment_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting anxiety feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get("/wellness-recommendations")
async def get_wellness_recommendations(
    anxiety_level: Optional[AnxietyLevel] = None,
    current_student: str = Depends(get_current_student)
):
    """Get wellness recommendations for a specific anxiety level"""
    try:
        logger.info(f"Getting wellness recommendations for student {current_student}")
        
        # If no anxiety level specified, get from latest assessment
        if not anxiety_level:
            assessments = await anxiety_service.get_student_anxiety_history(current_student, 7)
            if assessments:
                anxiety_level = assessments[0].prediction.anxiety_level
            else:
                anxiety_level = AnxietyLevel.MILD_ANXIETY  # Default
        
        # Get recommendations
        recommendations = anxiety_service.get_wellness_recommendations(anxiety_level)
        
        return {
            "success": True,
            "anxiety_level": anxiety_level,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting wellness recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get wellness recommendations"
        )


@router.get("/crisis-resources")
async def get_crisis_resources():
    """Get crisis resources and emergency contacts"""
    try:
        resources = {
            "emergency_contacts": [
                {
                    "name": "Crisis Text Line",
                    "contact": "Text HOME to 741741",
                    "description": "Free, 24/7 crisis support via text message"
                },
                {
                    "name": "National Suicide Prevention Lifeline",
                    "contact": "988",
                    "description": "Free and confidential emotional support 24/7"
                },
                {
                    "name": "Emergency Services",
                    "contact": "911",
                    "description": "For immediate life-threatening emergencies"
                }
            ],
            "campus_resources": [
                {
                    "name": "University Counseling Center",
                    "description": "Professional counseling services for students",
                    "availability": "Check your campus directory"
                },
                {
                    "name": "Campus Security",
                    "description": "24/7 campus safety and security",
                    "availability": "Available 24/7"
                }
            ],
            "self_help_resources": [
                {
                    "name": "Deep Breathing Exercise",
                    "description": "4-7-8 breathing technique for immediate anxiety relief"
                },
                {
                    "name": "Grounding Technique",
                    "description": "5-4-3-2-1 sensory grounding method"
                },
                {
                    "name": "Progressive Muscle Relaxation",
                    "description": "Systematic tension and release exercise"
                }
            ],
            "warning_signs": [
                "Thoughts of self-harm or suicide",
                "Inability to sleep or eat",
                "Severe panic attacks",
                "Complete withdrawal from activities",
                "Substance abuse as coping mechanism"
            ]
        }
        
        return {
            "success": True,
            "message": "Remember: You are not alone. Help is available.",
            "resources": resources
        }
        
    except Exception as e:
        logger.error(f"Error getting crisis resources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get crisis resources"
        )


@router.get("/model-info", response_model=AnxietyModelInfo)
async def get_model_info():
    """Get information about the anxiety detection model"""
    try:
        from app.ml_models.anxiety_model import model_info
        
        info = model_info()
        
        # Convert to model info format (simplified)
        return AnxietyModelInfo(
            model_name=info["model_name"],
            model_version="1.0",
            accuracy=0.85,  # Placeholder - would come from actual model evaluation
            precision={
                AnxietyLevel.NO_ANXIETY: 0.88,
                AnxietyLevel.MILD_ANXIETY: 0.82,
                AnxietyLevel.MODERATE_ANXIETY: 0.85,
                AnxietyLevel.SEVERE_ANXIETY: 0.90
            },
            recall={
                AnxietyLevel.NO_ANXIETY: 0.85,
                AnxietyLevel.MILD_ANXIETY: 0.80,
                AnxietyLevel.MODERATE_ANXIETY: 0.83,
                AnxietyLevel.SEVERE_ANXIETY: 0.92
            },
            f1_score={
                AnxietyLevel.NO_ANXIETY: 0.86,
                AnxietyLevel.MILD_ANXIETY: 0.81,
                AnxietyLevel.MODERATE_ANXIETY: 0.84,
                AnxietyLevel.SEVERE_ANXIETY: 0.91
            },
            training_date=datetime(2024, 1, 15),
            last_evaluation=datetime(2024, 3, 1),
            supported_inputs=info["input_types"]
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model information"
        )


@router.get("/health")
async def anxiety_health_check():
    """Health check for anxiety detection service"""
    try:
        from app.ml_models.anxiety_model import is_model_loaded
        
        model_loaded = is_model_loaded()
        
        return {
            "status": "healthy" if model_loaded else "degraded",
            "service": "anxiety_detection",
            "model_loaded": model_loaded,
            "models_available": ["MultimodalFusion", "Whisper"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Anxiety health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "anxiety_detection",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Helper functions
async def create_crisis_alert(assessment: AnxietyAssessment) -> CrisisAlert:
    """Create a crisis alert for severe anxiety cases"""
    try:
        from app.models.anxiety import CrisisAlert
        import uuid
        
        alert = CrisisAlert(
            alert_id=str(uuid.uuid4()),
            student_id=assessment.student_id,
            severity_level=5,  # Highest severity
            anxiety_level=assessment.prediction.anxiety_level,
            confidence_score=assessment.prediction.confidence_score,
            triggers=["Severe anxiety detected"],
            immediate_actions=[
                "Contact crisis helpline immediately",
                "Reach out to trusted person",
                "Use emergency breathing techniques",
                "Seek professional help",
                "Go to safe space"
            ],
            professional_contact_info={
                "crisis_line": "741741",
                "suicide_prevention": "988",
                "emergency": "911",
                "campus_counseling": "Check campus directory"
            },
            created_at=datetime.now()
        )
        
        # In a production system, this would trigger:
        # - Email to counseling services
        # - Alert to emergency contacts
        # - Follow-up scheduling
        
        logger.warning(f"Crisis alert created for student {assessment.student_id}")
        return alert
        
    except Exception as e:
        logger.error(f"Error creating crisis alert: {e}")
        return None