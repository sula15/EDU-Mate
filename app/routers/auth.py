"""
Authentication endpoints for student management
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.models.student import (
    StudentRegistration, StudentLogin, StudentProfile, 
    StudentProfileUpdate, StudentResponse, LearningStyleEnum,
    StudentSession
)
from app.services.personalization_service import PersonalizationService
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()
personalization_service = PersonalizationService()


# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        student_id: str = payload.get("sub")
        if student_id is None:
            raise JWTError("Invalid token")
        return {"student_id": student_id, "exp": payload.get("exp")}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_student(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current authenticated student"""
    token_data = verify_token(credentials.credentials)
    student_id = token_data["student_id"]
    
    # Check if student exists
    profile = await personalization_service.get_student_profile(student_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    # Update last activity
    await personalization_service.update_last_activity(student_id)
    
    return student_id


@router.post("/register", response_model=StudentResponse)
async def register_student(registration: StudentRegistration):
    """Register a new student"""
    try:
        logger.info(f"Registering new student: {registration.student_id}")
        
        # Check if student already exists
        existing_profile = await personalization_service.get_student_profile(registration.student_id)
        if existing_profile:
            return StudentResponse(
                success=False,
                message="Student already registered",
                student_profile=existing_profile
            )
        
        # Create new student profile
        profile = await personalization_service.create_student_profile(registration)
        
        logger.info(f"Successfully registered student: {registration.student_id}")
        return StudentResponse(
            success=True,
            message="Student registered successfully",
            student_profile=profile
        )
        
    except Exception as e:
        logger.error(f"Error registering student: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register student"
        )


@router.post("/login", response_model=dict)
async def login_student(login: StudentLogin):
    """Login student and return access token"""
    try:
        logger.info(f"Student login attempt: {login.student_id}")
        
        # Get or create student profile
        profile = await personalization_service.get_student_profile(login.student_id)
        
        if not profile:
            # Create profile if it doesn't exist (simplified registration)
            registration = StudentRegistration(
                student_id=login.student_id,
                name=login.name,
                default_learning_style=LearningStyleEnum.DETAILED
            )
            profile = await personalization_service.create_student_profile(registration)
            logger.info(f"Created new profile during login: {login.student_id}")
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": login.student_id, "name": login.name},
            expires_delta=access_token_expires
        )
        
        # Update last activity
        await personalization_service.update_last_activity(login.student_id)
        
        logger.info(f"Successfully logged in student: {login.student_id}")
        return {
            "success": True,
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "student_profile": profile
        }
        
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/profile", response_model=StudentProfile)
async def get_profile(current_student: str = Depends(get_current_student)):
    """Get current student profile"""
    try:
        profile = await personalization_service.get_student_profile(current_student)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found"
            )
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile"
        )


@router.put("/profile", response_model=StudentResponse)
async def update_profile(
    update_data: StudentProfileUpdate,
    current_student: str = Depends(get_current_student)
):
    """Update student profile"""
    try:
        updated_profile = await personalization_service.update_student_profile(
            current_student, update_data
        )
        
        if not updated_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed to update profile"
            )
        
        return StudentResponse(
            success=True,
            message="Profile updated successfully",
            student_profile=updated_profile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.put("/learning-style")
async def update_learning_style(
    learning_style: LearningStyleEnum,
    current_student: str = Depends(get_current_student)
):
    """Update student's preferred learning style"""
    try:
        success = await personalization_service.update_learning_style(
            current_student, learning_style
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Failed to update learning style"
            )
        
        return {
            "success": True,
            "message": f"Learning style updated to {learning_style}",
            "learning_style": learning_style
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating learning style: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update learning style"
        )


@router.get("/learning-styles")
async def get_learning_styles():
    """Get all available learning styles"""
    try:
        styles = personalization_service.get_all_learning_styles()
        return {
            "success": True,
            "learning_styles": styles
        }
        
    except Exception as e:
        logger.error(f"Error getting learning styles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning styles"
        )


@router.post("/refresh-token")
async def refresh_token(current_student: str = Depends(get_current_student)):
    """Refresh access token"""
    try:
        # Get student profile
        profile = await personalization_service.get_student_profile(current_student)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": current_student, "name": profile.name},
            expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )


@router.post("/logout")
async def logout(current_student: str = Depends(get_current_student)):
    """Logout student (invalidate token)"""
    try:
        # In a more complete implementation, you'd maintain a blacklist of tokens
        # For now, we just acknowledge the logout
        
        logger.info(f"Student logged out: {current_student}")
        return {
            "success": True,
            "message": "Logged out successfully"
        }
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/session-status")
async def get_session_status(current_student: str = Depends(get_current_student)):
    """Get current session status"""
    try:
        profile = await personalization_service.get_student_profile(current_student)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found"
            )
        
        # Check session timeout
        is_timeout = await personalization_service.check_session_timeout(current_student)
        
        return {
            "success": True,
            "student_id": current_student,
            "student_name": profile.name,
            "last_active": profile.last_active,
            "session_active": not is_timeout,
            "learning_style": profile.default_learning_style
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session status"
        )


# Health check for auth service
@router.get("/health")
async def auth_health():
    """Health check for authentication service"""
    try:
        # Test database connection by trying to get learning styles
        styles = personalization_service.get_all_learning_styles()
        
        return {
            "status": "healthy",
            "service": "authentication",
            "learning_styles_available": len(styles),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }