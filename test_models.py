#!/usr/bin/env python3
"""
Test script to verify all Pydantic models work correctly
"""

import sys
import os
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_student_models():
    """Test student-related models"""
    print("🎓 Testing Student Models...")
    
    try:
        from app.models.student import (
            StudentRegistration, StudentLogin, StudentProfile, 
            StudentInteraction, StudentFeedback, LearningStyleEnum,
            StudentAnalyticsSummary, AnxietyAssessment
        )
        
        # Test StudentRegistration
        registration = StudentRegistration(
            student_id="STU001",
            name="John Doe",
            email="john.doe@example.com",
            default_learning_style=LearningStyleEnum.DETAILED
        )
        print(f"✅ StudentRegistration: {registration.student_id} - {registration.name}")
        
        # Test StudentLogin
        login = StudentLogin(
            student_id="STU001",
            name="John Doe"
        )
        print(f"✅ StudentLogin: {login.student_id}")
        
        # Test StudentProfile
        profile = StudentProfile(
            student_id="STU001",
            name="John Doe",
            email="john.doe@example.com",
            default_learning_style=LearningStyleEnum.DETAILED,
            created_at=datetime.now(),
            last_active=datetime.now(),
            total_interactions=5
        )
        print(f"✅ StudentProfile: {profile.student_id} with {profile.total_interactions} interactions")
        
        # Test StudentFeedback
        feedback = StudentFeedback(
            interaction_id="INT001",
            helpful=True,
            rating=4
        )
        print(f"✅ StudentFeedback: Rating {feedback.rating}/5")
        
        print("✅ All student models working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Student models error: {e}")
        return False


def test_chat_models():
    """Test chat-related models"""
    print("💬 Testing Chat Models...")
    
    try:
        from app.models.chat import (
            ChatMessage, ChatQuery, ChatResponse, RetrievedSource,
            ImageResult, SearchRequest, MessageRole, SourceType
        )
        from app.models.student import LearningStyleEnum
        
        # Test ChatMessage
        message = ChatMessage(
            role=MessageRole.USER,
            content="How does TCP work?",
            timestamp=datetime.now()
        )
        print(f"✅ ChatMessage: {message.role} - {message.content[:20]}...")
        
        # Test ChatQuery
        query = ChatQuery(
            student_id="STU001",
            message="Explain machine learning",
            learning_style=LearningStyleEnum.ELI5,
            include_images=True
        )
        print(f"✅ ChatQuery: {query.message[:30]}... (style: {query.learning_style})")
        
        # Test RetrievedSource
        source = RetrievedSource(
            source_id="SRC001",
            source_type=SourceType.TEXT,
            content="Machine learning is a subset of AI...",
            similarity_score=0.85,
            lecture_number=3
        )
        print(f"✅ RetrievedSource: {source.source_type} with score {source.similarity_score}")
        
        # Test ImageResult
        image_result = ImageResult(
            image_id="IMG001",
            image_data="base64_encoded_image_data_here",
            similarity_score=0.92,
            lecture_code="IT3061",
            lecture_number=5
        )
        print(f"✅ ImageResult: {image_result.lecture_code} L{image_result.lecture_number} (score: {image_result.similarity_score})")
        
        print("✅ All chat models working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Chat models error: {e}")
        return False


def test_document_models():
    """Test document-related models"""
    print("📄 Testing Document Models...")
    
    try:
        from app.models.document import (
            DocumentMetadata, DocumentUpload, ProcessingResult,
            DocumentType, SourceType, ProcessingType, ProcessingStatus
        )
        
        # Test DocumentMetadata
        metadata = DocumentMetadata(
            module_id="MDPCC",
            lecture_code="IT3061", 
            lecture_number=3,
            lecture_title="Introduction to Neural Networks",
            source_type=SourceType.LECTURE
        )
        print(f"✅ DocumentMetadata: {metadata.module_id} - {metadata.lecture_title}")
        
        # Test DocumentUpload
        upload = DocumentUpload(
            filename="lecture_3_neural_networks.pdf",
            file_size=2048576,  # 2MB
            file_type=DocumentType.PDF,
            metadata=metadata,
            processing_type=ProcessingType.TEXT_AND_IMAGES
        )
        print(f"✅ DocumentUpload: {upload.filename} ({upload.file_size} bytes)")
        
        # Test ProcessingResult
        result = ProcessingResult(
            document_id="DOC001",
            status=ProcessingStatus.COMPLETED,
            text_chunks_created=25,
            total_images_found=8,
            unique_images_processed=6,
            images_stored=6,
            processing_time=45.2,
            started_at=datetime.now()
        )
        print(f"✅ ProcessingResult: {result.status} - {result.text_chunks_created} chunks, {result.unique_images_processed} images")
        
        print("✅ All document models working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Document models error: {e}")
        return False


def test_anxiety_models():
    """Test anxiety-related models"""
    print("🧠 Testing Anxiety Models...")
    
    try:
        from app.models.anxiety import (
            AnxietyAssessmentRequest, AnxietyPrediction, WellnessRecommendation,
            AnxietyAssessment, AnxietyLevel, InputType, WellnessRecommendationType
        )
        
        # Test AnxietyAssessmentRequest
        request = AnxietyAssessmentRequest(
            student_id="STU001",
            input_text="I'm feeling overwhelmed with my studies...",
            audio_data=None
        )
        print(f"✅ AnxietyAssessmentRequest: {request.student_id} - {request.input_text[:30]}...")
        
        # Test AnxietyPrediction
        prediction = AnxietyPrediction(
            anxiety_level=AnxietyLevel.MILD_ANXIETY,
            confidence_score=0.78,
            input_type=InputType.TEXT_ONLY,
            processing_time=1.2
        )
        print(f"✅ AnxietyPrediction: {prediction.anxiety_level} (confidence: {prediction.confidence_score})")
        
        # Test WellnessRecommendation
        recommendation = WellnessRecommendation(
            recommendation_id="REC001",
            type=WellnessRecommendationType.BREATHING,
            title="Deep Breathing Exercise",
            description="A simple breathing technique to reduce anxiety",
            instructions=["Sit comfortably", "Breathe in for 4 counts", "Hold for 4 counts", "Exhale for 6 counts"],
            duration_minutes=5,
            difficulty_level=1
        )
        print(f"✅ WellnessRecommendation: {recommendation.title} ({recommendation.duration_minutes} min)")
        
        # Test AnxietyAssessment
        assessment = AnxietyAssessment(
            assessment_id="ASS001",
            student_id="STU001",
            prediction=prediction,
            recommendations=[recommendation],
            input_text=request.input_text,
            has_audio=False,
            timestamp=datetime.now()
        )
        print(f"✅ AnxietyAssessment: {assessment.assessment_id} with {len(assessment.recommendations)} recommendations")
        
        print("✅ All anxiety models working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Anxiety models error: {e}")
        return False


def test_model_serialization():
    """Test JSON serialization/deserialization"""
    print("🔄 Testing Model Serialization...")
    
    try:
        from app.models.student import StudentProfile, LearningStyleEnum
        import json
        
        # Create a profile
        profile = StudentProfile(
            student_id="STU001",
            name="Jane Doe",
            email="jane.doe@example.com",
            default_learning_style=LearningStyleEnum.VISUAL,
            created_at=datetime.now(),
            last_active=datetime.now(),
            total_interactions=10
        )
        
        # Test serialization
        json_data = profile.model_dump_json()
        print(f"✅ Serialization: {len(json_data)} characters")
        
        # Test deserialization
        parsed_data = json.loads(json_data)
        reconstructed = StudentProfile.model_validate(parsed_data)
        print(f"✅ Deserialization: {reconstructed.student_id} - {reconstructed.name}")
        
        print("✅ Serialization working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Serialization error: {e}")
        return False


def main():
    """Run all model tests"""
    print("🚀 Testing All Pydantic Models\n")
    
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    if test_student_models():
        tests_passed += 1
        
    if test_chat_models():
        tests_passed += 1
        
    if test_document_models():
        tests_passed += 1
        
    if test_anxiety_models():
        tests_passed += 1
        
    if test_model_serialization():
        tests_passed += 1
    
    # Summary
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All Pydantic models are working correctly!")
        print("\n🚀 Ready for next phase: Implementing Services (business logic)")
        print("📋 Next: We'll adapt your existing code into service classes")
    else:
        print("⚠️  Some model tests failed. Please check the errors above.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    main()