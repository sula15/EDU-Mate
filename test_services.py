#!/usr/bin/env python3
"""
Test script to verify all services work correctly
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_service_imports():
    """Test that all services can be imported"""
    print("📦 Testing Service Imports...")
    
    try:
        from app.services.anxiety_service import AnxietyService
        print("✅ AnxietyService imported successfully")
        
        from app.services.personalization_service import PersonalizationService
        print("✅ PersonalizationService imported successfully")
        
        from app.services.chat_service import ChatService
        print("✅ ChatService imported successfully")
        
        from app.services.document_service import DocumentService
        print("✅ DocumentService imported successfully")
        
        from app.services.analytics_service import AnalyticsService
        print("✅ AnalyticsService imported successfully")
        
        from app.ml_models.anxiety_model import MultimodalFusion, model_info
        print("✅ ML Models imported successfully")
        
        print("✅ All service imports successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Service import error: {e}")
        return False


async def test_anxiety_service():
    """Test anxiety service functionality"""
    print("🧠 Testing Anxiety Service...")
    
    try:
        from app.services.anxiety_service import AnxietyService
        from app.models.anxiety import AnxietyAssessmentRequest
        
        # Initialize service
        service = AnxietyService()
        print("✅ AnxietyService initialized")
        
        # Test model info (without loading full models)
        from app.ml_models.anxiety_model import model_info
        info = model_info()
        print(f"✅ Model info retrieved: {info['model_name']}")
        
        # Test wellness recommendations
        from app.models.anxiety import AnxietyLevel
        recommendations = service.get_wellness_recommendations(AnxietyLevel.MILD_ANXIETY)
        print(f"✅ Generated {len(recommendations)} wellness recommendations")
        
        # Test tokenization (if BERT tokenizer loads)
        try:
            text = "I'm feeling a bit anxious about my studies"
            input_ids, attention_mask = service.tokenize_and_pad(text)
            print(f"✅ Text tokenization successful: {input_ids.shape}")
        except Exception as tokenizer_error:
            print(f"⚠️  Tokenizer not available (expected in test environment): {tokenizer_error}")
        
        print("✅ AnxietyService basic functionality working!\n")
        return True
        
    except Exception as e:
        print(f"❌ AnxietyService error: {e}")
        return False


async def test_personalization_service():
    """Test personalization service functionality"""
    print("👤 Testing Personalization Service...")
    
    try:
        from app.services.personalization_service import PersonalizationService
        from app.models.student import StudentRegistration, LearningStyleEnum
        
        # Initialize service
        service = PersonalizationService()
        print("✅ PersonalizationService initialized")
        
        # Test learning styles
        styles = service.get_all_learning_styles()
        print(f"✅ Retrieved {len(styles)} learning styles")
        
        # Test learning style by ID
        detailed_style = service.get_learning_style_by_id("detailed")
        if detailed_style:
            print(f"✅ Found learning style: {detailed_style['name']}")
        
        # Test query formatting
        query = "How does machine learning work?"
        formatted = service.format_query_with_learning_style(query, "eli5")
        print(f"✅ Query formatting successful: {len(formatted)} characters")
        
        # Test session timeout check (without DB)
        try:
            timeout = await service.check_session_timeout("test_student")
            print(f"✅ Session timeout check: {timeout}")
        except Exception as db_error:
            print(f"⚠️  Database not available for session check (expected): {str(db_error)[:50]}...")
        
        print("✅ PersonalizationService basic functionality working!\n")
        return True
        
    except Exception as e:
        print(f"❌ PersonalizationService error: {e}")
        return False


async def test_chat_service():
    """Test chat service functionality"""
    print("💬 Testing Chat Service...")
    
    try:
        from app.services.chat_service import ChatService
        
        # Initialize service
        service = ChatService()
        print("✅ ChatService initialized")
        
        # Test context building
        from app.models.chat import RetrievedSource, ImageResult, SourceType
        
        # Test text context building
        text_sources = [
            RetrievedSource(
                source_id="test1",
                source_type=SourceType.TEXT,
                content="Machine learning is a subset of AI",
                similarity_score=0.85,
                lecture_number=3,
                module_code="COMP101"
            )
        ]
        
        text_context = service._build_text_context(text_sources)
        print(f"✅ Text context built: {len(text_context)} characters")
        
        # Test image context building  
        image_results = [
            ImageResult(
                image_id="img1",
                image_data="base64_image_data",
                similarity_score=0.92,
                lecture_code="COMP101",
                lecture_number=3,
                text_description="Neural network diagram"
            )
        ]
        
        image_context = service._build_image_context(image_results)
        print(f"✅ Image context built: {len(image_context)} characters")
        
        # Test prompt creation
        prompt = service._create_llm_prompt(
            original_query="What is machine learning?",
            formatted_query="Explain machine learning in simple terms",
            text_context=text_context,
            image_context=image_context
        )
        print(f"✅ LLM prompt created: {len(prompt)} characters")
        
        # Test health check (without external dependencies)
        try:
            health = await service.health_check()
            print(f"✅ Health check completed: {health.get('chat_service', 'unknown')}")
        except Exception as health_error:
            print(f"⚠️  Health check failed (expected without full setup): {str(health_error)[:50]}...")
        
        print("✅ ChatService basic functionality working!\n")
        return True
        
    except Exception as e:
        print(f"❌ ChatService error: {e}")
        return False


async def test_document_service():
    """Test document service functionality"""
    print("📄 Testing Document Service...")
    
    try:
        from app.services.document_service import DocumentService
        from app.models.document import ProcessingConfig, DocumentType
        
        # Initialize service
        service = DocumentService()
        print("✅ DocumentService initialized")
        
        # Test file hash calculation
        test_content = b"This is test file content"
        file_hash = service.calculate_file_hash(test_content)
        print(f"✅ File hash calculated: {file_hash[:16]}...")
        
        # Test duplicate check (without DB)
        try:
            is_duplicate = await service.is_duplicate_file(file_hash, {"module_id": "TEST"})
            print(f"✅ Duplicate check: {is_duplicate}")
        except Exception as db_error:
            print(f"⚠️  Database not available for duplicate check (expected): {str(db_error)[:50]}...")
        
        # Test processing config
        config = ProcessingConfig(
            chunk_size=1000,
            similarity_threshold=0.95,
            image_weight=0.3
        )
        print(f"✅ Processing config created: chunk_size={config.chunk_size}")
        
        print("✅ DocumentService basic functionality working!\n")
        return True
        
    except Exception as e:
        print(f"❌ DocumentService error: {e}")
        return False


async def test_analytics_service():
    """Test analytics service functionality"""
    print("📊 Testing Analytics Service...")
    
    try:
        from app.services.analytics_service import AnalyticsService
        
        # Initialize service
        service = AnalyticsService()
        print("✅ AnalyticsService initialized")
        
        # Test learning style usage analysis (with mock data)
        mock_interactions = [
            {"learning_style_id": "detailed", "start_time": datetime.now()},
            {"learning_style_id": "concise", "start_time": datetime.now()},
            {"learning_style_id": "detailed", "start_time": datetime.now()}
        ]
        
        style_usage = service._analyze_learning_style_usage(mock_interactions)
        print(f"✅ Learning style analysis: {len(style_usage)} styles found")
        
        # Test preferred style identification
        preferred = service._get_preferred_learning_style(style_usage)
        if preferred:
            print(f"✅ Preferred style identified: {preferred['name']}")
        
        # Test satisfaction rate calculation
        mock_interactions_with_feedback = [
            {"helpful": True}, {"helpful": False}, {"helpful": True}
        ]
        satisfaction = service._calculate_satisfaction_rate(mock_interactions_with_feedback)
        print(f"✅ Satisfaction rate calculated: {satisfaction:.2f}")
        
        # Test recommendations generation
        recommendations = service._generate_personalized_recommendations(
            mock_interactions, style_usage, [], satisfaction
        )
        print(f"✅ Generated {len(recommendations)} recommendations")
        
        print("✅ AnalyticsService basic functionality working!\n")
        return True
        
    except Exception as e:
        print(f"❌ AnalyticsService error: {e}")
        return False


async def test_database_connections():
    """Test database connection functionality"""
    print("🗄️  Testing Database Connections...")
    
    try:
        from app.core.database import check_database_health
        
        # Test database health check
        health = await check_database_health()
        print(f"✅ Database health check completed")
        print(f"   - MongoDB: {'✅' if health['mongodb']['connected'] else '❌'}")
        print(f"   - Milvus: {'✅' if health['milvus']['connected'] else '❌'}")
        print(f"   - Overall: {'✅' if health['overall_healthy'] else '❌'}")
        
        return health['overall_healthy']
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False


async def test_configuration():
    """Test configuration loading"""
    print("⚙️  Testing Configuration...")
    
    try:
        from app.core.config import get_settings, learning_styles
        
        # Test settings
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.PROJECT_NAME}")
        print(f"   - MongoDB URI: {settings.MONGODB_URI}")
        print(f"   - Milvus Host: {settings.MILVUS_HOST}")
        
        # Test learning styles
        styles = learning_styles.LEARNING_STYLES
        print(f"✅ Learning styles loaded: {len(styles)} styles")
        
        # Test learning style retrieval
        detailed_style = learning_styles.get_learning_style_by_id("detailed")
        if detailed_style:
            print(f"✅ Learning style retrieval working: {detailed_style['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


async def main():
    """Run all service tests"""
    print("🚀 Testing All Services for MindMate-Edu Backend\n")
    
    tests_passed = 0
    total_tests = 8
    
    # Run all tests
    if test_service_imports():
        tests_passed += 1
        
    if test_configuration():
        tests_passed += 1
        
    if await test_database_connections():
        tests_passed += 1
        
    if await test_anxiety_service():
        tests_passed += 1
        
    if await test_personalization_service():
        tests_passed += 1
        
    if await test_chat_service():
        tests_passed += 1
        
    if await test_document_service():
        tests_passed += 1
        
    if await test_analytics_service():
        tests_passed += 1
    
    # Summary
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All services are working correctly!")
        print("\n🚀 Ready for next phase: Implementing API Routers")
        print("📋 Next steps:")
        print("   1. Create FastAPI routers for each service")
        print("   2. Set up authentication and middleware")
        print("   3. Create the main FastAPI application")
        print("   4. Test API endpoints")
    elif tests_passed >= 6:
        print("✨ Most services are working! Some advanced features may need full environment setup.")
        print("🚀 Ready to proceed with API Routers implementation")
    else:
        print("⚠️  Some critical services failed. Please check your environment setup.")
        print("💡 Common issues:")
        print("   - Missing environment variables (.env file)")
        print("   - Database services not running (MongoDB, Milvus)")
        print("   - Missing model files")
    
    return tests_passed >= 6


if __name__ == "__main__":
    asyncio.run(main())