#!/usr/bin/env python3
"""
Test script to verify core configuration and database setup
"""

import sys
import os
import asyncio

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import get_settings, learning_styles
from app.core.database import db_manager, check_database_health


def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    
    try:
        settings = get_settings()
        print(f"✅ Settings loaded successfully")
        print(f"   - Project: {settings.PROJECT_NAME}")
        print(f"   - Debug: {settings.DEBUG}")
        print(f"   - MongoDB URI: {settings.MONGODB_URI}")
        print(f"   - Milvus Host: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        
        # Test learning styles
        print(f"✅ Learning styles loaded: {len(learning_styles.LEARNING_STYLES)} styles")
        for style in learning_styles.LEARNING_STYLES:
            print(f"   - {style['name']} ({style['id']})")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_mongodb():
    """Test MongoDB connection"""
    print("\n📊 Testing MongoDB Connection...")
    
    try:
        client = db_manager.connect_mongodb()
        db = db_manager.get_mongo_db()
        
        # Test basic operations
        collections = db.list_collection_names()
        print(f"✅ MongoDB connected successfully")
        print(f"   - Database: {db.name}")
        print(f"   - Collections: {len(collections)}")
        
        # Test collection access
        profiles_collection = db_manager.get_mongo_collection("test_collection")
        print(f"✅ Collection access working")
        
        return True
    except Exception as e:
        print(f"❌ MongoDB error: {e}")
        return False


def test_milvus():
    """Test Milvus connection"""
    print("\n🔍 Testing Milvus Connection...")
    
    try:
        success = db_manager.connect_milvus()
        if success:
            print(f"✅ Milvus connected successfully")
            
            # Try to get collections
            from pymilvus import utility
            collections = utility.list_collections()
            print(f"   - Available collections: {len(collections)}")
            for collection in collections:
                print(f"     * {collection}")
            
            return True
        else:
            print(f"❌ Failed to connect to Milvus")
            return False
    except Exception as e:
        print(f"❌ Milvus error: {e}")
        return False


async def test_database_health():
    """Test database health check"""
    print("\n🏥 Testing Database Health Check...")
    
    try:
        health = await check_database_health()
        print(f"✅ Health check completed")
        print(f"   - MongoDB: {'✅' if health['mongodb']['connected'] else '❌'}")
        print(f"   - Milvus: {'✅' if health['milvus']['connected'] else '❌'}")
        print(f"   - Overall: {'✅' if health['overall_healthy'] else '❌'}")
        
        return health['overall_healthy']
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Core Setup for MindMate-Edu Backend\n")
    
    tests_passed = 0
    total_tests = 4
    
    # Test configuration
    if test_configuration():
        tests_passed += 1
    
    # Test MongoDB
    if test_mongodb():
        tests_passed += 1
    
    # Test Milvus
    if test_milvus():
        tests_passed += 1
    
    # Test health check
    if asyncio.run(test_database_health()):
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Core setup is working correctly.")
        print("\n🚀 Ready for next phase: Implementing Pydantic models")
    else:
        print("⚠️  Some tests failed. Please check your configuration.")
        print("💡 Make sure MongoDB and Milvus are running:")
        print("   - MongoDB: sudo systemctl start mongodb")
        print("   - Milvus: docker-compose up -d")
    
    # Cleanup
    db_manager.close_all_connections()


if __name__ == "__main__":
    main()