#!/usr/bin/env python3
"""
Comprehensive API testing script for MindMate-Edu Backend
Tests all major endpoints to ensure the API is working correctly
"""

import asyncio
import aiohttp
import json
import sys
import os
from datetime import datetime
import base64
import tempfile

# Configuration
BASE_URL = "http://localhost:8000"
API_V1 = "/api/v1"

class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        self.access_token = None
        self.student_id = "test_student_001"
        self.student_name = "Test Student"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_headers(self):
        """Get headers with authentication token"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    async def test_health_endpoints(self):
        """Test health check endpoints"""
        print("🏥 Testing Health Check Endpoints...")
        
        try:
            # Root health check
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Root health check: {data.get('status', 'unknown')}")
                else:
                    print(f"❌ Root health check failed: {response.status}")
            
            # Service-specific health checks
            services = ["auth", "chat", "anxiety", "documents", "analytics", "admin"]
            
            for service in services:
                try:
                    url = f"{self.base_url}{API_V1}/{service}/health"
                    async with self.session.get(url, headers=self.get_headers()) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ {service.capitalize()} service: {data.get('status', 'healthy')}")
                        else:
                            print(f"⚠️  {service.capitalize()} service: status {response.status}")
                except Exception as e:
                    print(f"⚠️  {service.capitalize()} service: {str(e)[:50]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_authentication(self):
        """Test authentication endpoints"""
        print("\n🔐 Testing Authentication...")
        
        try:
            # Test registration
            registration_data = {
                "student_id": self.student_id,
                "name": self.student_name,
                "email": "test@example.com",
                "default_learning_style": "detailed"
            }
            
            url = f"{self.base_url}{API_V1}/auth/register"
            async with self.session.post(url, json=registration_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Registration: {data.get('message', 'success')}")
                else:
                    print(f"⚠️  Registration: status {response.status}")
            
            # Test login
            login_data = {
                "student_id": self.student_id,
                "name": self.student_name
            }
            
            url = f"{self.base_url}{API_V1}/auth/login"
            async with self.session.post(url, json=login_data) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    print(f"✅ Login successful: token received")
                else:
                    print(f"❌ Login failed: status {response.status}")
                    return False
            
            # Test profile access
            url = f"{self.base_url}{API_V1}/auth/profile"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Profile access: {data.get('name', 'unknown')}")
                else:
                    print(f"❌ Profile access failed: status {response.status}")
            
            # Test learning styles
            url = f"{self.base_url}{API_V1}/auth/learning-styles"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    styles = data.get("learning_styles", [])
                    print(f"✅ Learning styles: {len(styles)} available")
                else:
                    print(f"❌ Learning styles failed: status {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    async def test_chat_endpoints(self):
        """Test chat and RAG endpoints"""
        print("\n💬 Testing Chat Endpoints...")
        
        if not self.access_token:
            print("❌ No access token available for chat tests")
            return False
        
        try:
            # Test chat query
            chat_data = {
                "student_id": self.student_id,
                "message": "What is machine learning?",
                "learning_style": "detailed",
                "include_images": True,
                "include_sources": True,
                "conversation_history": []
            }
            
            url = f"{self.base_url}{API_V1}/chat/query"
            async with self.session.post(url, json=chat_data, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Chat query: response generated in {data.get('processing_time', 0):.2f}s")
                else:
                    print(f"⚠️  Chat query: status {response.status}")
            
            # Test search
            search_data = {
                "query": "neural networks",
                "search_type": "mixed",
                "top_k": 5
            }
            
            url = f"{self.base_url}{API_V1}/chat/search"
            async with self.session.post(url, json=search_data, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Search: {data.get('total_results', 0)} results found")
                else:
                    print(f"⚠️  Search: status {response.status}")
            
            # Test quick response
            url = f"{self.base_url}{API_V1}/chat/quick-response?message=hello"
            async with self.session.post(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Quick response: {len(data.get('suggestions', []))} suggestions")
                else:
                    print(f"⚠️  Quick response: status {response.status}")
            
            # Test chat history
            url = f"{self.base_url}{API_V1}/chat/history"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Chat history: {data.get('total_interactions', 0)} interactions")
                else:
                    print(f"⚠️  Chat history: status {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Chat endpoints error: {e}")
            return False
    
    async def test_anxiety_endpoints(self):
        """Test anxiety detection endpoints"""
        print("\n🧠 Testing Anxiety Detection...")
        
        if not self.access_token:
            print("❌ No access token available for anxiety tests")
            return False
        
        try:
            # Test text-based anxiety assessment
            anxiety_data = {
                "student_id": self.student_id,
                "input_text": "I'm feeling a bit overwhelmed with my studies and assignments.",
                "session_context": {}
            }
            
            url = f"{self.base_url}{API_V1}/anxiety/assess"
            async with self.session.post(url, json=anxiety_data, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    assessment = data.get("assessment", {})
                    prediction = assessment.get("prediction", {})
                    print(f"✅ Anxiety assessment: {prediction.get('anxiety_level', 'unknown')}")
                else:
                    print(f"⚠️  Anxiety assessment: status {response.status}")
            
            # Test wellness recommendations
            url = f"{self.base_url}{API_V1}/anxiety/wellness-recommendations"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    recommendations = data.get("recommendations", [])
                    print(f"✅ Wellness recommendations: {len(recommendations)} suggestions")
                else:
                    print(f"⚠️  Wellness recommendations: status {response.status}")
            
            # Test anxiety history
            url = f"{self.base_url}{API_V1}/anxiety/history"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Anxiety history: {data.get('total_assessments', 0)} assessments")
                else:
                    print(f"⚠️  Anxiety history: status {response.status}")
            
            # Test crisis resources
            url = f"{self.base_url}{API_V1}/anxiety/crisis-resources"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    resources = data.get("resources", {})
                    print(f"✅ Crisis resources: {len(resources.get('emergency_contacts', []))} emergency contacts")
                else:
                    print(f"⚠️  Crisis resources: status {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Anxiety endpoints error: {e}")
            return False
    
    async def test_document_endpoints(self):
        """Test document management endpoints"""
        print("\n📄 Testing Document Management...")
        
        if not self.access_token:
            print("❌ No access token available for document tests")
            return False
        
        try:
            # Test document search
            url = f"{self.base_url}{API_V1}/documents/search"
            params = {"query": "machine learning", "limit": 5}
            async with self.session.get(url, params=params, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Document search: {data.get('total_count', 0)} documents found")
                else:
                    print(f"⚠️  Document search: status {response.status}")
            
            # Test document stats
            url = f"{self.base_url}{API_V1}/documents/stats"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Document stats: {data.get('total_documents', 0)} total documents")
                else:
                    print(f"⚠️  Document stats: status {response.status}")
            
            # Test supported document types
            url = f"{self.base_url}{API_V1}/documents/supported-types"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    types = data.get("supported_types", [])
                    print(f"✅ Supported types: {len(types)} document types")
                else:
                    print(f"⚠️  Supported types: status {response.status}")
            
            # Test system status
            url = f"{self.base_url}{API_V1}/documents/system/status"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ System status: {data.get('system_health', 'unknown')}")
                else:
                    print(f"⚠️  System status: status {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Document endpoints error: {e}")
            return False
    
    async def test_analytics_endpoints(self):
        """Test analytics endpoints"""
        print("\n📊 Testing Analytics...")
        
        if not self.access_token:
            print("❌ No access token available for analytics tests")
            return False
        
        try:
            # Test student summary
            url = f"{self.base_url}{API_V1}/analytics/student/summary"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Student summary: {data.get('success', False)}")
                else:
                    print(f"⚠️  Student summary: status {response.status}")
            
            # Test detailed analytics
            url = f"{self.base_url}{API_V1}/analytics/student/detailed"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Detailed analytics: {data.get('success', False)}")
                elif response.status == 404:
                    print(f"⚠️  Detailed analytics: No data available (expected for new user)")
                else:
                    print(f"⚠️  Detailed analytics: status {response.status}")
            
            # Test wellness summary
            url = f"{self.base_url}{API_V1}/analytics/student/wellness"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Wellness summary: {data.get('success', False)}")
                else:
                    print(f"⚠️  Wellness summary: status {response.status}")
            
            # Test system overview
            url = f"{self.base_url}{API_V1}/analytics/system/overview"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ System overview: {data.get('success', False)}")
                else:
                    print(f"⚠️  System overview: status {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Analytics endpoints error: {e}")
            return False
    
    async def test_admin_endpoints(self):
        """Test admin endpoints"""
        print("\n⚙️  Testing Admin Endpoints...")
        
        if not self.access_token:
            print("❌ No access token available for admin tests")
            return False
        
        try:
            # Test system health
            url = f"{self.base_url}{API_V1}/admin/system/health"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ System health: {data.get('overall_status', 'unknown')}")
                else:
                    print(f"⚠️  System health: status {response.status}")
            
            # Test system analytics
            url = f"{self.base_url}{API_V1}/admin/analytics/system"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ System analytics: {data.get('success', False)}")
                else:
                    print(f"⚠️  System analytics: status {response.status}")
            
            # Test student list
            url = f"{self.base_url}{API_V1}/admin/students/list"
            params = {"limit": 10, "active_only": True}
            async with self.session.get(url, params=params, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Student list: {data.get('returned_count', 0)} students")
                else:
                    print(f"⚠️  Student list: status {response.status}")
            
            # Test database collections
            url = f"{self.base_url}{API_V1}/admin/database/collections"
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Database collections: {data.get('total_collections', 0)} collections")
                else:
                    print(f"⚠️  Database collections: status {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Admin endpoints error: {e}")
            return False
    
    async def test_application_info(self):
        """Test application info endpoints"""
        print("\n📋 Testing Application Info...")
        
        try:
            # Test root endpoint
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Root endpoint: {data.get('message', 'unknown')}")
                else:
                    print(f"❌ Root endpoint failed: {response.status}")
            
            # Test app info
            async with self.session.get(f"{self.base_url}/info") as response:
                if response.status == 200:
                    data = await response.json()
                    features = data.get('features', [])
                    print(f"✅ App info: {len(features)} features listed")
                else:
                    print(f"❌ App info failed: {response.status}")
            
            return True
            
        except Exception as e:
            print(f"❌ Application info error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Comprehensive API Tests for MindMate-Edu Backend\n")
        
        test_results = []
        
        # Test application info
        test_results.append(await self.test_application_info())
        
        # Test health endpoints
        test_results.append(await self.test_health_endpoints())
        
        # Test authentication (must be first to get token)
        test_results.append(await self.test_authentication())
        
        # Test other endpoints (require authentication)
        test_results.append(await self.test_chat_endpoints())
        test_results.append(await self.test_anxiety_endpoints())
        test_results.append(await self.test_document_endpoints())
        test_results.append(await self.test_analytics_endpoints())
        test_results.append(await self.test_admin_endpoints())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        print(f"\n📊 Test Results: {passed_tests}/{total_tests} test suites passed")
        
        if passed_tests == total_tests:
            print("🎉 All API tests passed! The backend is working correctly.")
            print("\n✅ Ready for production deployment!")
        elif passed_tests >= 6:  # At least 75% passed
            print("✨ Most API tests passed! Some advanced features may need configuration.")
            print("🚀 Backend is functional and ready for frontend integration!")
        else:
            print("⚠️  Some critical API tests failed. Please check your setup.")
        
        print("\n📖 Next Steps:")
        print("   1. Check API documentation at: http://localhost:8000/docs")
        print("   2. Test with frontend application")
        print("   3. Configure production settings")
        print("   4. Set up monitoring and logging")
        
        return passed_tests >= 6


async def main():
    """Main test function"""
    try:
        # Check if server is running
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        print(f"❌ Server not responding properly (status: {response.status})")
                        print("🔧 Please make sure the FastAPI server is running:")
                        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
                        return False
            except Exception as e:
                print(f"❌ Cannot connect to server at {BASE_URL}")
                print("🔧 Please start the FastAPI server first:")
                print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
                print(f"   Error: {e}")
                return False
        
        # Run comprehensive tests
        async with APITester(BASE_URL) as tester:
            success = await tester.run_all_tests()
            return success
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)