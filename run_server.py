#!/usr/bin/env python3
"""
Easy startup script for MindMate-Edu Backend
Handles environment setup and server startup
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_virtual_environment():
    """Check if running in virtual environment"""
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")
        print("   Consider activating a virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate     # Windows")
    
    return True


def check_environment_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    elif env_example.exists():
        print("⚠️  .env file not found, but .env.example exists")
        print("   Please copy .env.example to .env and configure it:")
        print("   cp .env.example .env")
        return False
    else:
        print("❌ No .env or .env.example file found")
        print("   Please create .env file with required configuration")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pymongo",
        "pymilvus",
        "langchain",
        "transformers",
        "torch"
    ]
    
    print("🔍 Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True


async def check_database_connections():
    """Check database connections"""
    print("🗄️ Checking database connections...")
    
    try:
        # Import database manager
        sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
        from app.core.database import check_database_health
        
        health = await check_database_health()
        
        if health["mongodb"]["connected"]:
            print("✅ MongoDB connection successful")
        else:
            print(f"❌ MongoDB connection failed: {health['mongodb'].get('error', 'Unknown error')}")
        
        if health["milvus"]["connected"]:
            print("✅ Milvus connection successful")
        else:
            print(f"❌ Milvus connection failed: {health['milvus'].get('error', 'Unknown error')}")
        
        return health["overall_healthy"]
        
    except Exception as e:
        print(f"❌ Database connection check failed: {e}")
        return False


def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server"""
    print(f"\n🚀 Starting MindMate-Edu Backend Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   API Docs: http://{host}:{port}/docs")
    print(f"   Health Check: http://{host}:{port}/health")
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", host,
            "--port", str(port),
        ]
        
        if reload:
            cmd.append("--reload")
        
        print(f"\n▶️  Executing: {' '.join(cmd)}")
        print("   Press Ctrl+C to stop the server\n")
        
        # Run the server
        process = subprocess.run(cmd)
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False


def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    MindMate-Edu Backend                      ║
    ║              AI-Powered Educational Assistant                ║
    ║                                                              ║
    ║  Features:                                                   ║
    ║  • Intelligent Q&A with RAG                                 ║
    ║  • Multimodal Document Processing                           ║
    ║  • Anxiety Detection & Wellness                             ║
    ║  • Personalized Learning Styles                             ║
    ║  • Learning Analytics & Progress Tracking                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def main():
    """Main startup function"""
    print_banner()
    
    print("🔧 Pre-flight System Check...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check virtual environment
    check_virtual_environment()
    
    # Check environment file
    if not check_environment_file():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check database connections
    db_healthy = await check_database_connections()
    if not db_healthy:
        print("\n⚠️  Database connections failed, but server can still start")
        print("   Some features may not work properly")
        print("   Please ensure MongoDB and Milvus are running")
        
        response = input("\n❓ Do you want to continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            return False
    
    print("\n✅ Pre-flight check completed!")
    
    # Get server configuration
    try:
        from app.core.config import get_settings
        settings = get_settings()
        host = settings.API_HOST
        port = settings.API_PORT
        debug = settings.DEBUG
    except Exception as e:
        print(f"⚠️  Could not load settings: {e}")
        print("   Using default configuration")
        host = "0.0.0.0"
        port = 8000
        debug = True
    
    # Start server
    success = start_server(host=host, port=port, reload=debug)
    
    if success:
        print("\n✅ Server started successfully!")
    else:
        print("\n❌ Server failed to start")
    
    return success


def print_help():
    """Print help information"""
    help_text = """
MindMate-Edu Backend Startup Script

Usage:
    python run_server.py [options]

Options:
    --help, -h          Show this help message
    --host HOST         Server host (default: 0.0.0.0)
    --port PORT         Server port (default: 8000)
    --no-reload         Disable auto-reload
    --check-only        Only run system checks, don't start server

Examples:
    python run_server.py                    # Start with default settings
    python run_server.py --port 8080        # Start on port 8080
    python run_server.py --check-only       # Just run system checks
    
Environment Setup:
    1. Create virtual environment: python -m venv venv
    2. Activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)
    3. Install dependencies: pip install -r requirements.txt
    4. Copy .env.example to .env and configure
    5. Start databases (MongoDB, Milvus)
    6. Run this script
    
For more information, visit: http://localhost:8000/docs
    """
    print(help_text)


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="MindMate-Edu Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--check-only", action="store_true", help="Only run system checks")
    
    args = parser.parse_args()
    
    if args.check_only:
        # Run only system checks
        async def check_only():
            print_banner()
            print("🔧 Running System Checks Only...")
            
            checks = [
                check_python_version(),
                check_virtual_environment(),
                check_environment_file(),
                check_dependencies(),
                await check_database_connections()
            ]
            
            passed = sum(checks)
            total = len(checks)
            
            print(f"\n📊 System Check Results: {passed}/{total} checks passed")
            
            if passed == total:
                print("✅ System is ready for deployment!")
            else:
                print("⚠️  Some checks failed. Please resolve issues before starting server.")
            
            return passed == total
        
        success = asyncio.run(check_only())
        sys.exit(0 if success else 1)
    else:
        # Run full startup
        success = asyncio.run(main())
        sys.exit(0 if success else 1)