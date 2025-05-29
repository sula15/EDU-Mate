"""
Main FastAPI application entry point
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import get_settings
from app.core.database import lifespan, check_database_health
from app.routers import auth, chat, anxiety, documents, analytics, admin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME}")
    
    # Initialize database connections
    async with lifespan(app):
        # Health check on startup
        health = await check_database_health()
        if health["overall_healthy"]:
            logger.info("✅ All database connections healthy")
        else:
            logger.warning("⚠️ Some database connections failed")
            logger.warning(f"MongoDB: {health['mongodb']}")
            logger.warning(f"Milvus: {health['milvus']}")
        
        yield
    
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered educational assistant with anxiety detection and personalized learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*"]
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error occurred",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_STR}/auth",
    tags=["Authentication"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["Chat & RAG"]
)

app.include_router(
    anxiety.router,
    prefix=f"{settings.API_V1_STR}/anxiety",
    tags=["Anxiety Detection"]
)

app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["Document Management"]
)

app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_STR}/analytics",
    tags=["Analytics"]
)

app.include_router(
    admin.router,
    prefix=f"{settings.API_V1_STR}/admin",
    tags=["Administration"]
)


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_health = await check_database_health()
        
        return {
            "status": "healthy" if db_health["overall_healthy"] else "degraded",
            "timestamp": time.time(),
            "services": {
                "api": "healthy",
                "mongodb": "healthy" if db_health["mongodb"]["connected"] else "unhealthy",
                "milvus": "healthy" if db_health["milvus"]["connected"] else "unhealthy"
            },
            "database_details": db_health
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }


@app.get("/info")
async def app_info():
    """Application information"""
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "AI-powered educational assistant",
        "features": [
            "Intelligent Q&A with RAG",
            "Multimodal document processing",
            "Anxiety detection and wellness",
            "Personalized learning styles",
            "Learning analytics",
            "Student progress tracking"
        ],
        "api_version": settings.API_V1_STR,
        "environment": "development" if settings.DEBUG else "production"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )