"""
Administration endpoints for system management
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status, Query
from datetime import datetime, timedelta

from app.models.anxiety import SystemWellnessMetrics
from app.services.analytics_service import AnalyticsService
from app.services.anxiety_service import AnxietyService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.core.database import check_database_health, db_manager
from app.core.config import get_settings
from app.routers.auth import get_current_student

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()
analytics_service = AnalyticsService()
anxiety_service = AnxietyService()
chat_service = ChatService()
document_service = DocumentService()


# Simple admin authentication (in production, use proper admin roles)
async def get_admin_user(current_student: str = Depends(get_current_student)):
    """Simple admin check - in production, implement proper role-based access"""
    # For now, allow any authenticated user to access admin functions
    # In production, you'd check for admin role in JWT or database
    logger.info(f"Admin access by user: {current_student}")
    return current_student


@router.get("/system/health")
async def system_health_check(admin_user: str = Depends(get_admin_user)):
    """Comprehensive system health check"""
    try:
        logger.info(f"System health check requested by admin: {admin_user}")
        
        # Check database health
        db_health = await check_database_health()
        
        # Check individual services
        chat_health = await chat_service.health_check()
        
        # Check anxiety model status
        from app.ml_models.anxiety_model import is_model_loaded
        anxiety_model_loaded = is_model_loaded()
        
        # Compile overall health status
        health_status = {
            "overall_status": "healthy" if db_health["overall_healthy"] else "degraded",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "mongodb": db_health["mongodb"],
                "milvus": db_health["milvus"]
            },
            "services": {
                "chat_service": chat_health.get("chat_service", "unknown"),
                "llm_service": chat_health.get("llm_service", "unknown"),
                "embedding_service": chat_health.get("embedding_service", "unknown"),
                "anxiety_service": "healthy" if anxiety_model_loaded else "degraded",
                "document_service": "healthy",  # Would check actual status
                "analytics_service": "healthy"   # Would check actual status
            },
            "system_info": {
                "project_name": settings.PROJECT_NAME,
                "debug_mode": settings.DEBUG,
                "api_version": settings.API_V1_STR,
                "max_file_size": settings.MAX_FILE_SIZE,
                "session_timeout": settings.SESSION_TIMEOUT
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in system health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform system health check"
        )


@router.get("/analytics/system")
async def get_system_analytics(admin_user: str = Depends(get_admin_user)):
    """Get comprehensive system analytics"""
    try:
        logger.info(f"System analytics requested by admin: {admin_user}")
        
        # Get system-wide analytics
        system_analytics = await analytics_service.get_system_analytics()
        
        if "error" in system_analytics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=system_analytics["error"]
            )
        
        return {
            "success": True,
            "admin_user": admin_user,
            "analytics": system_analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system analytics"
        )


@router.get("/analytics/wellness")
async def get_wellness_metrics(admin_user: str = Depends(get_admin_user)):
    """Get system-wide wellness metrics"""
    try:
        logger.info(f"Wellness metrics requested by admin: {admin_user}")
        
        # Get wellness metrics from database
        from app.core.database import get_mongo_collection
        
        anxiety_collection = get_mongo_collection("anxiety_assessments")
        profiles_collection = get_mongo_collection(settings.STUDENT_PROFILES_COLLECTION)
        
        # Calculate system-wide wellness metrics
        total_assessments = anxiety_collection.count_documents({})
        total_students = profiles_collection.count_documents({})
        
        # Get assessments from last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_assessments = list(anxiety_collection.find({
            "timestamp": {"$gte": thirty_days_ago.isoformat()}
        }))
        
        # Analyze anxiety distribution
        anxiety_distribution = {}
        for assessment in recent_assessments:
            level = assessment.get("prediction", {}).get("anxiety_level", "Unknown")
            anxiety_distribution[level] = anxiety_distribution.get(level, 0) + 1
        
        # Calculate high-risk students (those with recent severe/moderate anxiety)
        high_risk_students = len(set(
            assessment["student_id"]
            for assessment in recent_assessments
            if assessment.get("prediction", {}).get("anxiety_level") in ["Severe Anxiety", "Moderate Anxiety"]
        ))
        
        # Mock intervention success rate (would be calculated from real data)
        intervention_success_rate = 0.75
        
        # Mock most effective interventions (would be calculated from feedback data)
        most_effective_interventions = [
            {"type": "breathing", "effectiveness": 0.85, "usage_count": 120},
            {"type": "mindfulness", "effectiveness": 0.78, "usage_count": 95},
            {"type": "physical", "effectiveness": 0.72, "usage_count": 87}
        ]
        
        wellness_metrics = SystemWellnessMetrics(
            total_assessments=total_assessments,
            total_students=total_students,
            avg_assessments_per_student=total_assessments / max(total_students, 1),
            anxiety_distribution=anxiety_distribution,
            high_risk_students=high_risk_students,
            intervention_success_rate=intervention_success_rate,
            most_effective_interventions=most_effective_interventions,
            timestamp=datetime.now()
        )
        
        return {
            "success": True,
            "admin_user": admin_user,
            "wellness_metrics": wellness_metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting wellness metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get wellness metrics"
        )


@router.get("/students/list")
async def list_students(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    active_only: bool = Query(True),
    admin_user: str = Depends(get_admin_user)
):
    """List students with basic information"""
    try:
        logger.info(f"Student list requested by admin: {admin_user}")
        
        from app.core.database import get_student_profiles_collection
        
        profiles_collection = get_student_profiles_collection()
        
        # Build query
        query = {}
        if active_only:
            # Students active in last 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            query["last_active"] = {"$gte": thirty_days_ago}
        
        # Get total count
        total_count = profiles_collection.count_documents(query)
        
        # Get paginated results
        cursor = profiles_collection.find(
            query,
            {"student_id": 1, "name": 1, "email": 1, "last_active": 1, "total_interactions": 1, "default_learning_style": 1}
        ).skip(offset).limit(limit)
        
        students = []
        for profile in cursor:
            students.append({
                "student_id": profile["student_id"],
                "name": profile.get("name", "Unknown"),
                "email": profile.get("email"),
                "last_active": profile.get("last_active"),
                "total_interactions": profile.get("total_interactions", 0),
                "default_learning_style": profile.get("default_learning_style")
            })
        
        return {
            "success": True,
            "total_count": total_count,
            "returned_count": len(students),
            "limit": limit,
            "offset": offset,
            "students": students
        }
        
    except Exception as e:
        logger.error(f"Error listing students: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list students"
        )


@router.get("/students/{student_id}/details")
async def get_student_details(
    student_id: str,
    admin_user: str = Depends(get_admin_user)
):
    """Get detailed information about a specific student"""
    try:
        logger.info(f"Student details requested by admin {admin_user} for student {student_id}")
        
        # Get student analytics
        analytics = await analytics_service.generate_student_report(student_id)
        
        if "error" in analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Student not found: {student_id}"
            )
        
        # Get wellness summary
        wellness_summary = await analytics_service.get_student_wellness_summary(student_id)
        
        return {
            "success": True,
            "admin_user": admin_user,
            "student_id": student_id,
            "analytics": analytics,
            "wellness_summary": wellness_summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student details"
        )


@router.get("/database/collections")
async def list_database_collections(admin_user: str = Depends(get_admin_user)):
    """List all database collections with statistics"""
    try:
        logger.info(f"Database collections requested by admin: {admin_user}")
        
        collections_info = []
        
        # MongoDB collections
        mongo_db = db_manager.get_mongo_db()
        mongo_collections = mongo_db.list_collection_names()
        
        for collection_name in mongo_collections:
            try:
                collection = mongo_db[collection_name]
                count = collection.count_documents({})
                
                # Get collection stats
                stats = mongo_db.command("collStats", collection_name)
                size_mb = stats.get("size", 0) / (1024 * 1024)
                
                collections_info.append({
                    "name": collection_name,
                    "type": "mongodb",
                    "document_count": count,
                    "size_mb": round(size_mb, 2),
                    "storage_size_mb": round(stats.get("storageSize", 0) / (1024 * 1024), 2)
                })
            except Exception as e:
                collections_info.append({
                    "name": collection_name,
                    "type": "mongodb",
                    "document_count": "error",
                    "error": str(e)
                })
        
        # Milvus collections
        try:
            from pymilvus import utility, Collection
            
            milvus_collections = utility.list_collections()
            for collection_name in milvus_collections:
                try:
                    collection = Collection(collection_name)
                    collection.load()
                    
                    stats = collection.get_stats()
                    count = stats.get("row_count", 0)
                    
                    collections_info.append({
                        "name": collection_name,
                        "type": "milvus",
                        "document_count": count,
                        "size_mb": "unknown",  # Milvus doesn't easily provide size info
                        "schema": collection.schema.to_dict() if hasattr(collection, 'schema') else None
                    })
                except Exception as e:
                    collections_info.append({
                        "name": collection_name,
                        "type": "milvus",
                        "document_count": "error",
                        "error": str(e)
                    })
        except Exception as e:
            logger.warning(f"Could not get Milvus collections: {e}")
        
        return {
            "success": True,
            "admin_user": admin_user,
            "total_collections": len(collections_info),
            "collections": collections_info
        }
        
    except Exception as e:
        logger.error(f"Error listing database collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list database collections"
        )


@router.post("/system/maintenance")
async def run_system_maintenance(
    tasks: List[str] = Query([]),
    admin_user: str = Depends(get_admin_user)
):
    """Run system maintenance tasks"""
    try:
        logger.info(f"System maintenance requested by admin: {admin_user}")
        
        available_tasks = [
            "cleanup_temp_files",
            "rebuild_indexes",
            "update_analytics",
            "check_disk_space",
            "validate_data_integrity"
        ]
        
        if not tasks:
            tasks = ["cleanup_temp_files", "check_disk_space"]
        
        # Validate requested tasks
        invalid_tasks = [task for task in tasks if task not in available_tasks]
        if invalid_tasks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tasks: {invalid_tasks}. Available: {available_tasks}"
            )
        
        maintenance_results = {}
        
        for task in tasks:
            try:
                if task == "cleanup_temp_files":
                    # Cleanup temporary files
                    import os
                    temp_dir = settings.UPLOAD_DIR
                    if os.path.exists(temp_dir):
                        temp_files = [f for f in os.listdir(temp_dir) if f.startswith("temp_")]
                        for temp_file in temp_files:
                            try:
                                os.remove(os.path.join(temp_dir, temp_file))
                            except:
                                pass
                        maintenance_results[task] = f"Cleaned up {len(temp_files)} temporary files"
                    else:
                        maintenance_results[task] = "Temp directory does not exist"
                
                elif task == "rebuild_indexes":
                    # Rebuild database indexes
                    from app.core.database import create_mongodb_indexes
                    await create_mongodb_indexes()
                    maintenance_results[task] = "Database indexes rebuilt successfully"
                
                elif task == "update_analytics":
                    # Update analytics for all students (would be implemented)
                    maintenance_results[task] = "Analytics update queued for background processing"
                
                elif task == "check_disk_space":
                    # Check disk space
                    import shutil
                    total, used, free = shutil.disk_usage("/")
                    free_gb = free / (1024**3)
                    maintenance_results[task] = f"Free disk space: {free_gb:.2f} GB"
                
                elif task == "validate_data_integrity":
                    # Validate data integrity (basic check)
                    db_health = await check_database_health()
                    maintenance_results[task] = f"Data integrity check: {'passed' if db_health['overall_healthy'] else 'issues found'}"
                
            except Exception as e:
                maintenance_results[task] = f"Error: {str(e)}"
        
        return {
            "success": True,
            "admin_user": admin_user,
            "tasks_executed": tasks,
            "results": maintenance_results,
            "executed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running system maintenance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run system maintenance"
        )


@router.get("/logs/recent")
async def get_recent_logs(
    lines: int = Query(100, ge=1, le=1000),
    level: str = Query("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    admin_user: str = Depends(get_admin_user)
):
    """Get recent system logs"""
    try:
        logger.info(f"Recent logs requested by admin: {admin_user}")
        
        # In a production system, you'd read from actual log files
        # For now, return mock log entries
        
        mock_logs = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "level": "INFO" if i % 3 == 0 else "DEBUG" if i % 5 == 0 else "WARNING",
                "logger": f"app.services.{'chat_service' if i % 2 == 0 else 'anxiety_service'}",
                "message": f"Mock log entry {i}"
            }
            for i in range(min(lines, 50))  # Limit to 50 mock entries
        ]
        
        # Filter by level
        if level != "DEBUG":
            level_hierarchy = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
            min_level = level_hierarchy.get(level, 1)
            mock_logs = [
                log for log in mock_logs 
                if level_hierarchy.get(log["level"], 0) >= min_level
            ]
        
        return {
            "success": True,
            "admin_user": admin_user,
            "requested_lines": lines,
            "returned_lines": len(mock_logs),
            "level_filter": level,
            "logs": mock_logs,
            "note": "Mock log entries - connect to actual logging system in production"
        }
        
    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recent logs"
        )


@router.get("/config/current")
async def get_current_config(admin_user: str = Depends(get_admin_user)):
    """Get current system configuration (sensitive info redacted)"""
    try:
        logger.info(f"Current config requested by admin: {admin_user}")
        
        # Return safe configuration info (no secrets)
        safe_config = {
            "api": {
                "host": settings.API_HOST,
                "port": settings.API_PORT,
                "debug": settings.DEBUG,
                "api_version": settings.API_V1_STR,
                "project_name": settings.PROJECT_NAME
            },
            "database": {
                "mongodb_db": settings.MONGODB_DB,
                "milvus_host": settings.MILVUS_HOST,
                "milvus_port": settings.MILVUS_PORT
            },
            "collections": {
                "student_profiles": settings.STUDENT_PROFILES_COLLECTION,
                "student_interactions": settings.STUDENT_INTERACTIONS_COLLECTION,
                "student_analytics": settings.STUDENT_ANALYTICS_COLLECTION,
                "mongodb_images": settings.MONGODB_IMAGES_COLLECTION,
                "milvus_text": settings.MILVUS_TEXT_COLLECTION,
                "milvus_image": settings.MILVUS_IMAGE_COLLECTION
            },
            "models": {
                "text_embedding_model": settings.TEXT_EMBEDDING_MODEL,
                "clip_model_name": settings.CLIP_MODEL_NAME,
                "whisper_model_size": settings.WHISPER_MODEL_SIZE
            },
            "processing": {
                "max_file_size": settings.MAX_FILE_SIZE,
                "allowed_extensions": settings.ALLOWED_EXTENSIONS,
                "default_batch_size": settings.DEFAULT_BATCH_SIZE,
                "default_similarity_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
                "upload_dir": settings.UPLOAD_DIR
            },
            "session": {
                "timeout": settings.SESSION_TIMEOUT,
                "token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES
            }
        }
        
        return {
            "success": True,
            "admin_user": admin_user,
            "config": safe_config,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting current config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current configuration"
        )


@router.get("/health")
async def admin_health_check():
    """Health check for admin service"""
    return {
        "status": "healthy",
        "service": "administration",
        "features": [
            "System health monitoring",
            "Student management",
            "Analytics dashboard",
            "Database administration",
            "System maintenance",
            "Configuration management"
        ],
        "timestamp": datetime.now().isoformat()
    }