"""
Analytics endpoints for learning analytics and reporting
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta

from app.models.student import (
    StudentAnalyticsSummary, StudentDetailedAnalytics, ModuleAnalytics
)
from app.services.analytics_service import AnalyticsService
from app.services.personalization_service import PersonalizationService
from app.routers.auth import get_current_student

logger = logging.getLogger(__name__)

router = APIRouter()
analytics_service = AnalyticsService()
personalization_service = PersonalizationService()


@router.get("/student/summary")
async def get_student_summary(
    current_student: str = Depends(get_current_student)
):
    """Get student analytics summary"""
    try:
        logger.info(f"Getting student summary for {current_student}")
        
        # Get detailed analytics from personalization service
        analytics = await personalization_service.get_student_analytics(current_student)
        
        if not analytics:
            return {
                "success": False,
                "message": "No analytics data available",
                "student_id": current_student
            }
        
        return {
            "success": True,
            "student_id": current_student,
            "summary": analytics.summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting student summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get student summary"
        )


@router.get("/student/detailed")
async def get_detailed_analytics(
    current_student: str = Depends(get_current_student)
):
    """Get detailed student analytics"""
    try:
        logger.info(f"Getting detailed analytics for {current_student}")
        
        # Get detailed analytics
        analytics = await personalization_service.get_student_analytics(current_student)
        
        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No analytics data found"
            )
        
        return {
            "success": True,
            "student_id": current_student,
            "analytics": analytics,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detailed analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get detailed analytics"
        )


@router.get("/student/report")
async def generate_student_report(
    current_student: str = Depends(get_current_student)
):
    """Generate comprehensive student analytics report"""
    try:
        logger.info(f"Generating student report for {current_student}")
        
        # Generate comprehensive report
        report = await analytics_service.generate_student_report(current_student)
        
        if "error" in report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=report["error"]
            )
        
        return {
            "success": True,
            "report": report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating student report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate student report"
        )


@router.get("/student/module/{module_id}")
async def get_module_analytics(
    module_id: str,
    current_student: str = Depends(get_current_student)
):
    """Get analytics for a specific module"""
    try:
        logger.info(f"Getting module analytics for {current_student}, module {module_id}")
        
        # Get module-specific analytics
        module_analytics = await personalization_service.get_module_analytics(current_student, module_id)
        
        if not module_analytics:
            return {
                "success": False,
                "message": f"No analytics data found for module {module_id}",
                "student_id": current_student,
                "module_id": module_id
            }
        
        return {
            "success": True,
            "student_id": current_student,
            "module_id": module_id,
            "analytics": module_analytics,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting module analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get module analytics"
        )


@router.get("/student/wellness")
async def get_wellness_summary(
    current_student: str = Depends(get_current_student)
):
    """Get student wellness summary including anxiety analytics"""
    try:
        logger.info(f"Getting wellness summary for {current_student}")
        
        # Get wellness summary from analytics service
        wellness_summary = await analytics_service.get_student_wellness_summary(current_student)
        
        return {
            "success": True,
            "student_id": current_student,
            "wellness_summary": wellness_summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting wellness summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get wellness summary"
        )


@router.get("/student/export")
async def export_student_data(
    format: str = Query("json", regex="^(json|csv)$"),
    current_student: str = Depends(get_current_student)
):
    """Export comprehensive student data"""
    try:
        logger.info(f"Exporting student data for {current_student} in {format} format")
        
        # Export data
        export_data = await analytics_service.export_student_data(current_student, format)
        
        if "error" in export_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=export_data["error"]
            )
        
        # Return appropriate response based on format
        if format == "json":
            return JSONResponse(content=export_data)
        else:  # CSV format would be implemented here
            return JSONResponse(content={
                "success": False,
                "message": "CSV export not yet implemented",
                "data": export_data
            })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting student data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export student data"
        )


@router.get("/student/progress")
async def get_learning_progress(
    days: int = Query(30, ge=1, le=365),
    current_student: str = Depends(get_current_student)
):
    """Get learning progress over time"""
    try:
        logger.info(f"Getting learning progress for {current_student} over {days} days")
        
        # Get detailed analytics for progress calculation
        analytics = await personalization_service.get_student_analytics(current_student)
        
        if not analytics:
            return {
                "success": False,
                "message": "No progress data available",
                "student_id": current_student
            }
        
        # Calculate progress metrics
        progress_data = {
            "total_study_time": analytics.summary.total_study_time,
            "total_interactions": analytics.summary.total_interactions,
            "satisfaction_rate": analytics.summary.satisfaction_rate,
            "daily_activity": analytics.daily_activity[-days:] if analytics.daily_activity else [],
            "module_progress": [
                {
                    "module_id": module.module_id,
                    "interactions": module.total_interactions,
                    "time_spent": module.total_time_spent,
                    "satisfaction": module.satisfaction_rate
                }
                for module in analytics.module_analytics
            ],
            "learning_style_evolution": analytics.learning_style_usage,
            "strengths": analytics.strengths,
            "improvement_areas": analytics.areas_for_improvement,
            "recommendations": analytics.recommendations
        }
        
        return {
            "success": True,
            "student_id": current_student,
            "period_days": days,
            "progress": progress_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting learning progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get learning progress"
        )


@router.get("/student/recommendations")
async def get_personalized_recommendations(
    current_student: str = Depends(get_current_student)
):
    """Get personalized learning recommendations"""
    try:
        logger.info(f"Getting personalized recommendations for {current_student}")
        
        # Generate comprehensive report to get recommendations
        report = await analytics_service.generate_student_report(current_student)
        
        if "error" in report:
            return {
                "success": False,
                "message": "No data available for recommendations",
                "student_id": current_student
            }
        
        # Extract recommendations
        recommendations = {
            "learning_recommendations": report.get("recommendations", []),
            "strengths": report.get("strengths", []),
            "areas_for_improvement": report.get("areas_for_improvement", []),
            "preferred_learning_style": report.get("preferred_learning_style", {}),
            "module_suggestions": []
        }
        
        # Add module-specific suggestions
        modules_activity = report.get("modules_activity", [])
        for module in modules_activity:
            if module.get("satisfaction_rate", 100) < 70:
                recommendations["module_suggestions"].append({
                    "module_id": module["module_id"],
                    "suggestion": f"Consider reviewing materials for {module['module_id']}",
                    "reason": f"Satisfaction rate: {module.get('satisfaction_rate', 0):.1f}%"
                })
        
        return {
            "success": True,
            "student_id": current_student,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get personalized recommendations"
        )


@router.get("/student/performance-trends")
async def get_performance_trends(
    current_student: str = Depends(get_current_student)
):
    """Get performance trends analysis"""
    try:
        logger.info(f"Getting performance trends for {current_student}")
        
        # Generate report for trend analysis
        report = await analytics_service.generate_student_report(current_student)
        
        if "error" in report:
            return {
                "success": False,
                "message": "No data available for trend analysis",
                "student_id": current_student
            }
        
        # Extract trends
        trends = {
            "overall_trend": report.get("performance_trends", {}).get("trend", "stable"),
            "satisfaction_trend": {
                "current": report.get("satisfaction_rate"),
                "change": report.get("performance_trends", {}).get("satisfaction_change", 0)
            },
            "activity_patterns": {
                "daily_activity": report.get("daily_activity", []),
                "most_active_times": [],  # Could be calculated from activity data
                "consistency_score": 0.0   # Could be calculated from activity patterns
            },
            "learning_style_trends": report.get("learning_style_usage", []),
            "module_performance_trends": [
                {
                    "module_id": module["module_id"],
                    "trend": "stable",  # Would calculate based on historical data
                    "current_satisfaction": module.get("satisfaction_rate")
                }
                for module in report.get("modules_activity", [])
            ]
        }
        
        return {
            "success": True,
            "student_id": current_student,
            "trends": trends,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance trends"
        )


@router.get("/system/overview")
async def get_system_analytics(
    current_student: str = Depends(get_current_student)
):
    """Get system-wide analytics (aggregated view for students)"""
    try:
        logger.info(f"Getting system analytics overview for {current_student}")
        
        # Get system analytics
        system_analytics = await analytics_service.get_system_analytics()
        
        if "error" in system_analytics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=system_analytics["error"]
            )
        
        # Filter to show only general statistics (privacy-preserving)
        public_analytics = {
            "total_active_students": system_analytics["student_metrics"]["active_students_30d"],
            "activity_rate": system_analytics["student_metrics"]["activity_rate"],
            "overall_satisfaction": system_analytics["interaction_metrics"]["overall_satisfaction_rate"],
            "popular_learning_styles": system_analytics["learning_preferences"]["style_distribution"],
            "popular_modules": system_analytics["learning_preferences"]["popular_modules"][:5],
            "system_activity_trend": system_analytics["activity_trends"]["daily_activity_30d"][-7:]  # Last 7 days
        }
        
        return {
            "success": True,
            "system_analytics": public_analytics,
            "generated_at": system_analytics["generated_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system analytics"
        )


@router.get("/comparisons/learning-styles")
async def compare_learning_styles(
    current_student: str = Depends(get_current_student)
):
    """Compare effectiveness of different learning styles for the student"""
    try:
        logger.info(f"Comparing learning styles for {current_student}")
        
        # Get detailed analytics
        analytics = await personalization_service.get_student_analytics(current_student)
        
        if not analytics:
            return {
                "success": False,
                "message": "No data available for learning style comparison",
                "student_id": current_student
            }
        
        # Analyze learning style effectiveness
        style_effectiveness = {}
        
        for style, usage_count in analytics.learning_style_usage.items():
            if usage_count > 0:
                # Calculate effectiveness metrics (simplified)
                effectiveness_score = 0.75  # Placeholder - would calculate from actual satisfaction rates
                
                style_effectiveness[style.value] = {
                    "usage_count": usage_count,
                    "usage_percentage": (usage_count / analytics.summary.total_interactions) * 100,
                    "effectiveness_score": effectiveness_score,
                    "recommendation": "Continue using" if effectiveness_score > 0.7 else "Consider trying other styles"
                }
        
        # Generate recommendations
        best_style = max(style_effectiveness.items(), key=lambda x: x[1]["effectiveness_score"])[0] if style_effectiveness else None
        
        return {
            "success": True,
            "student_id": current_student,
            "current_preferred_style": analytics.summary.preferred_learning_style.id if analytics.summary.preferred_learning_style else None,
            "style_effectiveness": style_effectiveness,
            "best_performing_style": best_style,
            "recommendations": [
                f"Your most effective learning style appears to be: {best_style}" if best_style else "Try different learning styles to find what works best for you",
                "Consider experimenting with visual learning for complex topics",
                "Use quiz-based learning for review sessions"
            ],
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing learning styles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare learning styles"
        )


@router.get("/health")
async def analytics_health_check():
    """Health check for analytics service"""
    try:
        # Test basic analytics functionality
        from app.core.database import check_database_health
        
        db_health = await check_database_health()
        
        return {
            "status": "healthy" if db_health["overall_healthy"] else "degraded",
            "service": "analytics",
            "database_status": "healthy" if db_health["overall_healthy"] else "unhealthy",
            "features": [
                "Student analytics",
                "Learning progress tracking",
                "Performance trends",
                "Personalized recommendations",
                "Wellness analytics",
                "Data export"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "analytics",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }