"""
Analytics Service - Comprehensive learning analytics and reporting
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from app.models.student import (
    StudentAnalyticsSummary, StudentDetailedAnalytics, ModuleAnalytics,
    LearningStyleEnum
)
from app.models.anxiety import AnxietyAnalytics, AnxietyLevel
from app.core.config import get_settings
from app.core.database import (
    get_student_profiles_collection, get_student_interactions_collection,
    get_student_analytics_collection, get_mongo_collection
)

logger = logging.getLogger(__name__)
settings = get_settings()


class AnalyticsService:
    """Service for comprehensive learning analytics and insights"""
    
    def __init__(self):
        self.settings = settings
    
    # Student Performance Analytics
    
    async def generate_student_report(self, student_id: str) -> Dict[str, Any]:
        """Generate comprehensive student analytics report"""
        try:
            # Get student profile
            profiles_collection = get_student_profiles_collection()
            profile = profiles_collection.find_one({"_id": student_id})
            
            if not profile:
                return {"error": "Student profile not found"}
            
            # Get interactions
            interactions_collection = get_student_interactions_collection()
            interactions = list(interactions_collection.find({"student_id": student_id}))
            
            if not interactions:
                return {"error": "No interaction data found"}
            
            # Calculate basic metrics
            total_interactions = len(interactions)
            completed_interactions = sum(1 for i in interactions if i.get("end_time"))
            total_time = sum(i.get("time_spent", 0) for i in interactions if i.get("time_spent"))
            
            # Calculate satisfaction rate
            helpful_ratings = [i.get("helpful") for i in interactions if i.get("helpful") is not None]
            satisfaction_rate = sum(1 for r in helpful_ratings if r) / max(len(helpful_ratings), 1) if helpful_ratings else None
            
            # Learning style analysis
            style_usage = self._analyze_learning_style_usage(interactions)
            preferred_style = self._get_preferred_learning_style(style_usage)
            
            # Module performance
            module_analytics = self._analyze_module_performance(interactions)
            
            # Daily activity patterns
            daily_activity = self._analyze_daily_activity(interactions)
            
            # Performance trends
            trends = self._analyze_performance_trends(interactions)
            
            # Recommendations
            recommendations = self._generate_personalized_recommendations(
                interactions, style_usage, module_analytics, satisfaction_rate
            )
            
            # Strengths and areas for improvement
            strengths, improvements = self._identify_strengths_and_improvements(
                interactions, module_analytics, satisfaction_rate
            )
            
            return {
                "student_id": student_id,
                "student_name": profile.get("name", "Unknown"),
                "report_generated": datetime.now().isoformat(),
                
                # Basic metrics
                "total_study_time": total_time / 60,  # Convert to minutes
                "total_interactions": total_interactions,
                "completed_interactions": completed_interactions,
                "completion_rate": completed_interactions / max(total_interactions, 1),
                "satisfaction_rate": satisfaction_rate * 100 if satisfaction_rate else None,
                
                # Learning preferences
                "preferred_learning_style": preferred_style,
                "learning_style_usage": style_usage,
                
                # Performance by module
                "modules_activity": module_analytics,
                
                # Activity patterns
                "daily_activity": daily_activity,
                "performance_trends": trends,
                
                # Insights
                "strengths": strengths,
                "areas_for_improvement": improvements,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating student report: {e}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    def _analyze_learning_style_usage(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze learning style usage patterns"""
        style_counts = Counter()
        
        for interaction in interactions:
            style = interaction.get("learning_style_id")
            if style:
                style_counts[style] += 1
        
        # Convert to list format
        usage_data = []
        total_interactions = sum(style_counts.values())
        
        for style_id, count in style_counts.most_common():
            try:
                # Get style name
                from app.core.config import learning_styles
                style_info = learning_styles.get_learning_style_by_id(style_id)
                style_name = style_info["name"] if style_info else style_id.replace("_", " ").title()
                
                usage_data.append({
                    "style_id": style_id,
                    "style_name": style_name,
                    "count": count,
                    "percentage": (count / max(total_interactions, 1)) * 100
                })
            except Exception as e:
                logger.error(f"Error processing learning style {style_id}: {e}")
                continue
        
        return usage_data
    
    def _get_preferred_learning_style(self, style_usage: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the most preferred learning style"""
        if not style_usage:
            return None
        
        return {
            "id": style_usage[0]["style_id"],
            "name": style_usage[0]["style_name"],
            "usage_percentage": style_usage[0]["percentage"]
        }
    
    def _analyze_module_performance(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze performance by module"""
        module_data = defaultdict(list)
        
        # Group interactions by module
        for interaction in interactions:
            module_id = interaction.get("module_id")
            if module_id:
                module_data[module_id].append(interaction)
        
        module_analytics = []
        
        for module_id, module_interactions in module_data.items():
            total_interactions = len(module_interactions)
            completed = sum(1 for i in module_interactions if i.get("end_time"))
            total_time = sum(i.get("time_spent", 0) for i in module_interactions if i.get("time_spent"))
            
            # Calculate satisfaction for this module
            helpful_ratings = [i.get("helpful") for i in module_interactions if i.get("helpful") is not None]
            satisfaction = sum(1 for r in helpful_ratings if r) / max(len(helpful_ratings), 1) if helpful_ratings else None
            
            # Average time per interaction
            avg_time = total_time / max(completed, 1)
            
            # Last accessed
            last_accessed = max(
                (i.get("start_time") for i in module_interactions if i.get("start_time")),
                default=None
            )
            
            module_analytics.append({
                "module_id": module_id,
                "time_spent": total_time / 60,  # Convert to minutes
                "interactions": total_interactions,
                "completed_interactions": completed,
                "avg_time_per_query": avg_time,
                "satisfaction_rate": satisfaction * 100 if satisfaction else None,
                "last_accessed": last_accessed.isoformat() if last_accessed else None
            })
        
        # Sort by time spent (descending)
        module_analytics.sort(key=lambda x: x["time_spent"], reverse=True)
        return module_analytics
    
    def _analyze_daily_activity(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze daily activity patterns"""
        daily_data = defaultdict(lambda: {"date": None, "count": 0, "time_spent": 0})
        
        for interaction in interactions:
            start_time = interaction.get("start_time")
            if not start_time:
                continue
            
            # Get date string
            date_str = start_time.strftime("%Y-%m-%d") if isinstance(start_time, datetime) else str(start_time)[:10]
            
            daily_data[date_str]["date"] = date_str
            daily_data[date_str]["count"] += 1
            daily_data[date_str]["time_spent"] += interaction.get("time_spent", 0) / 60  # Convert to minutes
        
        # Convert to list and sort by date
        daily_activity = sorted(daily_data.values(), key=lambda x: x["date"])
        return daily_activity[-30:]  # Last 30 days
    
    def _analyze_performance_trends(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(interactions) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort interactions by time
        sorted_interactions = sorted(
            [i for i in interactions if i.get("start_time")],
            key=lambda x: x["start_time"]
        )
        
        if len(sorted_interactions) < 2:
            return {"trend": "insufficient_data"}
        
        # Analyze recent vs older performance
        mid_point = len(sorted_interactions) // 2
        older_interactions = sorted_interactions[:mid_point]
        recent_interactions = sorted_interactions[mid_point:]
        
        # Calculate metrics for both periods
        older_satisfaction = self._calculate_satisfaction_rate(older_interactions)
        recent_satisfaction = self._calculate_satisfaction_rate(recent_interactions)
        
        older_avg_time = np.mean([i.get("time_spent", 0) for i in older_interactions if i.get("time_spent")])
        recent_avg_time = np.mean([i.get("time_spent", 0) for i in recent_interactions if i.get("time_spent")])
        
        # Determine trend
        satisfaction_change = recent_satisfaction - older_satisfaction if older_satisfaction and recent_satisfaction else 0
        time_change = recent_avg_time - older_avg_time
        
        if satisfaction_change > 0.1:
            trend = "improving"
        elif satisfaction_change < -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "satisfaction_change": satisfaction_change,
            "avg_time_change": time_change,
            "older_period_satisfaction": older_satisfaction,
            "recent_period_satisfaction": recent_satisfaction
        }
    
    def _calculate_satisfaction_rate(self, interactions: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate satisfaction rate for a set of interactions"""
        helpful_ratings = [i.get("helpful") for i in interactions if i.get("helpful") is not None]
        if not helpful_ratings:
            return None
        return sum(1 for r in helpful_ratings if r) / len(helpful_ratings)
    
    def _generate_personalized_recommendations(self, interactions: List[Dict[str, Any]], 
                                             style_usage: List[Dict[str, Any]],
                                             module_analytics: List[Dict[str, Any]],
                                             satisfaction_rate: Optional[float]) -> List[Dict[str, Any]]:
        """Generate personalized learning recommendations"""
        recommendations = []
        
        try:
            # Recommendation based on satisfaction rate
            if satisfaction_rate is not None and satisfaction_rate < 0.7:
                if style_usage:
                    current_style = style_usage[0]["style_id"]
                    # Suggest trying a different learning style
                    recommendations.append({
                        "type": "learning_style",
                        "priority": "high",
                        "title": "Try a Different Learning Style",
                        "description": f"Your current satisfaction rate is {satisfaction_rate*100:.1f}%. Consider experimenting with different learning styles.",
                        "action": "Switch from your current style to see if visual or quiz-based learning works better for you."
                    })
            
            # Recommendation based on study time
            avg_session_time = np.mean([i.get("time_spent", 0) for i in interactions if i.get("time_spent")])
            if avg_session_time < 30:  # Less than 30 seconds per query
                recommendations.append({
                    "type": "engagement",
                    "priority": "medium",
                    "title": "Increase Engagement Time",
                    "description": "You tend to move quickly through material. Consider spending more time on complex topics.",
                    "action": "Try to engage more deeply with responses and ask follow-up questions."
                })
            
            # Module-specific recommendations
            for module in module_analytics[:3]:  # Top 3 modules
                if module.get("satisfaction_rate") is not None and module["satisfaction_rate"] < 60:
                    recommendations.append({
                        "type": "module_focus",
                        "priority": "high",
                        "title": f"Focus on {module['module_id']}",
                        "description": f"Your satisfaction rate in {module['module_id']} is {module['satisfaction_rate']:.1f}%.",
                        "action": f"Consider reviewing fundamental concepts in {module['module_id']} or asking more specific questions."
                    })
            
            # Activity pattern recommendations
            daily_activity = self._analyze_daily_activity(interactions)
            if len(daily_activity) > 0:
                recent_activity = daily_activity[-7:]  # Last 7 days
                avg_daily_interactions = np.mean([day["count"] for day in recent_activity])
                
                if avg_daily_interactions < 1:
                    recommendations.append({
                        "type": "consistency",
                        "priority": "medium",
                        "title": "Increase Study Consistency",
                        "description": "Regular study sessions can improve learning outcomes.",
                        "action": "Try to interact with the system at least once daily, even for quick questions."
                    })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _identify_strengths_and_improvements(self, interactions: List[Dict[str, Any]],
                                           module_analytics: List[Dict[str, Any]],
                                           satisfaction_rate: Optional[float]) -> Tuple[List[str], List[str]]:
        """Identify student strengths and areas for improvement"""
        strengths = []
        improvements = []
        
        try:
            # Strengths based on satisfaction rate
            if satisfaction_rate is not None and satisfaction_rate > 0.8:
                strengths.append(f"High overall satisfaction ({satisfaction_rate*100:.1f}%) indicates effective learning approach")
            
            # Strengths based on module performance
            high_performing_modules = [m for m in module_analytics if m.get("satisfaction_rate", 0) > 80]
            if high_performing_modules:
                module_names = [m["module_id"] for m in high_performing_modules[:3]]
                strengths.append(f"Strong performance in {', '.join(module_names)}")
            
            # Strengths based on consistency
            if len(interactions) > 20:
                strengths.append("Consistent engagement with learning materials")
            
            # Areas for improvement based on low satisfaction
            if satisfaction_rate is not None and satisfaction_rate < 0.6:
                improvements.append("Overall satisfaction could be improved - consider trying different learning approaches")
            
            # Areas for improvement based on module performance
            low_performing_modules = [m for m in module_analytics if m.get("satisfaction_rate", 100) < 60]
            if low_performing_modules:
                module_names = [m["module_id"] for m in low_performing_modules[:3]]
                improvements.append(f"Focus needed in {', '.join(module_names)}")
            
            # Areas for improvement based on engagement time
            avg_time = np.mean([i.get("time_spent", 0) for i in interactions if i.get("time_spent")])
            if avg_time < 20:
                improvements.append("Consider spending more time engaging with complex topics")
            
        except Exception as e:
            logger.error(f"Error identifying strengths and improvements: {e}")
        
        return strengths, improvements
    
    # System-wide Analytics
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics across all students"""
        try:
            profiles_collection = get_student_profiles_collection()
            interactions_collection = get_student_interactions_collection()
            
            # Basic counts
            total_students = profiles_collection.count_documents({})
            total_interactions = interactions_collection.count_documents({})
            
            # Active students (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            active_students = profiles_collection.count_documents({
                "last_active": {"$gte": thirty_days_ago}
            })
            
            # Learning style distribution
            style_pipeline = [
                {"$group": {"_id": "$learning_style_id", "count": {"$sum": 1}}}
            ]
            style_results = list(interactions_collection.aggregate(style_pipeline))
            learning_style_distribution = {item["_id"]: item["count"] for item in style_results}
            
            # Module popularity
            module_pipeline = [
                {"$match": {"module_id": {"$ne": None}}},
                {"$group": {"_id": "$module_id", "count": {"$sum": 1}}}
            ]
            module_results = list(interactions_collection.aggregate(module_pipeline))
            module_popularity = sorted(
                [(item["_id"], item["count"]) for item in module_results],
                key=lambda x: x[1], reverse=True
            )
            
            # Satisfaction metrics
            satisfaction_pipeline = [
                {"$match": {"helpful": {"$ne": None}}},
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "positive": {"$sum": {"$cond": ["$helpful", 1, 0]}}
                }}
            ]
            satisfaction_results = list(interactions_collection.aggregate(satisfaction_pipeline))
            
            overall_satisfaction = 0
            if satisfaction_results:
                result = satisfaction_results[0]
                overall_satisfaction = (result["positive"] / result["total"]) * 100
            
            # Daily activity trend (last 30 days)
            daily_activity = []
            for i in range(30):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                
                day_interactions = interactions_collection.count_documents({
                    "start_time": {
                        "$gte": date.replace(hour=0, minute=0, second=0),
                        "$lt": date.replace(hour=23, minute=59, second=59)
                    }
                })
                
                daily_activity.append({
                    "date": date_str,
                    "interactions": day_interactions
                })
            
            daily_activity.reverse()  # Chronological order
            
            return {
                "generated_at": datetime.now().isoformat(),
                "student_metrics": {
                    "total_students": total_students,
                    "active_students_30d": active_students,
                    "activity_rate": (active_students / max(total_students, 1)) * 100
                },
                "interaction_metrics": {
                    "total_interactions": total_interactions,
                    "avg_interactions_per_student": total_interactions / max(total_students, 1),
                    "overall_satisfaction_rate": overall_satisfaction
                },
                "learning_preferences": {
                    "style_distribution": learning_style_distribution,
                    "popular_modules": module_popularity[:10]
                },
                "activity_trends": {
                    "daily_activity_30d": daily_activity
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system analytics: {e}")
            return {"error": f"Failed to generate system analytics: {str(e)}"}
    
    # Anxiety Analytics Integration
    
    async def get_student_wellness_summary(self, student_id: str) -> Dict[str, Any]:
        """Get wellness summary including anxiety analytics"""
        try:
            # Get anxiety assessments
            anxiety_collection = get_mongo_collection("anxiety_assessments")
            
            # Last 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_assessments = list(anxiety_collection.find({
                "student_id": student_id,
                "timestamp": {"$gte": thirty_days_ago.isoformat()}
            }).sort("timestamp", -1))
            
            if not recent_assessments:
                return {"message": "No recent wellness data available"}
            
            # Analyze anxiety levels
            anxiety_levels = [a.get("prediction", {}).get("anxiety_level") for a in recent_assessments]
            anxiety_distribution = Counter(anxiety_levels)
            
            # Calculate trend
            if len(recent_assessments) >= 2:
                recent_level = recent_assessments[0].get("prediction", {}).get("anxiety_level")
                older_level = recent_assessments[-1].get("prediction", {}).get("anxiety_level")
                
                level_order = ["No Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"]
                recent_index = level_order.index(recent_level) if recent_level in level_order else 1
                older_index = level_order.index(older_level) if older_level in level_order else 1
                
                if recent_index < older_index:
                    trend = "improving"
                elif recent_index > older_index:
                    trend = "concerning"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Recommendations used
            all_recommendations = []
            for assessment in recent_assessments:
                recommendations = assessment.get("recommendations", [])
                all_recommendations.extend([r.get("type") for r in recommendations])
            
            recommendation_usage = Counter(all_recommendations)
            
            return {
                "student_id": student_id,
                "period": "30_days",
                "total_assessments": len(recent_assessments),
                "anxiety_distribution": dict(anxiety_distribution),
                "current_level": recent_assessments[0].get("prediction", {}).get("anxiety_level") if recent_assessments else None,
                "trend": trend,
                "most_common_recommendations": dict(recommendation_usage.most_common(5)),
                "last_assessment": recent_assessments[0].get("timestamp") if recent_assessments else None
            }
            
        except Exception as e:
            logger.error(f"Error getting wellness summary: {e}")
            return {"error": f"Failed to generate wellness summary: {str(e)}"}
    
    # Export and Reporting
    
    async def export_student_data(self, student_id: str, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive student data"""
        try:
            # Get all student data
            profile = get_student_profiles_collection().find_one({"_id": student_id})
            interactions = list(get_student_interactions_collection().find({"student_id": student_id}))
            analytics = get_student_analytics_collection().find_one({"student_id": student_id})
            
            # Get anxiety data
            anxiety_data = list(get_mongo_collection("anxiety_assessments").find({"student_id": student_id}))
            
            # Generate report
            report = await self.generate_student_report(student_id)
            
            export_data = {
                "export_metadata": {
                    "student_id": student_id,
                    "generated_at": datetime.now().isoformat(),
                    "format": format,
                    "data_privacy_notice": "This data export contains personal learning analytics."
                },
                "profile": profile,
                "interactions_summary": {
                    "total_interactions": len(interactions),
                    "date_range": {
                        "first": min(i.get("start_time") for i in interactions if i.get("start_time")).isoformat() if interactions else None,
                        "last": max(i.get("start_time") for i in interactions if i.get("start_time")).isoformat() if interactions else None
                    }
                },
                "analytics_report": report,
                "wellness_summary": await self.get_student_wellness_summary(student_id),
                "raw_data": {
                    "interactions_count": len(interactions),
                    "anxiety_assessments_count": len(anxiety_data),
                    "analytics_last_updated": analytics.get("last_updated").isoformat() if analytics and analytics.get("last_updated") else None
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting student data: {e}")
            return {"error": f"Failed to export data: {str(e)}"}