"""
Student Personalization Service - Adapted from student_personalization.py
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from app.models.student import (
    StudentRegistration, StudentProfile, StudentProfileUpdate,
    StudentInteractionCreate, StudentInteraction, StudentInteractionUpdate,
    StudentFeedback, LearningStyleEnum, StudentAnalyticsSummary,
    StudentDetailedAnalytics, ModuleAnalytics
)
from app.core.config import get_settings, learning_styles
from app.core.database import (
    get_student_profiles_collection, get_student_interactions_collection,
    get_student_analytics_collection
)

logger = logging.getLogger(__name__)
settings = get_settings()


class PersonalizationService:
    """Service for student personalization and learning analytics"""
    
    def __init__(self):
        self.config = settings
        self.learning_styles = learning_styles
        
    # Student Profile Management
    
    async def create_student_profile(self, registration: StudentRegistration) -> StudentProfile:
        """Create a new student profile"""
        try:
            # Check if student already exists
            profiles_collection = get_student_profiles_collection()
            existing_student = profiles_collection.find_one({"_id": registration.student_id})
            
            if existing_student:
                logger.info(f"Student profile already exists for {registration.student_id}")
                return StudentProfile.model_validate(existing_student)
            
            # Create new profile
            now = datetime.now()
            profile_data = {
                "_id": registration.student_id,
                "student_id": registration.student_id,
                "name": registration.name,
                "email": registration.email,
                "default_learning_style": registration.default_learning_style,
                "created_at": now,
                "last_active": now,
                "preferences": {},
                "modules_accessed": [],
                "total_interactions": 0
            }
            
            profiles_collection.insert_one(profile_data)
            
            logger.info(f"Created new student profile for {registration.student_id}")
            return StudentProfile.model_validate(profile_data)
            
        except Exception as e:
            logger.error(f"Error creating student profile: {e}")
            raise
    
    async def get_student_profile(self, student_id: str) -> Optional[StudentProfile]:
        """Get student profile by ID"""
        try:
            profiles_collection = get_student_profiles_collection()
            profile_data = profiles_collection.find_one({"_id": student_id})
            
            if profile_data:
                return StudentProfile.model_validate(profile_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting student profile: {e}")
            return None
    
    async def update_student_profile(self, student_id: str, 
                                   update_data: StudentProfileUpdate) -> Optional[StudentProfile]:
        """Update student profile"""
        try:
            profiles_collection = get_student_profiles_collection()
            
            # Prepare update document
            update_doc = {"$set": {"last_active": datetime.now()}}
            
            if update_data.name is not None:
                update_doc["$set"]["name"] = update_data.name
            if update_data.email is not None:
                update_doc["$set"]["email"] = update_data.email
            if update_data.default_learning_style is not None:
                update_doc["$set"]["default_learning_style"] = update_data.default_learning_style
            if update_data.preferences is not None:
                update_doc["$set"]["preferences"] = update_data.preferences
            
            # Update profile
            result = profiles_collection.update_one(
                {"_id": student_id},
                update_doc
            )
            
            if result.modified_count > 0:
                return await self.get_student_profile(student_id)
            return None
            
        except Exception as e:
            logger.error(f"Error updating student profile: {e}")
            return None
    
    async def update_learning_style(self, student_id: str, 
                                  learning_style: LearningStyleEnum) -> bool:
        """Update student's preferred learning style"""
        try:
            profiles_collection = get_student_profiles_collection()
            
            result = profiles_collection.update_one(
                {"_id": student_id},
                {
                    "$set": {
                        "default_learning_style": learning_style,
                        "last_active": datetime.now()
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating learning style: {e}")
            return False
    
    # Learning Style Management
    
    def get_all_learning_styles(self) -> List[Dict[str, Any]]:
        """Get all available learning styles"""
        return self.learning_styles.LEARNING_STYLES
    
    def get_learning_style_by_id(self, style_id: str) -> Optional[Dict[str, Any]]:
        """Get learning style by ID"""
        return self.learning_styles.get_learning_style_by_id(style_id)
    
    def format_query_with_learning_style(self, query: str, learning_style_id: str) -> str:
        """Format query with learning style prompt template"""
        return self.learning_styles.get_prompt_template(learning_style_id) + "\n\n" + query
    
    # Interaction Management
    
    async def start_interaction(self, interaction_data: StudentInteractionCreate) -> str:
        """Start a new student interaction"""
        try:
            # Get student profile
            profile = await self.get_student_profile(interaction_data.student_id)
            if not profile:
                raise ValueError(f"Student profile not found: {interaction_data.student_id}")
            
            # Use provided learning style or default
            learning_style = interaction_data.learning_style_id or profile.default_learning_style
            
            # Generate interaction ID
            interaction_id = str(uuid.uuid4())
            now = datetime.now()
            
            # Create interaction document
            interaction_doc = {
                "_id": interaction_id,
                "interaction_id": interaction_id,
                "student_id": interaction_data.student_id,
                "query": interaction_data.query,
                "learning_style_id": learning_style,
                "module_id": interaction_data.module_id,
                "lecture_code": interaction_data.lecture_code,
                "lecture_number": None,
                "start_time": now,
                "end_time": None,
                "time_spent": None,
                "feedback": None,
                "helpful": None,
                "follow_up_queries": [],
                "retrieved_sources": []
            }
            
            # Store interaction
            interactions_collection = get_student_interactions_collection()
            interactions_collection.insert_one(interaction_doc)
            
            # Update student profile
            profiles_collection = get_student_profiles_collection()
            update_doc = {
                "$set": {"last_active": now},
                "$inc": {"total_interactions": 1}
            }
            
            if interaction_data.module_id:
                update_doc["$addToSet"] = {"modules_accessed": interaction_data.module_id}
            
            profiles_collection.update_one({"_id": interaction_data.student_id}, update_doc)
            
            logger.info(f"Started interaction {interaction_id} for student {interaction_data.student_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error starting interaction: {e}")
            raise
    
    async def end_interaction(self, interaction_id: str, 
                            update_data: StudentInteractionUpdate) -> bool:
        """End a student interaction"""
        try:
            interactions_collection = get_student_interactions_collection()
            
            # Get interaction
            interaction = interactions_collection.find_one({"_id": interaction_id})
            if not interaction:
                logger.error(f"Interaction {interaction_id} not found")
                return False
            
            now = datetime.now()
            
            # Calculate time spent if not provided
            time_spent = update_data.time_spent
            if time_spent is None and interaction.get("start_time"):
                time_diff = now - interaction["start_time"]
                time_spent = min(time_diff.total_seconds(), settings.SESSION_TIMEOUT)
            
            # Update interaction
            update_doc = {
                "$set": {
                    "end_time": update_data.end_time or now,
                    "time_spent": time_spent
                }
            }
            
            if update_data.feedback is not None:
                update_doc["$set"]["feedback"] = update_data.feedback
            if update_data.helpful is not None:
                update_doc["$set"]["helpful"] = update_data.helpful
            if update_data.retrieved_sources is not None:
                update_doc["$set"]["retrieved_sources"] = update_data.retrieved_sources
            
            result = interactions_collection.update_one(
                {"_id": interaction_id},
                update_doc
            )
            
            if result.modified_count > 0:
                # Trigger analytics update
                await self._update_student_analytics(interaction["student_id"])
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error ending interaction: {e}")
            return False
    
    async def submit_feedback(self, feedback: StudentFeedback) -> bool:
        """Submit feedback for an interaction"""
        try:
            # Update interaction with feedback
            interactions_collection = get_student_interactions_collection()
            
            result = interactions_collection.update_one(
                {"_id": feedback.interaction_id},
                {
                    "$set": {
                        "feedback": feedback.feedback,
                        "helpful": feedback.helpful,
                        "rating": feedback.rating
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False
    
    # Analytics and Insights
    
    async def _update_student_analytics(self, student_id: str) -> bool:
        """Update analytics for a student"""
        try:
            # Get all interactions for this student
            interactions_collection = get_student_interactions_collection()
            interactions = list(interactions_collection.find({"student_id": student_id}))
            
            if not interactions:
                return False
            
            # Calculate overall analytics
            now = datetime.now()
            total_interactions = len(interactions)
            completed_interactions = sum(1 for i in interactions if i.get("end_time"))
            total_time_spent = sum(i.get("time_spent", 0) for i in interactions if i.get("time_spent"))
            avg_time_per_query = total_time_spent / max(completed_interactions, 1)
            
            # Calculate satisfaction rate
            helpful_ratings = [i.get("helpful") for i in interactions if i.get("helpful") is not None]
            satisfaction_rate = sum(1 for r in helpful_ratings if r) / max(len(helpful_ratings), 1) if helpful_ratings else None
            
            # Learning style preferences
            style_counts = {}
            for interaction in interactions:
                style = interaction.get("learning_style_id")
                if style:
                    style_counts[style] = style_counts.get(style, 0) + 1
            
            preferred_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Module-specific analytics
            module_interactions = {}
            for interaction in interactions:
                module_id = interaction.get("module_id")
                if module_id:
                    if module_id not in module_interactions:
                        module_interactions[module_id] = []
                    module_interactions[module_id].append(interaction)
            
            module_analytics = []
            for module_id, mod_interactions in module_interactions.items():
                mod_total = len(mod_interactions)
                mod_completed = sum(1 for i in mod_interactions if i.get("end_time"))
                mod_time_spent = sum(i.get("time_spent", 0) for i in mod_interactions if i.get("time_spent"))
                mod_avg_time = mod_time_spent / max(mod_completed, 1)
                
                mod_helpful = [i.get("helpful") for i in mod_interactions if i.get("helpful") is not None]
                mod_satisfaction = sum(1 for r in mod_helpful if r) / max(len(mod_helpful), 1) if mod_helpful else None
                
                # Module learning style preferences
                mod_style_counts = {}
                for interaction in mod_interactions:
                    style = interaction.get("learning_style_id")
                    if style:
                        mod_style_counts[style] = mod_style_counts.get(style, 0) + 1
                
                mod_preferred_styles = sorted(mod_style_counts.items(), key=lambda x: x[1], reverse=True)
                
                module_analytics.append({
                    "module_id": module_id,
                    "total_interactions": mod_total,
                    "completed_interactions": mod_completed,
                    "total_time_spent": mod_time_spent,
                    "avg_time_per_query": mod_avg_time,
                    "satisfaction_rate": mod_satisfaction,
                    "preferred_styles": mod_preferred_styles[:2]
                })
            
            # Create analytics document
            analytics_doc = {
                "student_id": student_id,
                "last_updated": now,
                "total_interactions": total_interactions,
                "completed_interactions": completed_interactions,
                "total_time_spent": total_time_spent,
                "avg_time_per_query": avg_time_per_query,
                "satisfaction_rate": satisfaction_rate,
                "preferred_styles": preferred_styles[:3],
                "module_analytics": module_analytics,
                "interaction_history": [
                    {
                        "date": i.get("start_time"),
                        "module_id": i.get("module_id"),
                        "lecture_code": i.get("lecture_code"),
                        "time_spent": i.get("time_spent"),
                        "helpful": i.get("helpful"),
                        "learning_style_id": i.get("learning_style_id")
                    }
                    for i in interactions if i.get("start_time")
                ]
            }
            
            # Upsert analytics
            analytics_collection = get_student_analytics_collection()
            analytics_collection.update_one(
                {"student_id": student_id},
                {"$set": analytics_doc},
                upsert=True
            )
            
            logger.info(f"Updated analytics for student {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
            return False
    
    async def get_student_analytics(self, student_id: str) -> Optional[StudentDetailedAnalytics]:
        """Get detailed analytics for a student"""
        try:
            # Check if analytics need to be updated
            analytics_collection = get_student_analytics_collection()
            analytics = analytics_collection.find_one({"student_id": student_id})
            
            if not analytics or (datetime.now() - analytics.get("last_updated", datetime.min)).total_seconds() > 24 * 3600:
                await self._update_student_analytics(student_id)
                analytics = analytics_collection.find_one({"student_id": student_id})
            
            if not analytics:
                return None
            
            # Get student profile
            profile = await self.get_student_profile(student_id)
            if not profile:
                return None
            
            # Build summary
            summary = StudentAnalyticsSummary(
                student_id=student_id,
                total_interactions=analytics.get("total_interactions", 0),
                total_study_time=analytics.get("total_time_spent", 0) / 60,  # Convert to minutes
                avg_session_duration=analytics.get("avg_time_per_query", 0) / 60,
                satisfaction_rate=analytics.get("satisfaction_rate"),
                preferred_learning_style=None,  # Will be set below
                modules_studied=[m["module_id"] for m in analytics.get("module_analytics", [])],
                last_activity=profile.last_active
            )
            
            # Add preferred learning style
            preferred_styles = analytics.get("preferred_styles", [])
            if preferred_styles:
                style_id = preferred_styles[0][0]
                style_info = self.get_learning_style_by_id(style_id)
                if style_info:
                    summary.preferred_learning_style = {
                        "id": style_id,
                        "name": style_info["name"],
                        "description": style_info["description"]
                    }
            
            # Build module analytics
            module_analytics = []
            for mod_data in analytics.get("module_analytics", []):
                module_analytics.append(ModuleAnalytics(
                    module_id=mod_data["module_id"],
                    total_interactions=mod_data["total_interactions"],
                    total_time_spent=mod_data["total_time_spent"] / 60,
                    avg_time_per_query=mod_data["avg_time_per_query"],
                    satisfaction_rate=mod_data.get("satisfaction_rate"),
                    preferred_styles=mod_data.get("preferred_styles", []),
                    last_accessed=None  # Could be calculated from interaction history
                ))
            
            # Learning style usage
            learning_style_usage = {}
            for style_id, count in analytics.get("preferred_styles", []):
                try:
                    learning_style_usage[LearningStyleEnum(style_id)] = count
                except ValueError:
                    continue  # Skip invalid learning style IDs
            
            # Daily activity (simplified)
            daily_activity = []
            for interaction in analytics.get("interaction_history", [])[-30:]:  # Last 30
                if interaction.get("date"):
                    daily_activity.append({
                        "date": interaction["date"].strftime("%Y-%m-%d") if isinstance(interaction["date"], datetime) else str(interaction["date"]),
                        "interactions": 1,
                        "time_spent": interaction.get("time_spent", 0) / 60
                    })
            
            # Generate recommendations (simplified)
            recommendations = await self._generate_recommendations(student_id, analytics)
            
            return StudentDetailedAnalytics(
                student_id=student_id,
                summary=summary,
                module_analytics=module_analytics,
                learning_style_usage=learning_style_usage,
                daily_activity=daily_activity,
                weekly_activity=[],  # Could be implemented
                strengths=[],  # Could be implemented
                areas_for_improvement=[],  # Could be implemented
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error getting student analytics: {e}")
            return None
    
    async def _generate_recommendations(self, student_id: str, analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations"""
        recommendations = []
        
        try:
            # Based on average time per query
            avg_time = analytics.get("avg_time_per_query", 0)
            if avg_time < 20:  # Less than 20 seconds
                recommendations.append({
                    "type": "study_pace",
                    "title": "Consider Deeper Engagement",
                    "description": "You seem to move quickly through material. Consider spending more time on complex concepts.",
                    "priority": "medium"
                })
            
            # Based on satisfaction rate
            satisfaction = analytics.get("satisfaction_rate")
            if satisfaction and satisfaction < 0.7:
                # Find most used learning style
                preferred_styles = analytics.get("preferred_styles", [])
                if preferred_styles:
                    current_style = preferred_styles[0][0]
                    # Recommend trying a different style
                    all_styles = self.get_all_learning_styles()
                    for style in all_styles:
                        if style["id"] != current_style:
                            recommendations.append({
                                "type": "learning_style",
                                "style_id": style["id"],
                                "title": f"Try {style['name']} Learning Style",
                                "description": style["description"],
                                "reason": "This learning style might help you understand concepts better.",
                                "priority": "high"
                            })
                            break
            
            # Based on module performance
            module_analytics = analytics.get("module_analytics", [])
            for module in module_analytics:
                if module.get("satisfaction_rate") and module["satisfaction_rate"] < 0.6:
                    recommendations.append({
                        "type": "module_focus",
                        "title": f"Focus on {module['module_id']}",
                        "description": f"Consider reviewing materials for {module['module_id']} module.",
                        "module_id": module["module_id"],
                        "priority": "medium"
                    })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    async def get_module_analytics(self, student_id: str, module_id: str) -> Optional[ModuleAnalytics]:
        """Get analytics for a specific module"""
        try:
            analytics = await self.get_student_analytics(student_id)
            if not analytics:
                return None
            
            for module_analytics in analytics.module_analytics:
                if module_analytics.module_id == module_id:
                    return module_analytics
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting module analytics: {e}")
            return None
    
    # Utility Methods
    
    async def check_session_timeout(self, student_id: str) -> bool:
        """Check if student session has timed out"""
        try:
            profile = await self.get_student_profile(student_id)
            if not profile:
                return True
            
            time_diff = datetime.now() - profile.last_active
            return time_diff.total_seconds() > settings.SESSION_TIMEOUT
            
        except Exception as e:
            logger.error(f"Error checking session timeout: {e}")
            return True
    
    async def update_last_activity(self, student_id: str) -> bool:
        """Update student's last activity timestamp"""
        try:
            profiles_collection = get_student_profiles_collection()
            result = profiles_collection.update_one(
                {"_id": student_id},
                {"$set": {"last_active": datetime.now()}}
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating last activity: {e}")
            return False