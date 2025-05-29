"""
Chat endpoints for AI-powered Q&A with RAG functionality
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from datetime import datetime

from app.models.chat import (
    ChatQuery, ChatResponse, SearchRequest, SearchResults,
    ChatMessage, MessageRole, QuickResponse, ChatMetrics,
    ChatHealthCheck, ErrorResponse
)
from app.models.student import StudentFeedback
from app.services.chat_service import ChatService
from app.services.personalization_service import PersonalizationService
from app.routers.auth import get_current_student

logger = logging.getLogger(__name__)

router = APIRouter()
chat_service = ChatService()
personalization_service = PersonalizationService()


@router.on_event("startup")
async def startup_event():
    """Initialize chat service on startup"""
    try:
        await chat_service.initialize()
        logger.info("Chat service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat service: {e}")


@router.post("/query", response_model=ChatResponse)
async def chat_query(
    query: ChatQuery,
    background_tasks: BackgroundTasks,
    current_student: str = Depends(get_current_student)
):
    """Process a chat query with RAG"""
    try:
        # Validate student matches token
        if query.student_id != current_student:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Student ID mismatch"
            )
        
        logger.info(f"Processing chat query for student {current_student}")
        logger.debug(f"Query: {query.message[:100]}...")
        
        # Generate response using chat service
        response = await chat_service.generate_response(query)
        
        # Add background task to update analytics
        background_tasks.add_task(
            update_analytics_background,
            current_student,
            query,
            response
        )
        
        logger.info(f"Chat response generated in {response.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )


@router.post("/search", response_model=SearchResults)
async def search_content(
    search_request: SearchRequest,
    current_student: str = Depends(get_current_student)
):
    """Search for content in the knowledge base"""
    try:
        logger.info(f"Search request from student {current_student}")
        logger.debug(f"Search query: {search_request.query}")
        
        # Perform search
        results = await chat_service.search(search_request)
        
        logger.info(f"Search completed: {results.total_results} results in {results.processing_time:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform search"
        )


@router.post("/feedback")
async def submit_feedback(
    feedback: StudentFeedback,
    current_student: str = Depends(get_current_student)
):
    """Submit feedback for a chat interaction"""
    try:
        logger.info(f"Feedback received from student {current_student}")
        
        # Submit feedback through personalization service
        success = await personalization_service.submit_feedback(feedback)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to submit feedback"
            )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_id": feedback.interaction_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get("/history")
async def get_chat_history(
    limit: int = 50,
    current_student: str = Depends(get_current_student)
):
    """Get chat history for the current student"""
    try:
        logger.info(f"Getting chat history for student {current_student}")
        
        # Get chat history from service
        history = await chat_service.get_chat_history(current_student, limit)
        
        return {
            "success": True,
            "student_id": current_student,
            "history": history,
            "total_interactions": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat history"
        )


@router.post("/quick-response", response_model=QuickResponse)
async def get_quick_response(
    message: str,
    current_student: str = Depends(get_current_student)
):
    """Get a quick response without full RAG processing"""
    try:
        logger.info(f"Quick response request from student {current_student}")
        
        # Simple keyword-based quick responses
        quick_responses = {
            "hello": "Hello! How can I help you with your studies today?",
            "hi": "Hi there! What would you like to learn about?",
            "help": "I'm here to help! You can ask me questions about your course materials, or I can help assess your anxiety levels.",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "thank you": "You're very welcome! Feel free to ask me anything else.",
        }
        
        message_lower = message.lower().strip()
        
        # Check for quick responses
        for keyword, response in quick_responses.items():
            if keyword in message_lower:
                suggestions = [
                    "Ask me about course materials",
                    "Get help with specific topics",
                    "Check your anxiety levels",
                    "Review your learning progress"
                ]
                
                return QuickResponse(
                    message=response,
                    suggestions=suggestions,
                    response_type="quick"
                )
        
        # Default response for unrecognized patterns
        return QuickResponse(
            message="I'd be happy to help! Please ask me a specific question about your studies or use the full chat for detailed assistance.",
            suggestions=[
                "Use the main chat for detailed questions",
                "Ask about specific course topics",
                "Request study materials",
                "Get anxiety assessment"
            ],
            response_type="fallback"
        )
        
    except Exception as e:
        logger.error(f"Error generating quick response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quick response"
        )


@router.get("/suggestions")
async def get_chat_suggestions(
    module_id: Optional[str] = None,
    current_student: str = Depends(get_current_student)
):
    """Get personalized chat suggestions"""
    try:
        logger.info(f"Getting chat suggestions for student {current_student}")
        
        # Get student profile for personalization
        profile = await personalization_service.get_student_profile(current_student)
        
        # Base suggestions
        suggestions = [
            "Explain this concept in simple terms",
            "What are the key points I should remember?",
            "Can you provide examples for this topic?",
            "How does this relate to other concepts?",
            "What are common misconceptions about this?"
        ]
        
        # Personalize based on learning style
        if profile and profile.default_learning_style:
            style = profile.default_learning_style
            
            if style == "visual":
                suggestions.extend([
                    "Show me diagrams or visual representations",
                    "Can you describe this visually?",
                    "What would this look like in a chart?"
                ])
            elif style == "quiz":
                suggestions.extend([
                    "Quiz me on this topic",
                    "Create practice questions for me",
                    "Test my understanding"
                ])
            elif style == "eli5":
                suggestions.extend([
                    "Explain this like I'm a beginner",
                    "Use simple analogies",
                    "Break this down into basic steps"
                ])
        
        # Add module-specific suggestions if provided
        if module_id:
            suggestions.extend([
                f"What are the main topics in {module_id}?",
                f"Summary of {module_id} concepts",
                f"Practice problems for {module_id}"
            ])
        
        return {
            "success": True,
            "suggestions": suggestions[:10],  # Limit to top 10
            "personalized": profile is not None,
            "learning_style": profile.default_learning_style if profile else None
        }
        
    except Exception as e:
        logger.error(f"Error getting chat suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get suggestions"
        )


@router.get("/metrics", response_model=ChatMetrics)
async def get_chat_metrics(
    current_student: str = Depends(get_current_student)
):
    """Get chat metrics for the current student"""
    try:
        logger.info(f"Getting chat metrics for student {current_student}")
        
        # Get student analytics
        analytics = await personalization_service.get_student_analytics(current_student)
        
        if not analytics:
            # Return empty metrics if no data
            return ChatMetrics(
                total_conversations=0,
                avg_conversation_length=0.0,
                avg_response_time=0.0,
                popular_topics=[],
                learning_style_usage={},
                user_satisfaction=None,
                timestamp=datetime.now()
            )
        
        # Convert analytics to chat metrics
        popular_topics = []
        for module_analytics in analytics.module_analytics:
            popular_topics.append({
                "topic": module_analytics.module_id,
                "interactions": module_analytics.total_interactions,
                "avg_time": module_analytics.avg_time_per_query
            })
        
        return ChatMetrics(
            total_conversations=analytics.summary.total_interactions,
            avg_conversation_length=analytics.summary.avg_session_duration,
            avg_response_time=2.5,  # Placeholder - could be calculated from response times
            popular_topics=popular_topics,
            learning_style_usage=analytics.learning_style_usage,
            user_satisfaction=analytics.summary.satisfaction_rate,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting chat metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get chat metrics"
        )


@router.get("/health", response_model=ChatHealthCheck)
async def chat_health_check():
    """Health check for chat service"""
    try:
        health_status = await chat_service.health_check()
        
        return ChatHealthCheck(
            chat_service_status=health_status.get("chat_service", "unknown"),
            llm_service_status=health_status.get("llm_service", "unknown"),
            embedding_service_status=health_status.get("embedding_service", "unknown"),
            database_status=health_status.get("database_status", "unknown"),
            response_time=0.5,  # Placeholder
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat health check failed: {e}")
        return ChatHealthCheck(
            chat_service_status="unhealthy",
            llm_service_status="unknown",
            embedding_service_status="unknown",
            database_status="unknown",
            response_time=0.0,
            timestamp=datetime.now()
        )


# Background task functions
async def update_analytics_background(
    student_id: str,
    query: ChatQuery,
    response: ChatResponse
):
    """Background task to update analytics after chat interaction"""
    try:
        # This would normally update detailed analytics
        # For now, we just log the interaction
        logger.info(f"Analytics updated for student {student_id}: {response.processing_time:.2f}s")
        
        # Could add:
        # - Update response time analytics
        # - Track popular topics
        # - Update learning style effectiveness
        # - Store interaction patterns
        
    except Exception as e:
        logger.error(f"Error updating analytics in background: {e}")


# WebSocket support (for real-time chat)
@router.websocket("/ws/{student_id}")
async def websocket_chat(websocket, student_id: str):
    """WebSocket endpoint for real-time chat (future implementation)"""
    # Placeholder for WebSocket chat functionality
    # This would enable real-time bidirectional communication
    await websocket.accept()
    await websocket.send_text("WebSocket chat not yet implemented")
    await websocket.close()