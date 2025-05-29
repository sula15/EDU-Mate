"""
Pydantic models for chat and RAG-related data structures
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

from .student import LearningStyleEnum


class MessageRole(str, Enum):
    """Enum for message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceType(str, Enum):
    """Enum for source types"""
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


class ChatMessage(BaseModel):
    """Model for individual chat messages"""
    role: MessageRole = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatQuery(BaseModel):
    """Model for incoming chat queries"""
    student_id: str = Field(..., description="Student identifier")
    message: str = Field(..., min_length=1, description="User's message/query")
    learning_style: Optional[LearningStyleEnum] = Field(None, description="Preferred learning style for this query")
    module_id: Optional[str] = Field(None, description="Current module context")
    include_images: bool = Field(default=True, description="Whether to include image results")
    include_sources: bool = Field(default=False, description="Whether to include text sources")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Previous conversation context")


class RetrievedSource(BaseModel):
    """Model for retrieved source information"""
    source_id: str = Field(..., description="Unique source identifier")
    source_type: SourceType = Field(..., description="Type of source (text/image)")
    content: str = Field(..., description="Source content or description")
    module_code: Optional[str] = Field(None, description="Module code")
    module_name: Optional[str] = Field(None, description="Module name")
    lecture_number: Optional[int] = Field(None, description="Lecture number")
    lecture_title: Optional[str] = Field(None, description="Lecture title")
    lecture_code: Optional[str] = Field(None, description="Lecture code")
    page_number: Optional[int] = Field(None, description="Page number")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ImageResult(BaseModel):
    """Model for image search results"""
    image_id: str = Field(..., description="Unique image identifier")
    image_data: str = Field(..., description="Base64 encoded image data")
    similarity_score: float = Field(..., description="Similarity score")
    lecture_code: Optional[str] = Field(None, description="Lecture code")
    lecture_number: Optional[int] = Field(None, description="Lecture number")
    lecture_title: Optional[str] = Field(None, description="Lecture title")
    module_id: Optional[str] = Field(None, description="Module identifier")
    page_number: Optional[int] = Field(None, description="Page number")
    text_description: Optional[str] = Field(None, description="Associated text description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatResponse(BaseModel):
    """Model for chat response"""
    response_id: str = Field(..., description="Unique response identifier")
    interaction_id: Optional[str] = Field(None, description="Associated interaction ID")
    answer_text: str = Field(..., description="Generated answer text")
    learning_style_used: LearningStyleEnum = Field(..., description="Learning style applied")
    text_sources: List[RetrievedSource] = Field(default_factory=list, description="Retrieved text sources")
    image_results: List[ImageResult] = Field(default_factory=list, description="Retrieved image results")
    processing_time: float = Field(..., description="Response processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatSession(BaseModel):
    """Model for chat session data"""
    session_id: str = Field(..., description="Unique session identifier")
    student_id: str = Field(..., description="Student identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Session messages")
    current_module: Optional[str] = Field(None, description="Current module context")
    learning_style: LearningStyleEnum = Field(..., description="Session learning style")
    started_at: datetime = Field(default_factory=datetime.now, description="Session start time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    is_active: bool = Field(default=True, description="Whether session is active")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchRequest(BaseModel):
    """Model for search requests"""
    query: str = Field(..., min_length=1, description="Search query")
    search_type: Optional[SourceType] = Field(SourceType.MIXED, description="Type of search (text/image/mixed)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    module_filter: Optional[str] = Field(None, description="Filter by module")
    lecture_filter: Optional[int] = Field(None, description="Filter by lecture number")


class SearchResults(BaseModel):
    """Model for search results"""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results")
    text_results: List[RetrievedSource] = Field(default_factory=list, description="Text search results")
    image_results: List[ImageResult] = Field(default_factory=list, description="Image search results")
    processing_time: float = Field(..., description="Search processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Search timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationSummary(BaseModel):
    """Model for conversation summary"""
    session_id: str = Field(..., description="Session identifier")
    student_id: str = Field(..., description="Student identifier")
    total_messages: int = Field(..., description="Total number of messages")
    duration: float = Field(..., description="Conversation duration in minutes")
    topics_discussed: List[str] = Field(default_factory=list, description="Main topics discussed")
    modules_referenced: List[str] = Field(default_factory=list, description="Modules referenced")
    learning_style_used: LearningStyleEnum = Field(..., description="Learning style used")
    satisfaction_indicators: Dict[str, Any] = Field(default_factory=dict, description="Satisfaction indicators")
    timestamp: datetime = Field(default_factory=datetime.now, description="Summary timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QuickResponse(BaseModel):
    """Model for quick responses without full processing"""
    message: str = Field(..., description="Quick response message")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    response_type: str = Field(default="quick", description="Type of response")


class ChatMetrics(BaseModel):
    """Model for chat metrics and analytics"""
    total_conversations: int = Field(..., description="Total number of conversations")
    avg_conversation_length: float = Field(..., description="Average conversation length")
    avg_response_time: float = Field(..., description="Average response time in seconds")
    popular_topics: List[Dict[str, Any]] = Field(default_factory=list, description="Popular discussion topics")
    learning_style_usage: Dict[LearningStyleEnum, int] = Field(default_factory=dict, description="Learning style usage stats")
    user_satisfaction: Optional[float] = Field(None, description="Overall user satisfaction rate")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: bool = Field(default=True, description="Indicates this is an error response")
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatHealthCheck(BaseModel):
    """Model for chat system health check"""
    chat_service_status: str = Field(..., description="Chat service status")
    llm_service_status: str = Field(..., description="LLM service status")
    embedding_service_status: str = Field(..., description="Embedding service status")
    database_status: str = Field(..., description="Database status")
    response_time: float = Field(..., description="Health check response time")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }