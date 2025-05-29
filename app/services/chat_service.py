"""
Chat Service - Adapted from multimodal_rag.py
"""

import logging
import uuid
import time
import base64
import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models.chat import (
    ChatQuery, ChatResponse, RetrievedSource, ImageResult,
    SearchRequest, SearchResults, SourceType, MessageRole
)
from app.models.student import LearningStyleEnum
from app.core.config import get_settings
from app.core.database import get_milvus_collection, get_mongo_collection
from app.services.personalization_service import PersonalizationService

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    """Service for handling chat and RAG functionality"""
    
    def __init__(self):
        self.settings = settings
        self.llm = None
        self.text_embeddings = None
        self.personalization_service = PersonalizationService()
        
    async def initialize(self):
        """Initialize the chat service with models"""
        try:
            # Initialize LLM
            if not self.llm:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=self.settings.GEMINI_API_KEY,
                    temperature=0.2,
                    convert_system_message_to_human=True
                )
            
            # Initialize text embeddings
            if not self.text_embeddings:
                self.text_embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.TEXT_EMBEDDING_MODEL
                )
            
            logger.info("Chat service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chat service: {e}")
            raise
    
    async def search_text_chunks(self, query: str, top_k: int = 5, 
                               collection_name: Optional[str] = None) -> List[RetrievedSource]:
        """Search for relevant text chunks"""
        try:
            if not self.text_embeddings:
                await self.initialize()
            
            collection_name = collection_name or self.settings.MILVUS_TEXT_COLLECTION
            
            # Connect to Milvus
            vectorstore = Milvus(
                embedding_function=self.text_embeddings,
                collection_name=collection_name,
                connection_args={"host": self.settings.MILVUS_HOST, "port": self.settings.MILVUS_PORT}
            )
            
            # Search for relevant documents
            docs = vectorstore.similarity_search_with_score(query, k=top_k)
            
            # Format results
            results = []
            for doc, score in docs:
                source = RetrievedSource(
                    source_id=str(uuid.uuid4()),
                    source_type=SourceType.TEXT,
                    content=doc.page_content,
                    module_code=doc.metadata.get("module_code", "Unknown"),
                    module_name=doc.metadata.get("module_name", "Unknown"),
                    lecture_number=doc.metadata.get("lecture_number"),
                    lecture_title=doc.metadata.get("lecture_title"),
                    lecture_code=doc.metadata.get("lecture_code"),
                    page_number=doc.metadata.get("page", doc.metadata.get("page_number")),
                    similarity_score=float(1 - score),  # Convert distance to similarity
                    metadata=doc.metadata
                )
                results.append(source)
            
            logger.info(f"Found {len(results)} text chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching text chunks: {e}")
            return []
    
    async def search_images(self, query: str, top_k: int = 3,
                          milvus_collection: Optional[str] = None,
                          mongodb_collection: Optional[str] = None) -> List[ImageResult]:
        """Search for relevant images"""
        try:
            milvus_collection = milvus_collection or self.settings.MILVUS_IMAGE_COLLECTION
            mongodb_collection = mongodb_collection or self.settings.MONGODB_IMAGES_COLLECTION
            
            # Import the image search function from your existing code
            from pdf_processor import search_images_by_text
            
            # Use existing image search functionality
            matches = search_images_by_text(
                query=query,
                top_k=top_k,
                milvus_collection=milvus_collection,
                mongodb_collection=mongodb_collection
            )
            
            # Convert to ImageResult models
            results = []
            for match in matches:
                try:
                    # Convert PIL image to base64
                    if match.get("image"):
                        buffer = io.BytesIO()
                        match["image"].save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    else:
                        img_str = ""
                    
                    result = ImageResult(
                        image_id=str(uuid.uuid4()),
                        image_data=img_str,
                        similarity_score=match.get("similarity_score", 0.0),
                        lecture_code=match.get("lecture_code"),
                        lecture_number=match.get("lecture_number"),
                        lecture_title=match.get("lecture_title"),
                        module_id=match.get("module_id"),
                        page_number=match.get("page_number"),
                        text_description=match.get("text", ""),
                        metadata={}
                    )
                    results.append(result)
                    
                except Exception as img_err:
                    logger.error(f"Error processing image result: {img_err}")
                    continue
            
            logger.info(f"Found {len(results)} images for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
    
    async def generate_response(self, query: ChatQuery) -> ChatResponse:
        """Generate a complete chat response with RAG"""
        start_time = time.time()
        response_id = str(uuid.uuid4())
        
        try:
            # Ensure LLM is initialized
            if not self.llm:
                await self.initialize()
            
            # Get or create interaction
            interaction_id = None
            if query.student_id:
                # Start tracking this interaction
                from app.models.student import StudentInteractionCreate
                interaction_data = StudentInteractionCreate(
                    student_id=query.student_id,
                    query=query.message,
                    learning_style_id=query.learning_style,
                    module_id=query.module_id
                )
                interaction_id = await self.personalization_service.start_interaction(interaction_data)
            
            # Determine learning style
            learning_style = query.learning_style or LearningStyleEnum.DETAILED
            if query.student_id and not query.learning_style:
                # Get from student profile
                profile = await self.personalization_service.get_student_profile(query.student_id)
                if profile:
                    learning_style = profile.default_learning_style
            
            # Format query with learning style
            formatted_query = self.personalization_service.format_query_with_learning_style(
                query.message, learning_style
            )
            
            # Search for relevant content
            text_sources = []
            image_results = []
            
            # Search text chunks
            text_sources = await self.search_text_chunks(
                query.message, 
                top_k=5,
                collection_name=getattr(query, 'text_collection', None)
            )
            
            # Search images if requested
            if query.include_images:
                image_results = await self.search_images(
                    query.message,
                    top_k=3
                )
            
            # Build context for LLM
            text_context = self._build_text_context(text_sources)
            image_context = self._build_image_context(image_results)
            
            # Create comprehensive prompt
            prompt = self._create_llm_prompt(
                original_query=query.message,
                formatted_query=formatted_query,
                text_context=text_context,
                image_context=image_context,
                conversation_history=query.conversation_history
            )
            
            # Generate response
            llm_response = self.llm.invoke(prompt)
            answer_text = llm_response.content
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response
            response = ChatResponse(
                response_id=response_id,
                interaction_id=interaction_id,
                answer_text=answer_text,
                learning_style_used=learning_style,
                text_sources=text_sources,
                image_results=image_results,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Update interaction with retrieved sources
            if interaction_id:
                from app.models.student import StudentInteractionUpdate
                sources_data = [
                    {
                        "type": "text",
                        "source_id": source.source_id,
                        "module_code": source.module_code,
                        "lecture_number": source.lecture_number,
                        "similarity_score": source.similarity_score
                    }
                    for source in text_sources
                ] + [
                    {
                        "type": "image", 
                        "image_id": img.image_id,
                        "lecture_code": img.lecture_code,
                        "lecture_number": img.lecture_number,
                        "similarity_score": img.similarity_score
                    }
                    for img in image_results
                ]
                
                update_data = StudentInteractionUpdate(
                    retrieved_sources=sources_data
                )
                await self.personalization_service.end_interaction(interaction_id, update_data)
            
            logger.info(f"Generated response for query in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Return error response
            processing_time = time.time() - start_time
            return ChatResponse(
                response_id=response_id,
                interaction_id=interaction_id,
                answer_text="I apologize, but I encountered an error while processing your question. Please try again or rephrase your query.",
                learning_style_used=learning_style or LearningStyleEnum.DETAILED,
                text_sources=[],
                image_results=[],
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    def _build_text_context(self, text_sources: List[RetrievedSource]) -> str:
        """Build text context from retrieved sources"""
        if not text_sources:
            return "No relevant text information found in the knowledge base."
        
        context_parts = []
        for source in text_sources:
            lecture_info = f"Lecture {source.lecture_number}" if source.lecture_number else "Unknown Lecture"
            context_part = (
                f"Source: {source.module_code}, {lecture_info}, Page {source.page_number}\n"
                f"Content: {source.content}"
            )
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _build_image_context(self, image_results: List[ImageResult]) -> str:
        """Build image context from retrieved images"""
        if not image_results:
            return "No relevant images found in the knowledge base."
        
        context_parts = []
        for i, img in enumerate(image_results):
            lecture_info = f"Lecture {img.lecture_number}" if img.lecture_number else "Unknown Lecture"
            context_part = (
                f"Image {i+1}: From {img.lecture_code}, {lecture_info}, "
                f"Page {img.page_number} - "
                f"Description: {img.text_description[:100]}..."
            )
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_llm_prompt(self, original_query: str, formatted_query: str,
                          text_context: str, image_context: str,
                          conversation_history: List = None) -> str:
        """Create comprehensive prompt for the LLM"""
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 4 messages
            for msg in recent_history:
                if msg.role == MessageRole.USER:
                    conversation_context += f"Previous User: {msg.content}\n"
                elif msg.role == MessageRole.ASSISTANT:
                    conversation_context += f"Previous Assistant: {msg.content[:200]}...\n"
        
        prompt = f"""
        {conversation_context}
        
        Current User Query: {original_query}
        
        Learning Style Instructions: {formatted_query}
        
        Available Text Sources:
        {text_context}
        
        Available Visual References:
        {image_context}
        
        INSTRUCTIONS:
        1. Answer the user's query using the provided context when available.
        2. If the sources contain relevant information, use it and cite specific lecture numbers naturally (e.g., "As explained in Lecture 3...").
        3. If the provided sources don't have sufficient information, use your knowledge to provide a helpful answer, but indicate this clearly (e.g., "While this isn't covered in the provided materials...").
        4. NEVER respond with "I don't have enough information" - always provide the best possible answer.
        5. When referencing images, mention them naturally (e.g., "As shown in Image 1 from Lecture 2...").
        6. Follow the learning style instructions for how to structure and present your response.
        7. Make lecture citations a natural part of your explanation, not just references at the end.
        8. Provide comprehensive, accurate, and helpful responses.
        """
        
        return prompt
    
    async def search(self, request: SearchRequest) -> SearchResults:
        """Perform a search across text and/or images"""
        start_time = time.time()
        
        try:
            text_results = []
            image_results = []
            
            # Search text if requested
            if request.search_type in [SourceType.TEXT, SourceType.MIXED]:
                text_results = await self.search_text_chunks(
                    request.query,
                    top_k=request.top_k
                )
                
                # Apply filters
                if request.module_filter:
                    text_results = [r for r in text_results if r.module_code == request.module_filter]
                if request.lecture_filter:
                    text_results = [r for r in text_results if r.lecture_number == request.lecture_filter]
                if request.similarity_threshold:
                    text_results = [r for r in text_results if r.similarity_score >= request.similarity_threshold]
            
            # Search images if requested
            if request.search_type in [SourceType.IMAGE, SourceType.MIXED]:
                image_results = await self.search_images(
                    request.query,
                    top_k=request.top_k
                )
                
                # Apply filters
                if request.module_filter:
                    image_results = [r for r in image_results if r.module_id == request.module_filter]
                if request.lecture_filter:
                    image_results = [r for r in image_results if r.lecture_number == request.lecture_filter]
                if request.similarity_threshold:
                    image_results = [r for r in image_results if r.similarity_score >= request.similarity_threshold]
            
            # Limit results
            text_results = text_results[:request.top_k]
            image_results = image_results[:request.top_k]
            
            processing_time = time.time() - start_time
            total_results = len(text_results) + len(image_results)
            
            return SearchResults(
                query=request.query,
                total_results=total_results,
                text_results=text_results,
                image_results=image_results,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            processing_time = time.time() - start_time
            
            return SearchResults(
                query=request.query,
                total_results=0,
                text_results=[],
                image_results=[],
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    async def get_chat_history(self, student_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a student"""
        try:
            # Get recent interactions
            interactions_collection = get_mongo_collection(settings.STUDENT_INTERACTIONS_COLLECTION)
            
            cursor = interactions_collection.find(
                {"student_id": student_id}
            ).sort("start_time", -1).limit(limit)
            
            history = []
            for interaction in cursor:
                history.append({
                    "interaction_id": interaction.get("interaction_id"),
                    "query": interaction.get("query"),
                    "timestamp": interaction.get("start_time"),
                    "learning_style": interaction.get("learning_style_id"),
                    "module_id": interaction.get("module_id"),
                    "feedback": interaction.get("feedback"),
                    "helpful": interaction.get("helpful")
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def format_image_info(self, img: ImageResult) -> str:
        """Format image information for display"""
        source_info = f"**Source**: {img.lecture_code or 'Unknown'}"
        lecture_info = f"Lecture {img.lecture_number or 'Unknown'}"
        page_info = f"Page {img.page_number or 'Unknown'}"
        
        return f"{source_info}, {lecture_info}, {page_info}"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the chat service"""
        try:
            status = {
                "chat_service": "healthy",
                "llm_service": "unknown",
                "embedding_service": "unknown",
                "database_status": "unknown",
                "timestamp": datetime.now().isoformat()
            }
            
            # Test LLM
            try:
                if not self.llm:
                    await self.initialize()
                test_response = self.llm.invoke("Test")
                status["llm_service"] = "healthy"
            except Exception as e:
                status["llm_service"] = f"error: {str(e)}"
            
            # Test embeddings
            try:
                if not self.text_embeddings:
                    await self.initialize()
                test_embedding = self.text_embeddings.embed_query("test")
                status["embedding_service"] = "healthy"
            except Exception as e:
                status["embedding_service"] = f"error: {str(e)}"
            
            # Test database
            try:
                from app.core.database import check_database_health
                db_health = await check_database_health()
                status["database_status"] = "healthy" if db_health["overall_healthy"] else "unhealthy"
            except Exception as e:
                status["database_status"] = f"error: {str(e)}"
            
            return status
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "chat_service": f"error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }