"""
Core configuration settings for the FastAPI application
"""

import os
import warnings
from typing import Optional, List
from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MindMate-Edu Backend"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:8080",  # Alternative frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    # Database Configuration
    MONGODB_URI: str = "mongodb://localhost:27017/"
    MONGODB_DB: str = "student_analytics"
    
    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    
    # Collection Names
    MONGODB_IMAGES_COLLECTION: str = "pdf_images_7"
    MILVUS_TEXT_COLLECTION: str = "combined_text_collection"
    MILVUS_IMAGE_COLLECTION: str = "combined_embeddings_7"
    STUDENT_PROFILES_COLLECTION: str = "student_profiles"
    STUDENT_INTERACTIONS_COLLECTION: str = "student_interactions"
    STUDENT_ANALYTICS_COLLECTION: str = "student_analytics"
    
    # AI Model Configuration
    GEMINI_API_KEY: Optional[str] = None
    ANXIETY_MODEL_PATH: str = "best_multimodal_model.pth"
    WHISPER_MODEL_SIZE: str = "base"
    
    # Embedding Configuration
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    
    # File Upload Configuration
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".pptx", ".wav", ".mp3", ".m4a"]
    
    # Session Configuration
    SESSION_TIMEOUT: int = 1800  # 30 minutes
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Processing Configuration
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.98
    DEFAULT_IMAGE_WEIGHT: float = 0.3
    DEFAULT_TEXT_WEIGHT: float = 0.7
    DEFAULT_OUTPUT_DIM: int = 512
    
    # Learning Styles Configuration
    DEFAULT_LEARNING_STYLE: str = "detailed"
    
    # Redis Configuration (optional for background tasks)
    REDIS_URL: Optional[str] = None
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_gemini_key(cls, v):
        if not v:
            # For testing purposes, make this a warning instead of an error
            warnings.warn("GEMINI_API_KEY is not set. LLM functionality will not work.")
        return v
    
    def create_upload_dir(self):
        """Create upload directory if it doesn't exist"""
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        return self.UPLOAD_DIR
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }


class LearningStyleConfig:
    """Configuration for learning styles"""
    
    LEARNING_STYLES = [
        {
            "id": "detailed",
            "name": "Detailed Explanation",
            "description": "Comprehensive, thorough explanations with examples and deep context",
            "prompt_template": (
                "Please provide a detailed and comprehensive explanation with examples. "
                "Include background context, theory, and practical examples. "
                "Structure your answer with clear sections and thorough explanations."
            )
        },
        {
            "id": "concise",
            "name": "Concise Summary", 
            "description": "Brief, to-the-point explanations focusing on key concepts",
            "prompt_template": (
                "Please provide a concise, to-the-point explanation. "
                "Focus only on the most important concepts and key takeaways. "
                "Keep your answer brief and direct without unnecessary details."
            )
        },
        {
            "id": "bulleted",
            "name": "Bulleted List",
            "description": "Information organized in easy-to-scan bullet points",
            "prompt_template": (
                "Please format your response as a bulleted list. "
                "Organize information in clear, scannable bullet points with hierarchical structure. "
                "Use headings where appropriate and keep each bullet point focused on a single concept."
            )
        },
        {
            "id": "eli5",
            "name": "Explain Like I'm 5",
            "description": "Simple explanations using basic language and analogies",
            "prompt_template": (
                "Please explain this concept as if I'm a beginner with no background knowledge. "
                "Use simple language, analogies, and everyday examples I can relate to. "
                "Avoid technical jargon and complex terminology unless absolutely necessary and defined."
            )
        },
        {
            "id": "visual",
            "name": "Visual Learning",
            "description": "Focus on diagrams, charts, and visual explanations",
            "prompt_template": (
                "Please emphasize visual examples in your explanation. "
                "Refer to any diagrams, charts, or images that help illustrate the concepts. "
                "Describe visual relationships and spatial information clearly, and suggest visual ways to remember the information."
            )
        },
        {
            "id": "quiz",
            "name": "Quiz-Based",
            "description": "Information presented through practice questions",
            "prompt_template": (
                "Please structure your response as a series of practice questions with answers. "
                "First, provide a brief overview of the key concepts, then present 3-5 questions that test understanding. "
                "Include the answers with explanations at the end."
            )
        }
    ]
    
    @classmethod
    def get_learning_style_by_id(cls, style_id: str) -> Optional[dict]:
        """Get learning style configuration by ID"""
        for style in cls.LEARNING_STYLES:
            if style["id"] == style_id:
                return style
        return None
    
    @classmethod
    def get_prompt_template(cls, style_id: str) -> str:
        """Get prompt template for a learning style"""
        style = cls.get_learning_style_by_id(style_id)
        if style:
            return style["prompt_template"]
        return cls.get_learning_style_by_id("detailed")["prompt_template"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export commonly used settings
settings = get_settings()
learning_styles = LearningStyleConfig()