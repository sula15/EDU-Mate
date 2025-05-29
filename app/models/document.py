"""
Pydantic models for document processing and management
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Enum for document types"""
    PDF = "pdf"
    PPTX = "pptx"
    DOCX = "docx"
    TXT = "txt"


class SourceType(str, Enum):
    """Enum for document source types"""
    LECTURE = "lecture"
    LAB = "lab"
    TUTORIAL = "tutorial"
    INTERNET = "internet"
    TEXTBOOK = "textbook"
    OTHER = "other"


class ProcessingType(str, Enum):
    """Enum for processing types"""
    TEXT_ONLY = "text_only"
    IMAGES_ONLY = "images_only"
    TEXT_AND_IMAGES = "text_and_images"


class ProcessingStatus(str, Enum):
    """Enum for processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentMetadata(BaseModel):
    """Model for document metadata"""
    module_id: str = Field(..., min_length=1, description="Module identifier (e.g., MDPCC, IOT)")
    lecture_code: Optional[str] = Field(None, description="Lecture code (e.g., IT3061)")
    lecture_number: Optional[int] = Field(None, ge=0, description="Lecture number in sequence")
    lecture_title: Optional[str] = Field(None, description="Title of the lecture")
    source_type: SourceType = Field(default=SourceType.LECTURE, description="Type of document source")
    academic_year: Optional[str] = Field(None, description="Academic year (e.g., 2024-2025)")
    semester: Optional[str] = Field(None, description="Semester information")
    instructor: Optional[str] = Field(None, description="Instructor name")
    course_name: Optional[str] = Field(None, description="Full course name")
    tags: List[str] = Field(default_factory=list, description="Document tags for categorization")
    description: Optional[str] = Field(None, description="Document description")


class DocumentUpload(BaseModel):
    """Model for document upload request"""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., gt=0, description="File size in bytes")
    file_type: DocumentType = Field(..., description="Document type")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    processing_type: ProcessingType = Field(default=ProcessingType.TEXT_AND_IMAGES, description="Type of processing")
    
    # Processing configuration
    similarity_threshold: float = Field(default=0.98, ge=0.0, le=1.0, description="Similarity threshold for duplicate detection")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Text chunk overlap")
    image_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for image embeddings")
    batch_size: int = Field(default=8, ge=1, le=32, description="Processing batch size")


class ProcessingConfig(BaseModel):
    """Model for processing configuration"""
    # Text processing
    text_collection: str = Field(default="combined_text_collection", description="Milvus text collection name")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Text chunk overlap")
    
    # Image processing
    mongodb_collection: str = Field(default="pdf_images_7", description="MongoDB image collection name")
    milvus_image_collection: str = Field(default="combined_embeddings_7", description="Milvus image collection name")
    similarity_threshold: float = Field(default=0.98, ge=0.0, le=1.0, description="Similarity threshold")
    image_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Image embedding weight")
    text_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Text embedding weight")
    
    # Processing parameters
    batch_size: int = Field(default=8, ge=1, le=32, description="Processing batch size")
    use_dim_reduction: bool = Field(default=True, description="Use dimensionality reduction")
    output_dim: int = Field(default=512, description="Output embedding dimension")
    use_embedding_alignment: bool = Field(default=True, description="Use embedding alignment")


class DocumentInfo(BaseModel):
    """Model for document information"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="File hash for duplicate detection")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    processing_type: ProcessingType = Field(..., description="Type of processing applied")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    uploaded_by: Optional[str] = Field(None, description="Uploader identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingResult(BaseModel):
    """Model for processing results"""
    document_id: str = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Processing status")
    
    # Text processing results
    text_chunks_created: Optional[int] = Field(None, description="Number of text chunks created")
    text_collection_used: Optional[str] = Field(None, description="Text collection used")
    
    # Image processing results
    total_images_found: Optional[int] = Field(None, description="Total images found")
    images_filtered: Optional[int] = Field(None, description="Images filtered as duplicates")
    unique_images_processed: Optional[int] = Field(None, description="Unique images processed")
    images_stored: Optional[int] = Field(None, description="Images successfully stored")
    
    # Collections used
    mongodb_collection: Optional[str] = Field(None, description="MongoDB collection used")
    milvus_collections: List[str] = Field(default_factory=list, description="Milvus collections used")
    
    # Processing metrics
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    started_at: datetime = Field(..., description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentSearchQuery(BaseModel):
    """Model for document search queries"""
    query: Optional[str] = Field(None, description="Search query text")
    module_id: Optional[str] = Field(None, description="Filter by module ID")
    lecture_code: Optional[str] = Field(None, description="Filter by lecture code")
    lecture_number: Optional[int] = Field(None, description="Filter by lecture number")
    source_type: Optional[SourceType] = Field(None, description="Filter by source type")
    file_type: Optional[DocumentType] = Field(None, description="Filter by file type")
    tags: List[str] = Field(default_factory=list, description="Filter by tags")
    date_from: Optional[datetime] = Field(None, description="Filter by upload date (from)")
    date_to: Optional[datetime] = Field(None, description="Filter by upload date (to)")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Results offset for pagination")


class DocumentSearchResult(BaseModel):
    """Model for document search results"""
    total_count: int = Field(..., description="Total number of matching documents")
    documents: List[DocumentInfo] = Field(..., description="List of matching documents")
    query: DocumentSearchQuery = Field(..., description="Original search query")
    search_time: float = Field(..., description="Search execution time in seconds")


class DocumentStats(BaseModel):
    """Model for document statistics"""
    total_documents: int = Field(..., description="Total number of documents")
    documents_by_type: Dict[DocumentType, int] = Field(..., description="Documents grouped by type")
    documents_by_module: Dict[str, int] = Field(..., description="Documents grouped by module")
    documents_by_source: Dict[SourceType, int] = Field(..., description="Documents grouped by source type")
    total_storage_size: int = Field(..., description="Total storage size in bytes")
    processing_success_rate: float = Field(..., description="Processing success rate (0-1)")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    last_updated: datetime = Field(..., description="Statistics last updated timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CollectionInfo(BaseModel):
    """Model for collection information"""
    collection_name: str = Field(..., description="Collection name")
    collection_type: str = Field(..., description="Collection type (mongodb/milvus)")
    document_count: int = Field(..., description="Number of documents in collection")
    size_mb: Optional[float] = Field(None, description="Collection size in MB")
    schema_info: Optional[Dict[str, Any]] = Field(None, description="Collection schema information")
    last_updated: datetime = Field(..., description="Last updated timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatus(BaseModel):
    """Model for system status"""
    mongodb_status: str = Field(..., description="MongoDB connection status")
    milvus_status: str = Field(..., description="Milvus connection status")
    available_collections: List[CollectionInfo] = Field(..., description="Available collections")
    processing_queue_size: int = Field(default=0, description="Number of documents in processing queue")
    system_health: str = Field(..., description="Overall system health")
    last_check: datetime = Field(..., description="Last health check timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentResponse(BaseModel):
    """Standard response model for document operations"""
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    document_id: Optional[str] = Field(None, description="Document identifier")
    document_info: Optional[DocumentInfo] = Field(None, description="Document information")
    processing_result: Optional[ProcessingResult] = Field(None, description="Processing results")


class BulkUploadRequest(BaseModel):
    """Model for bulk document upload requests"""
    documents: List[DocumentUpload] = Field(..., description="List of documents to upload")
    processing_config: Optional[ProcessingConfig] = Field(None, description="Processing configuration")
    priority: int = Field(default=1, ge=1, le=5, description="Processing priority (1=highest, 5=lowest)")


class BulkUploadResponse(BaseModel):
    """Model for bulk upload response"""
    total_requested: int = Field(..., description="Total documents requested for upload")
    successfully_queued: int = Field(..., description="Documents successfully queued for processing")
    failed_uploads: List[Dict[str, str]] = Field(default_factory=list, description="Failed uploads with reasons")
    batch_id: str = Field(..., description="Batch identifier for tracking")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }