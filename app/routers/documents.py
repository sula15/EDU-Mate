"""
Document upload and processing endpoints
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import json

from app.models.document import (
    DocumentUpload, DocumentInfo, ProcessingResult, DocumentSearchQuery,
    DocumentSearchResult, DocumentStats, DocumentResponse, DocumentType,
    ProcessingType, DocumentMetadata, SourceType, BulkUploadRequest,
    BulkUploadResponse, SystemStatus
)
from app.services.document_service import DocumentService
from app.routers.auth import get_current_student
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()
document_service = DocumentService()


@router.on_event("startup")
async def startup_event():
    """Initialize document service on startup"""
    try:
        await document_service.initialize()
        logger.info("Document service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document service: {e}")


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(...),
    processing_type: ProcessingType = Form(ProcessingType.TEXT_AND_IMAGES),
    similarity_threshold: float = Form(0.98),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    image_weight: float = Form(0.3),
    batch_size: int = Form(8),
    current_student: str = Depends(get_current_student)
):
    """Upload and process a document"""
    try:
        logger.info(f"Document upload request from student {current_student}")
        
        # Validate file size
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Validate file type
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        file_extension = file.filename.split('.')[-1].lower()
        if f".{file_extension}" not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
            document_metadata = DocumentMetadata(**metadata_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata format: {e}"
            )
        
        # Determine document type
        doc_type_map = {
            "pdf": DocumentType.PDF,
            "pptx": DocumentType.PPTX,
            "docx": DocumentType.DOCX,
            "txt": DocumentType.TXT
        }
        document_type = doc_type_map.get(file_extension, DocumentType.PDF)
        
        # Create upload request
        upload_request = DocumentUpload(
            filename=file.filename,
            file_size=file_size,
            file_type=document_type,
            metadata=document_metadata,
            processing_type=processing_type,
            similarity_threshold=similarity_threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            image_weight=image_weight,
            batch_size=batch_size
        )
        
        logger.info(f"Processing document: {file.filename} ({file_size} bytes)")
        
        # Process document
        result = await document_service.upload_and_process_document(file_content, upload_request)
        
        # Add background task for cleanup
        background_tasks.add_task(cleanup_temp_files, result.document_id)
        
        logger.info(f"Document processing completed: {result.status}")
        
        return DocumentResponse(
            success=result.status.value == "completed",
            message=f"Document processed: {result.status.value}",
            document_id=result.document_id,
            processing_result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get("/search", response_model=DocumentSearchResult)
async def search_documents(
    query: Optional[str] = None,
    module_id: Optional[str] = None,
    lecture_code: Optional[str] = None,
    lecture_number: Optional[int] = None,
    source_type: Optional[SourceType] = None,
    file_type: Optional[DocumentType] = None,
    limit: int = 20,
    offset: int = 0,
    current_student: str = Depends(get_current_student)
):
    """Search for documents"""
    try:
        logger.info(f"Document search request from student {current_student}")
        
        # Create search query
        search_query = DocumentSearchQuery(
            query=query,
            module_id=module_id,
            lecture_code=lecture_code,
            lecture_number=lecture_number,
            source_type=source_type,
            file_type=file_type,
            limit=limit,
            offset=offset
        )
        
        # Perform search
        results = await document_service.search_documents(search_query)
        
        logger.info(f"Document search completed: {results.total_count} results")
        return results
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search documents"
        )


@router.get("/stats", response_model=DocumentStats)
async def get_document_stats(current_student: str = Depends(get_current_student)):
    """Get document statistics"""
    try:
        logger.info(f"Document stats request from student {current_student}")
        
        stats = await document_service.get_document_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document statistics"
        )


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document_info(
    document_id: str,
    current_student: str = Depends(get_current_student)
):
    """Get information about a specific document"""
    try:
        logger.info(f"Document info request: {document_id}")
        
        document_info = await document_service.get_document_info(document_id)
        
        if not document_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return document_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document information"
        )


@router.get("/{document_id}/processing-result", response_model=ProcessingResult)
async def get_processing_result(
    document_id: str,
    current_student: str = Depends(get_current_student)
):
    """Get processing result for a specific document"""
    try:
        logger.info(f"Processing result request: {document_id}")
        
        result = await document_service.get_processing_result(document_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Processing result not found"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing result"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_student: str = Depends(get_current_student)
):
    """Delete a document and its associated data"""
    try:
        logger.info(f"Document deletion request: {document_id}")
        
        success = await document_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or could not be deleted"
            )
        
        return {
            "success": True,
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.post("/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    metadata_list: str = Form(...),
    processing_type: ProcessingType = Form(ProcessingType.TEXT_AND_IMAGES),
    current_student: str = Depends(get_current_student)
):
    """Upload multiple documents for batch processing"""
    try:
        logger.info(f"Bulk upload request from student {current_student}: {len(files)} files")
        
        # Parse metadata list
        try:
            metadata_dicts = json.loads(metadata_list)
            if len(metadata_dicts) != len(files):
                raise ValueError("Metadata count must match file count")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata format: {e}"
            )
        
        # Process each file
        successful_uploads = 0
        failed_uploads = []
        batch_id = f"batch_{int(datetime.now().timestamp())}"
        
        for i, (file, metadata_dict) in enumerate(zip(files, metadata_dicts)):
            try:
                # Validate file
                file_content = await file.read()
                file_size = len(file_content)
                
                if file_size > settings.MAX_FILE_SIZE:
                    failed_uploads.append({
                        "filename": file.filename,
                        "reason": "File too large"
                    })
                    continue
                
                # Create metadata
                document_metadata = DocumentMetadata(**metadata_dict)
                
                # Determine document type
                file_extension = file.filename.split('.')[-1].lower()
                doc_type_map = {
                    "pdf": DocumentType.PDF,
                    "pptx": DocumentType.PPTX,
                    "docx": DocumentType.DOCX,
                    "txt": DocumentType.TXT
                }
                document_type = doc_type_map.get(file_extension, DocumentType.PDF)
                
                # Create upload request
                upload_request = DocumentUpload(
                    filename=file.filename,
                    file_size=file_size,
                    file_type=document_type,
                    metadata=document_metadata,
                    processing_type=processing_type
                )
                
                # Add to background processing
                background_tasks.add_task(
                    process_document_background,
                    file_content,
                    upload_request,
                    batch_id
                )
                
                successful_uploads += 1
                
            except Exception as e:
                failed_uploads.append({
                    "filename": file.filename,
                    "reason": str(e)
                })
        
        logger.info(f"Bulk upload queued: {successful_uploads} successful, {len(failed_uploads)} failed")
        
        return BulkUploadResponse(
            total_requested=len(files),
            successfully_queued=successful_uploads,
            failed_uploads=failed_uploads,
            batch_id=batch_id,
            estimated_completion_time=datetime.now() + timedelta(minutes=successful_uploads * 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process bulk upload"
        )


@router.get("/system/status", response_model=SystemStatus)
async def get_system_status(current_student: str = Depends(get_current_student)):
    """Get system status for document processing"""
    try:
        from app.core.database import check_database_health
        
        # Check database health
        db_health = await check_database_health()
        
        # Get collection info (simplified)
        collections = []
        
        # Mock collection info - in production, get real collection stats
        collections.extend([
            {
                "collection_name": settings.MONGODB_IMAGES_COLLECTION,
                "collection_type": "mongodb",
                "document_count": 1000,  # Would get real count
                "size_mb": 150.5,
                "last_updated": datetime.now()
            },
            {
                "collection_name": settings.MILVUS_TEXT_COLLECTION,
                "collection_type": "milvus",
                "document_count": 5000,  # Would get real count
                "size_mb": 45.2,
                "last_updated": datetime.now()
            }
        ])
        
        return SystemStatus(
            mongodb_status="healthy" if db_health["mongodb"]["connected"] else "unhealthy",
            milvus_status="healthy" if db_health["milvus"]["connected"] else "unhealthy",
            available_collections=collections,
            processing_queue_size=0,  # Would get real queue size
            system_health="healthy" if db_health["overall_healthy"] else "degraded",
            last_check=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )


@router.get("/health")
async def document_health_check():
    """Health check for document service"""
    try:
        from app.core.database import check_database_health
        
        db_health = await check_database_health()
        
        return {
            "status": "healthy" if db_health["overall_healthy"] else "degraded",
            "service": "document_processing",
            "mongodb": "healthy" if db_health["mongodb"]["connected"] else "unhealthy",
            "milvus": "healthy" if db_health["milvus"]["connected"] else "unhealthy",
            "upload_dir": settings.UPLOAD_DIR,
            "max_file_size": settings.MAX_FILE_SIZE,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "document_processing",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Background task functions
async def process_document_background(
    file_content: bytes,
    upload_request: DocumentUpload,
    batch_id: str
):
    """Background task to process document"""
    try:
        logger.info(f"Background processing document: {upload_request.filename} (batch: {batch_id})")
        
        result = await document_service.upload_and_process_document(file_content, upload_request)
        
        logger.info(f"Background processing completed: {upload_request.filename} - {result.status}")
        
    except Exception as e:
        logger.error(f"Error in background document processing: {e}")


async def cleanup_temp_files(document_id: str):
    """Background task to cleanup temporary files"""
    try:
        # In a production system, this would clean up any temporary files
        # created during document processing
        logger.info(f"Cleanup completed for document: {document_id}")
        
    except Exception as e:
        logger.error(f"Error in cleanup: {e}")


# Utility endpoints
@router.get("/supported-types")
async def get_supported_document_types():
    """Get supported document types and their configurations"""
    return {
        "supported_types": [
            {
                "type": "PDF",
                "extensions": [".pdf"],
                "description": "Portable Document Format",
                "supports_text": True,
                "supports_images": True
            },
            {
                "type": "PPTX",
                "extensions": [".pptx"],
                "description": "PowerPoint Presentation",
                "supports_text": True,
                "supports_images": True,
                "note": "Converted to PDF before processing"
            },
            {
                "type": "DOCX",
                "extensions": [".docx"],
                "description": "Word Document",
                "supports_text": True,
                "supports_images": False,
                "note": "Limited support"
            },
            {
                "type": "TXT",
                "extensions": [".txt"],
                "description": "Plain Text",
                "supports_text": True,
                "supports_images": False
            }
        ],
        "max_file_size": settings.MAX_FILE_SIZE,
        "processing_types": [
            "text_only",
            "images_only",
            "text_and_images"
        ]
    }