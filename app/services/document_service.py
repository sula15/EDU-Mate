"""
Document Processing Service - Adapted from pdf_processor.py and semantic_chunker.py
"""

import os
import tempfile
import hashlib
import logging
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

from app.models.document import (
    DocumentUpload, DocumentInfo, ProcessingResult, ProcessingConfig,
    DocumentSearchQuery, DocumentSearchResult, DocumentStats,
    DocumentType, ProcessingStatus, ProcessingType
)
from app.core.config import get_settings
from app.core.database import get_mongo_collection, get_milvus_collection

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentService:
    """Service for document processing and management"""
    
    def __init__(self):
        self.settings = settings
        self.text_embeddings = None
        
    async def initialize(self):
        """Initialize the document service"""
        try:
            if not self.text_embeddings:
                self.text_embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.TEXT_EMBEDDING_MODEL
                )
            logger.info("Document service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing document service: {e}")
            raise
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate MD5 hash of file content for duplicate detection"""
        return hashlib.md5(file_content).hexdigest()
    
    async def is_duplicate_file(self, file_hash: str, metadata: Dict[str, Any]) -> bool:
        """Check if file is a duplicate"""
        try:
            # Check in document info collection
            docs_collection = get_mongo_collection("document_info")
            
            query = {
                "file_hash": file_hash,
                "metadata.module_id": metadata.get("module_id", ""),
                "metadata.lecture_code": metadata.get("lecture_code", ""),
                "metadata.lecture_number": metadata.get("lecture_number", 0)
            }
            
            existing_doc = docs_collection.find_one(query)
            return existing_doc is not None
            
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            return False
    
    async def convert_pptx_to_pdf(self, input_path: str) -> str:
        """Convert PPTX file to PDF"""
        try:
            import subprocess
            import sys
            
            # Create output path
            output_dir = os.path.dirname(input_path)
            output_path = os.path.join(output_dir, 
                                     os.path.splitext(os.path.basename(input_path))[0] + ".pdf")
            
            # Determine LibreOffice command
            libreoffice_cmd = "soffice"
            if sys.platform == "darwin":  # macOS
                if os.path.exists("/Applications/LibreOffice.app/Contents/MacOS/soffice"):
                    libreoffice_cmd = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            
            # Convert using LibreOffice
            cmd = [
                libreoffice_cmd,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", output_dir,
                input_path
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise RuntimeError(f"LibreOffice conversion failed: {process.stderr}")
            
            if not os.path.exists(output_path):
                raise RuntimeError("PDF file was not created")
            
            logger.info(f"Successfully converted PPTX to PDF: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting PPTX to PDF: {e}")
            raise
    
    async def process_document_for_text(self, file_path: str, metadata: Dict[str, Any],
                                      config: ProcessingConfig) -> Dict[str, Any]:
        """Process document for text extraction and indexing"""
        try:
            if not self.text_embeddings:
                await self.initialize()
            
            # Load document
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata to each document
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata.update({
                    "page_number": doc.metadata.get("page", 0),
                    "source_type": metadata.get("source_type", ""),
                    "file_hash": metadata.get("file_hash", "")
                })
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"Split document into {len(chunks)} text chunks")
            
            # Create embeddings and store in Milvus
            vectorstore = Milvus.from_documents(
                documents=chunks,
                embedding=self.text_embeddings,
                collection_name=config.text_collection,
                connection_args={"host": self.settings.MILVUS_HOST, "port": self.settings.MILVUS_PORT},
                drop_old=False
            )
            
            logger.info(f"Stored text chunks in Milvus collection: {config.text_collection}")
            
            return {
                "success": True,
                "text_chunks_created": len(chunks),
                "text_collection_used": config.text_collection
            }
            
        except Exception as e:
            logger.error(f"Error processing document for text: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_document_for_images(self, file_path: str, metadata: Dict[str, Any],
                                        config: ProcessingConfig) -> Dict[str, Any]:
        """Process document for image extraction and indexing"""
        try:
            # Import from your existing pdf_processor
            from pdf_processor import EmbeddingConfig, create_embeddings_and_store
            
            # Convert ProcessingConfig to EmbeddingConfig
            embedding_config = EmbeddingConfig(
                image_weight=config.image_weight,
                text_weight=config.text_weight,
                similarity_threshold=config.similarity_threshold,
                batch_size=config.batch_size,
                use_dim_reduction=config.use_dim_reduction,
                output_dim=config.output_dim,
                use_embedding_alignment=config.use_embedding_alignment,
                mongodb_uri=self.settings.MONGODB_URI,
                mongodb_db=self.settings.MONGODB_DB,
                mongodb_collection=config.mongodb_collection,
                milvus_host=self.settings.MILVUS_HOST,
                milvus_port=self.settings.MILVUS_PORT,
                milvus_collection=config.milvus_image_collection
            )
            
            # Process images using existing functionality
            result = create_embeddings_and_store(
                pdf_path=file_path,
                metadata=metadata,
                config=embedding_config
            )
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"]
                }
            
            return {
                "success": True,
                "total_images_found": result.get("num_original_images", 0),
                "images_filtered": result.get("num_filtered", 0),
                "unique_images_processed": result.get("num_unique", 0),
                "images_stored": result.get("num_inserted_milvus", 0),
                "mongodb_collection": result.get("mongodb_collection"),
                "milvus_collection": result.get("milvus_collection")
            }
            
        except Exception as e:
            logger.error(f"Error processing document for images: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def upload_and_process_document(self, file_content: bytes, 
                                        upload_request: DocumentUpload) -> ProcessingResult:
        """Upload and process a document"""
        document_id = str(uuid.uuid4())
        started_at = datetime.now()
        
        try:
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_content)
            
            # Check for duplicates
            if await self.is_duplicate_file(file_hash, upload_request.metadata.model_dump()):
                return ProcessingResult(
                    document_id=document_id,
                    status=ProcessingStatus.FAILED,
                    error_message="Duplicate file detected",
                    started_at=started_at
                )
            
            # Create temporary file
            file_extension = f".{upload_request.file_type.value}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Convert PPTX to PDF if needed
                processing_file_path = temp_file_path
                if upload_request.file_type == DocumentType.PPTX:
                    processing_file_path = await self.convert_pptx_to_pdf(temp_file_path)
                
                # Create document info
                doc_info = DocumentInfo(
                    document_id=document_id,
                    filename=upload_request.filename,
                    file_type=upload_request.file_type,
                    file_size=upload_request.file_size,
                    file_hash=file_hash,
                    metadata=upload_request.metadata,
                    processing_type=upload_request.processing_type,
                    uploaded_at=started_at,
                    uploaded_by="system"  # Could be set from request context
                )
                
                # Store document info
                await self.store_document_info(doc_info)
                
                # Create processing config
                config = ProcessingConfig(
                    similarity_threshold=upload_request.similarity_threshold,
                    chunk_size=upload_request.chunk_size,
                    chunk_overlap=upload_request.chunk_overlap,
                    image_weight=upload_request.image_weight,
                    text_weight=1.0 - upload_request.image_weight,
                    batch_size=upload_request.batch_size
                )
                
                # Process based on type
                text_result = {}
                image_result = {}
                
                if upload_request.processing_type in [ProcessingType.TEXT_ONLY, ProcessingType.TEXT_AND_IMAGES]:
                    text_result = await self.process_document_for_text(
                        processing_file_path,
                        {**upload_request.metadata.model_dump(), "file_hash": file_hash},
                        config
                    )
                
                if upload_request.processing_type in [ProcessingType.IMAGES_ONLY, ProcessingType.TEXT_AND_IMAGES]:
                    image_result = await self.process_document_for_images(
                        processing_file_path,
                        {**upload_request.metadata.model_dump(), "file_hash": file_hash},
                        config
                    )
                
                # Determine final status
                text_success = text_result.get("success", True)
                image_success = image_result.get("success", True)
                
                if text_success and image_success:
                    status = ProcessingStatus.COMPLETED
                    error_message = None
                elif text_success or image_success:
                    status = ProcessingStatus.COMPLETED
                    error_message = "Partial processing completed"
                else:
                    status = ProcessingStatus.FAILED
                    error_message = f"Text error: {text_result.get('error', 'Unknown')}, Image error: {image_result.get('error', 'Unknown')}"
                
                # Create result
                result = ProcessingResult(
                    document_id=document_id,
                    status=status,
                    text_chunks_created=text_result.get("text_chunks_created"),
                    text_collection_used=text_result.get("text_collection_used"),
                    total_images_found=image_result.get("total_images_found"),
                    images_filtered=image_result.get("images_filtered"),
                    unique_images_processed=image_result.get("unique_images_processed"),
                    images_stored=image_result.get("images_stored"),
                    mongodb_collection=image_result.get("mongodb_collection"),
                    milvus_collections=[
                        text_result.get("text_collection_used", ""),
                        image_result.get("milvus_collection", "")
                    ],
                    processing_time=(datetime.now() - started_at).total_seconds(),
                    error_message=error_message,
                    started_at=started_at,
                    completed_at=datetime.now()
                )
                
                # Store processing result
                await self.store_processing_result(result)
                
                logger.info(f"Document processing completed: {document_id}")
                return result
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_file_path)
                    if processing_file_path != temp_file_path:
                        os.unlink(processing_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            
            return ProcessingResult(
                document_id=document_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.now()
            )
    
    async def store_document_info(self, doc_info: DocumentInfo) -> bool:
        """Store document information in database"""
        try:
            docs_collection = get_mongo_collection("document_info")
            doc_data = doc_info.model_dump()
            doc_data["uploaded_at"] = doc_info.uploaded_at.isoformat()
            
            docs_collection.insert_one(doc_data)
            return True
            
        except Exception as e:
            logger.error(f"Error storing document info: {e}")
            return False
    
    async def store_processing_result(self, result: ProcessingResult) -> bool:
        """Store processing result in database"""
        try:
            results_collection = get_mongo_collection("processing_results")
            result_data = result.model_dump()
            result_data["started_at"] = result.started_at.isoformat()
            if result.completed_at:
                result_data["completed_at"] = result.completed_at.isoformat()
            
            results_collection.insert_one(result_data)
            return True
            
        except Exception as e:
            logger.error(f"Error storing processing result: {e}")
            return False
    
    async def get_document_info(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document information by ID"""
        try:
            docs_collection = get_mongo_collection("document_info")
            doc_data = docs_collection.find_one({"document_id": document_id})
            
            if doc_data:
                doc_data["uploaded_at"] = datetime.fromisoformat(doc_data["uploaded_at"])
                return DocumentInfo.model_validate(doc_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return None
    
    async def get_processing_result(self, document_id: str) -> Optional[ProcessingResult]:
        """Get processing result by document ID"""
        try:
            results_collection = get_mongo_collection("processing_results")
            result_data = results_collection.find_one({"document_id": document_id})
            
            if result_data:
                result_data["started_at"] = datetime.fromisoformat(result_data["started_at"])
                if result_data.get("completed_at"):
                    result_data["completed_at"] = datetime.fromisoformat(result_data["completed_at"])
                return ProcessingResult.model_validate(result_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting processing result: {e}")
            return None
    
    async def search_documents(self, query: DocumentSearchQuery) -> DocumentSearchResult:
        """Search for documents based on criteria"""
        try:
            docs_collection = get_mongo_collection("document_info")
            
            # Build search filter
            search_filter = {}
            
            if query.module_id:
                search_filter["metadata.module_id"] = query.module_id
            if query.lecture_code:
                search_filter["metadata.lecture_code"] = query.lecture_code
            if query.lecture_number is not None:
                search_filter["metadata.lecture_number"] = query.lecture_number
            if query.source_type:
                search_filter["metadata.source_type"] = query.source_type
            if query.file_type:
                search_filter["file_type"] = query.file_type
            if query.tags:
                search_filter["metadata.tags"] = {"$in": query.tags}
            
            # Date range filter
            if query.date_from or query.date_to:
                date_filter = {}
                if query.date_from:
                    date_filter["$gte"] = query.date_from.isoformat()
                if query.date_to:
                    date_filter["$lte"] = query.date_to.isoformat()
                search_filter["uploaded_at"] = date_filter
            
            # Text search
            if query.query:
                search_filter["$or"] = [
                    {"filename": {"$regex": query.query, "$options": "i"}},
                    {"metadata.lecture_title": {"$regex": query.query, "$options": "i"}},
                    {"metadata.course_name": {"$regex": query.query, "$options": "i"}}
                ]
            
            # Count total results
            total_count = docs_collection.count_documents(search_filter)
            
            # Get paginated results
            cursor = docs_collection.find(search_filter).skip(query.offset).limit(query.limit)
            
            documents = []
            for doc_data in cursor:
                try:
                    doc_data["uploaded_at"] = datetime.fromisoformat(doc_data["uploaded_at"])
                    documents.append(DocumentInfo.model_validate(doc_data))
                except Exception as e:
                    logger.error(f"Error parsing document: {e}")
                    continue
            
            return DocumentSearchResult(
                total_count=total_count,
                documents=documents,
                query=query,
                search_time=0.0  # Could be measured
            )
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return DocumentSearchResult(
                total_count=0,
                documents=[],
                query=query,
                search_time=0.0
            )
    
    async def get_document_stats(self) -> DocumentStats:
        """Get document statistics"""
        try:
            docs_collection = get_mongo_collection("document_info")
            results_collection = get_mongo_collection("processing_results")
            
            # Total documents
            total_documents = docs_collection.count_documents({})
            
            # Documents by type
            type_pipeline = [
                {"$group": {"_id": "$file_type", "count": {"$sum": 1}}}
            ]
            type_results = list(docs_collection.aggregate(type_pipeline))
            documents_by_type = {item["_id"]: item["count"] for item in type_results}
            
            # Documents by module
            module_pipeline = [
                {"$group": {"_id": "$metadata.module_id", "count": {"$sum": 1}}}
            ]
            module_results = list(docs_collection.aggregate(module_pipeline))
            documents_by_module = {item["_id"]: item["count"] for item in module_results}
            
            # Documents by source
            source_pipeline = [
                {"$group": {"_id": "$metadata.source_type", "count": {"$sum": 1}}}
            ]
            source_results = list(docs_collection.aggregate(source_pipeline))
            documents_by_source = {item["_id"]: item["count"] for item in source_results}
            
            # Storage size (sum of file sizes)
            size_pipeline = [
                {"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}
            ]
            size_results = list(docs_collection.aggregate(size_pipeline))
            total_storage_size = size_results[0]["total_size"] if size_results else 0
            
            # Processing success rate
            total_processed = results_collection.count_documents({})
            successful_processed = results_collection.count_documents({"status": "completed"})
            processing_success_rate = successful_processed / max(total_processed, 1)
            
            # Average processing time
            time_pipeline = [
                {"$match": {"processing_time": {"$exists": True}}},
                {"$group": {"_id": None, "avg_time": {"$avg": "$processing_time"}}}
            ]
            time_results = list(results_collection.aggregate(time_pipeline))
            avg_processing_time = time_results[0]["avg_time"] if time_results else 0
            
            return DocumentStats(
                total_documents=total_documents,
                documents_by_type=documents_by_type,
                documents_by_module=documents_by_module,
                documents_by_source=documents_by_source,
                total_storage_size=total_storage_size,
                processing_success_rate=processing_success_rate,
                avg_processing_time=avg_processing_time,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return DocumentStats(
                total_documents=0,
                documents_by_type={},
                documents_by_module={},
                documents_by_source={},
                total_storage_size=0,
                processing_success_rate=0.0,
                avg_processing_time=0.0,
                last_updated=datetime.now()
            )
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated data"""
        try:
            # Get document info
            doc_info = await self.get_document_info(document_id)
            if not doc_info:
                return False
            
            # Delete from document_info collection
            docs_collection = get_mongo_collection("document_info")
            docs_collection.delete_one({"document_id": document_id})
            
            # Delete from processing_results collection
            results_collection = get_mongo_collection("processing_results")
            results_collection.delete_one({"document_id": document_id})
            
            # Note: Vector embeddings and images would need to be deleted from
            # Milvus and MongoDB respectively, but this requires more complex
            # tracking of which embeddings belong to which document
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False