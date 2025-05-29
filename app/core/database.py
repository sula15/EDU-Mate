"""
Database connection management for MongoDB and Milvus
"""

import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import pymongo
from pymongo import MongoClient
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseManager:
    """Manages database connections for MongoDB and Milvus"""
    
    def __init__(self):
        self._mongo_client: Optional[MongoClient] = None
        self._mongo_db = None
        self._milvus_connected = False
        
    # MongoDB Methods
    def connect_mongodb(self) -> MongoClient:
        """Connect to MongoDB"""
        try:
            if self._mongo_client is None:
                self._mongo_client = MongoClient(
                    settings.MONGODB_URI,
                    serverSelectionTimeoutMS=5000
                )
                # Test connection
                self._mongo_client.server_info()
                self._mongo_db = self._mongo_client[settings.MONGODB_DB]
                logger.info(f"Successfully connected to MongoDB: {settings.MONGODB_DB}")
            
            return self._mongo_client
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_mongo_db(self):
        """Get MongoDB database instance"""
        if self._mongo_db is None:
            self.connect_mongodb()
        return self._mongo_db
    
    def get_mongo_collection(self, collection_name: str):
        """Get MongoDB collection"""
        db = self.get_mongo_db()
        return db[collection_name]
    
    def close_mongodb(self):
        """Close MongoDB connection"""
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
            logger.info("MongoDB connection closed")
    
    # Milvus Methods
    def connect_milvus(self) -> bool:
        """Connect to Milvus"""
        try:
            if not self._milvus_connected:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=30
                )
                self._milvus_connected = True
                logger.info(f"Successfully connected to Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def get_milvus_collection(self, collection_name: str) -> Optional[Collection]:
        """Get Milvus collection"""
        try:
            self.connect_milvus()
            
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.load()
                return collection
            else:
                logger.warning(f"Milvus collection '{collection_name}' does not exist")
                return None
        except Exception as e:
            logger.error(f"Error getting Milvus collection '{collection_name}': {e}")
            return None
    
    def create_milvus_collection(self, collection_name: str, schema: CollectionSchema) -> Optional[Collection]:
        """Create a new Milvus collection"""
        try:
            self.connect_milvus()
            
            if utility.has_collection(collection_name):
                logger.info(f"Milvus collection '{collection_name}' already exists")
                return Collection(collection_name)
            
            collection = Collection(collection_name, schema)
            logger.info(f"Created Milvus collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error creating Milvus collection '{collection_name}': {e}")
            return None
    
    def close_milvus(self):
        """Close Milvus connection"""
        try:
            if self._milvus_connected:
                connections.disconnect("default")
                self._milvus_connected = False
                logger.info("Milvus connection closed")
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {e}")
    
    # Database Status Methods
    def get_mongodb_status(self) -> Dict[str, Any]:
        """Get MongoDB connection status"""
        try:
            client = self.connect_mongodb()
            info = client.server_info()
            db = self.get_mongo_db()
            collections = db.list_collection_names()
            
            return {
                "connected": True,
                "version": info.get("version", "Unknown"),
                "database": settings.MONGODB_DB,
                "collections": len(collections),
                "collection_names": collections
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    def get_milvus_status(self) -> Dict[str, Any]:
        """Get Milvus connection status"""
        try:
            self.connect_milvus()
            collections = utility.list_collections()
            
            collection_info = []
            for collection_name in collections:
                try:
                    collection = Collection(collection_name)
                    collection.load()
                    stats = collection.get_stats()
                    collection_info.append({
                        "name": collection_name,
                        "entities": stats.get("row_count", 0)
                    })
                except Exception as e:
                    collection_info.append({
                        "name": collection_name,
                        "entities": "Error",
                        "error": str(e)
                    })
            
            return {
                "connected": True,
                "collections": len(collections),
                "collection_info": collection_info
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    # Cleanup Methods
    def close_all_connections(self):
        """Close all database connections"""
        self.close_mongodb()
        self.close_milvus()


# Global database manager instance
db_manager = DatabaseManager()


# Helper functions for easy access
def get_mongo_collection(collection_name: str):
    """Get MongoDB collection - convenience function"""
    return db_manager.get_mongo_collection(collection_name)


def get_milvus_collection(collection_name: str) -> Optional[Collection]:
    """Get Milvus collection - convenience function"""
    return db_manager.get_milvus_collection(collection_name)


# Student-related collections
def get_student_profiles_collection():
    """Get student profiles collection"""
    return get_mongo_collection(settings.STUDENT_PROFILES_COLLECTION)


def get_student_interactions_collection():
    """Get student interactions collection"""
    return get_mongo_collection(settings.STUDENT_INTERACTIONS_COLLECTION)


def get_student_analytics_collection():
    """Get student analytics collection"""
    return get_mongo_collection(settings.STUDENT_ANALYTICS_COLLECTION)


def get_images_collection():
    """Get images collection"""
    return get_mongo_collection(settings.MONGODB_IMAGES_COLLECTION)


# Milvus collections
def get_text_embeddings_collection():
    """Get text embeddings collection"""
    return get_milvus_collection(settings.MILVUS_TEXT_COLLECTION)


def get_image_embeddings_collection():
    """Get image embeddings collection"""
    return get_milvus_collection(settings.MILVUS_IMAGE_COLLECTION)


# Lifespan events for FastAPI
@asynccontextmanager
async def lifespan(app):
    """Database lifespan manager for FastAPI"""
    try:
        # Startup
        logger.info("Starting up database connections...")
        db_manager.connect_mongodb()
        db_manager.connect_milvus()
        
        # Create upload directory
        settings.create_upload_dir()
        
        # Create MongoDB indexes
        await create_mongodb_indexes()
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down database connections...")
        db_manager.close_all_connections()


async def create_mongodb_indexes():
    """Create necessary MongoDB indexes"""
    try:
        # Student profiles indexes
        profiles_collection = get_student_profiles_collection()
        profiles_collection.create_index("email", unique=True, sparse=True)
        profiles_collection.create_index("last_active")
        
        # Student interactions indexes
        interactions_collection = get_student_interactions_collection()
        interactions_collection.create_index([("student_id", 1)])
        interactions_collection.create_index([("timestamp", -1)])
        interactions_collection.create_index([("module_id", 1)])
        interactions_collection.create_index([("learning_style_id", 1)])
        
        # Student analytics indexes
        analytics_collection = get_student_analytics_collection()
        analytics_collection.create_index([("student_id", 1)], unique=True)
        analytics_collection.create_index([("last_updated", -1)])
        
        # Images collection indexes
        images_collection = get_images_collection()
        images_collection.create_index("milvus_id")
        images_collection.create_index("lecture_code")
        images_collection.create_index("module_id")
        images_collection.create_index("page_number")
        
        logger.info("Successfully created MongoDB indexes")
        
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")


# Health check functions
async def check_database_health() -> Dict[str, Any]:
    """Check health of all database connections"""
    mongodb_status = db_manager.get_mongodb_status()
    milvus_status = db_manager.get_milvus_status()
    
    return {
        "mongodb": mongodb_status,
        "milvus": milvus_status,
        "overall_healthy": mongodb_status["connected"] and milvus_status["connected"]
    }