"""
Router module initialization
Exports all routers for easy importing in main.py
"""

from .auth import router as auth_router
from .chat import router as chat_router
from .anxiety import router as anxiety_router
from .documents import router as documents_router
from .analytics import router as analytics_router
from .admin import router as admin_router

__all__ = [
    "auth_router",
    "chat_router", 
    "anxiety_router",
    "documents_router",
    "analytics_router",
    "admin_router"
]