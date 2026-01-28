"""
API Module

FastAPI routers for face recognition endpoints.
"""

from fastapi import APIRouter

from .routes import router as main_router
from .websocket import router as ws_router

# Combined router with all endpoints
api_router = APIRouter()
api_router.include_router(main_router)
api_router.include_router(ws_router)

__all__ = [
    "api_router",
    "main_router",
    "ws_router",
]
