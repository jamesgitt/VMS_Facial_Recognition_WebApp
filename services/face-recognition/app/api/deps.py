"""
API Dependencies

FastAPI dependency injection for routes.
"""

from typing import Optional
from fastapi import Depends, HTTPException, UploadFile, File, Form

from core.config import settings, Settings
from core.state import app_state, AppState
from core.logger import get_logger

logger = get_logger(__name__)


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def get_state() -> AppState:
    """Get application state."""
    return app_state


def require_models_loaded(state: AppState = Depends(get_state)) -> AppState:
    """
    Dependency that ensures ML models are loaded.
    
    Raises:
        HTTPException: If models are not loaded
    """
    if not state.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Service is starting up."
        )
    return state


def require_initialized(state: AppState = Depends(get_state)) -> AppState:
    """
    Dependency that ensures application is fully initialized.
    
    Raises:
        HTTPException: If not initialized
    """
    if not state.initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing."
        )
    return state


async def get_image_input(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
) -> tuple[Optional[UploadFile], Optional[str]]:
    """
    Dependency for routes that accept image via file upload or base64.
    
    Returns:
        Tuple of (upload_file, base64_string) - at least one will be provided
    
    Raises:
        HTTPException: If neither image source is provided
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required (file or base64)")
    return image, image_base64


def get_threshold(
    threshold: float = Form(None),
) -> float:
    """Get similarity threshold with default from settings."""
    return threshold or settings.models.sface_similarity_threshold


__all__ = [
    "get_settings",
    "get_state",
    "require_models_loaded",
    "require_initialized",
    "get_image_input",
    "get_threshold",
]
