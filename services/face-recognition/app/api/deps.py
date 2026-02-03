"""
API Dependencies

FastAPI dependency injection for routes.
"""

from typing import Optional
from fastapi import Depends, HTTPException, UploadFile, File, Form, Header, Security
from fastapi.security import APIKeyHeader

from core.config import settings, Settings
from core.state import app_state, AppState
from core.logger import get_logger

logger = get_logger(__name__)

# API Key security scheme
api_key_header = APIKeyHeader(
    name=settings.auth.api_key_header,
    auto_error=False,
    description="API key for authentication"
)


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


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Verify API key authentication.
    
    If authentication is disabled (no API_KEY set), allows all requests.
    If enabled, validates the provided API key.
    
    Returns:
        The API key if valid, None if auth is disabled
    
    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    # If auth is not enabled, allow all requests
    if not settings.auth.is_enabled:
        return None
    
    # Auth is enabled, API key is required
    if api_key is None:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Validate API key
    if api_key != settings.auth.api_key:
        logger.warning("API request with invalid API key")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


def require_api_key(
    api_key: Optional[str] = Depends(verify_api_key),
) -> Optional[str]:
    """
    Dependency to require API key for a route.
    
    Usage:
        @router.post("/protected", dependencies=[Depends(require_api_key)])
        async def protected_endpoint():
            ...
    """
    return api_key


__all__ = [
    "get_settings",
    "get_state",
    "require_models_loaded",
    "require_initialized",
    "get_image_input",
    "get_threshold",
    "verify_api_key",
    "require_api_key",
    "api_key_header",
]
