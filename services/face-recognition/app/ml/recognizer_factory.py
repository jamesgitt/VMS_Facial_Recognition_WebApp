"""Factory for creating face recognizer instances."""

from typing import Optional
from functools import lru_cache

from .recognizer_base import FaceRecognizerBase
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Cached recognizer instance
_recognizer: Optional[FaceRecognizerBase] = None


def get_recognizer() -> FaceRecognizerBase:
    """
    Get the configured face recognizer instance.
    
    Returns cached instance based on RECOGNIZER_TYPE config.
    """
    global _recognizer
    
    if _recognizer is not None:
        return _recognizer
    
    recognizer_type = settings.models.recognizer_type.lower()
    
    if recognizer_type == "arcface":
        from .arcface_recognizer import ArcFaceRecognizer
        _recognizer = ArcFaceRecognizer(
            model_path=settings.models.arcface_path,
            feature_dim=settings.models.arcface_feature_dim
        )
    else:  # Default to sface
        from .sface_recognizer import SFaceRecognizer
        _recognizer = SFaceRecognizer(
            model_path=settings.models.sface_path
        )
    
    logger.info(f"Using recognizer: {_recognizer.name} ({_recognizer.feature_dim}-dim)")
    return _recognizer


def reset_recognizer():
    """Reset cached recognizer (useful for testing or config changes)."""
    global _recognizer
    _recognizer = None