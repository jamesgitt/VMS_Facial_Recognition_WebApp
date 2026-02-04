"""Factory for creating face recognizer instances with fallback support."""

from typing import Optional
from functools import lru_cache

from .recognizer_base import FaceRecognizerBase
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Cached recognizer instance
_recognizer: Optional[FaceRecognizerBase] = None
_fallback_used: bool = False


def _create_sface_recognizer() -> Optional[FaceRecognizerBase]:
    """Attempt to create SFace recognizer."""
    try:
        from .sface_recognizer import SFaceRecognizer
        recognizer = SFaceRecognizer(model_path=settings.models.sface_path)
        logger.info(f"SFace recognizer loaded: {settings.models.sface_path}")
        return recognizer
    except Exception as e:
        logger.warning(f"Failed to load SFace recognizer: {e}")
        return None


def _create_arcface_recognizer() -> Optional[FaceRecognizerBase]:
    """Attempt to create ArcFace recognizer."""
    try:
        from .arcface_recognizer import ArcFaceRecognizer
        recognizer = ArcFaceRecognizer(
            model_path=settings.models.arcface_path,
            feature_dim=settings.models.arcface_feature_dim
        )
        logger.info(f"ArcFace recognizer loaded: {settings.models.arcface_path}")
        return recognizer
    except Exception as e:
        logger.warning(f"Failed to load ArcFace recognizer: {e}")
        return None


def get_recognizer() -> FaceRecognizerBase:
    """
    Get the configured face recognizer instance with fallback support.
    
    If the primary recognizer (based on RECOGNIZER_TYPE) fails to load,
    automatically falls back to the alternative recognizer.
    
    Fallback order:
    - If RECOGNIZER_TYPE=sface: Try SFace -> Fallback to ArcFace
    - If RECOGNIZER_TYPE=arcface: Try ArcFace -> Fallback to SFace
    
    Returns cached instance after first successful load.
    
    Raises:
        RuntimeError: If both recognizers fail to load
    """
    global _recognizer, _fallback_used
    
    if _recognizer is not None:
        return _recognizer
    
    recognizer_type = settings.models.recognizer_type.lower()
    
    if recognizer_type == "arcface":
        # Primary: ArcFace, Fallback: SFace
        _recognizer = _create_arcface_recognizer()
        if _recognizer is None:
            logger.warning("ArcFace failed, falling back to SFace...")
            _recognizer = _create_sface_recognizer()
            if _recognizer is not None:
                _fallback_used = True
    else:
        # Primary: SFace, Fallback: ArcFace
        _recognizer = _create_sface_recognizer()
        if _recognizer is None:
            logger.warning("SFace failed, falling back to ArcFace...")
            _recognizer = _create_arcface_recognizer()
            if _recognizer is not None:
                _fallback_used = True
    
    if _recognizer is None:
        raise RuntimeError(
            "Failed to load any face recognizer. "
            "Ensure model files exist: "
            f"SFace: {settings.models.sface_path}, "
            f"ArcFace: {settings.models.arcface_path}"
        )
    
    if _fallback_used:
        logger.warning(f"Using FALLBACK recognizer: {_recognizer.name} ({_recognizer.feature_dim}-dim)")
    else:
        logger.info(f"Using recognizer: {_recognizer.name} ({_recognizer.feature_dim}-dim)")
    
    return _recognizer


def is_fallback_active() -> bool:
    """Check if the fallback recognizer is being used."""
    return _fallback_used


def get_active_recognizer_info() -> dict:
    """Get information about the currently active recognizer."""
    global _recognizer, _fallback_used
    
    if _recognizer is None:
        return {
            "loaded": False,
            "name": None,
            "feature_dim": None,
            "is_fallback": False,
            "configured_type": settings.models.recognizer_type,
        }
    
    return {
        "loaded": True,
        "name": _recognizer.name,
        "feature_dim": _recognizer.feature_dim,
        "is_fallback": _fallback_used,
        "configured_type": settings.models.recognizer_type,
    }


def reset_recognizer():
    """Reset cached recognizer (useful for testing or config changes)."""
    global _recognizer, _fallback_used
    _recognizer = None
    _fallback_used = False