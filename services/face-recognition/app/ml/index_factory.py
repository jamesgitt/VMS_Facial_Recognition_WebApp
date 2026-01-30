"""
Factory for creating HNSW index instances based on active recognizer.

Provides a single point of access to the HNSW index that automatically
matches the currently configured face recognizer (SFace or ArcFace).
"""

from typing import Optional

from .hnsw_index import HNSWIndexManager
from .recognizer_factory import get_recognizer
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Cached index instance
_index: Optional[HNSWIndexManager] = None


def get_index() -> HNSWIndexManager:
    """
    Get HNSW index for the currently configured recognizer.
    
    Returns cached instance matching the active recognizer.
    The index is automatically created with the correct dimension
    and recognizer-specific file naming.
    
    Returns:
        HNSWIndexManager instance for current recognizer
    """
    global _index
    
    recognizer = get_recognizer()
    
    # Check if cached index matches current recognizer
    if _index is not None:
        if (_index.recognizer_name == recognizer.name.lower() and 
            _index.dimension == recognizer.feature_dim):
            return _index
        else:
            # Recognizer changed, save old index and create new
            logger.warning(
                f"Recognizer changed from {_index.recognizer_name} to {recognizer.name}, "
                f"switching index"
            )
            _index.save()
    
    # Get index directory from settings
    index_dir = settings.hnsw.index_dir or settings.models.models_path
    
    # Create index for current recognizer
    _index = HNSWIndexManager(
        dimension=recognizer.feature_dim,
        recognizer_name=recognizer.name,
        m=settings.hnsw.m,
        ef_construction=settings.hnsw.ef_construction,
        ef_search=settings.hnsw.ef_search,
        index_dir=index_dir,
        max_elements=settings.hnsw.max_elements,
    )
    
    return _index


def reset_index() -> None:
    """
    Reset cached index.
    
    Saves the current index to disk and clears the cache.
    Useful for testing or after config changes.
    """
    global _index
    if _index is not None:
        _index.save()
        logger.info(f"Saved and reset index for {_index.recognizer_name}")
    _index = None


def get_index_for_recognizer(recognizer_name: str, feature_dim: int) -> HNSWIndexManager:
    """
    Get index for a specific recognizer (useful for migration).
    
    Does NOT affect the cached global index.
    
    Args:
        recognizer_name: Name of recognizer ('sface' or 'arcface')
        feature_dim: Feature dimension (128 or 512)
    
    Returns:
        HNSWIndexManager for the specified recognizer
    """
    index_dir = settings.hnsw.index_dir or settings.models.models_path
    
    return HNSWIndexManager(
        dimension=feature_dim,
        recognizer_name=recognizer_name,
        m=settings.hnsw.m,
        ef_construction=settings.hnsw.ef_construction,
        ef_search=settings.hnsw.ef_search,
        index_dir=index_dir,
        max_elements=settings.hnsw.max_elements,
    )


def get_index_stats() -> dict:
    """Get statistics for the current index."""
    index = get_index()
    return index.get_stats()


__all__ = [
    'get_index',
    'reset_index',
    'get_index_for_recognizer',
    'get_index_stats',
]
