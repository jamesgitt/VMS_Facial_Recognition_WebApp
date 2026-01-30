"""
Machine Learning Module

Provides face detection, recognition, and indexing capabilities using:
- YuNet: Face detection model
- SFace/ArcFace: Face recognition/feature extraction models (switchable)
- HNSW: Approximate nearest neighbor search for fast matching

Usage:
    from ml import (
        detect_faces,
        extract_face_features,
        compare_face_features,
        get_recognizer,
        get_index,
        HNSWIndexManager,
    )
    
    # Face detection
    faces = detect_faces(image, return_landmarks=True)
    
    # Feature extraction (uses configured recognizer: SFace or ArcFace)
    recognizer = get_recognizer()
    features = recognizer.extract_features(image, faces[0])
    
    # Face comparison
    score, is_match = recognizer.match(feature1, feature2)
    
    # HNSW index for fast search (auto-matches recognizer)
    index = get_index()
    index.add_visitor(visitor_id, features)
    matches = index.search(features, k=5)
"""

# Inference functions (face detection and recognition)
from .inference import (
    detect_faces,
    extract_face_features,
    compare_face_features,
    draw_face_rectangles,
    get_face_landmarks,
    get_face_detector,
    get_face_recognizer,
    YUNET_FILENAME,
    SFACE_FILENAME,
    YUNET_INPUT_SIZE,
    YUNET_SCORE_THRESHOLD,
    YUNET_NMS_THRESHOLD,
    YUNET_TOP_K,
    SFACE_SIMILARITY_THRESHOLD,
)

# HNSW index manager
from .hnsw_index import (
    HNSWIndexManager,
    DEFAULT_DIMENSION,
    DEFAULT_M,
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_MAX_ELEMENTS,
    HNSW_AVAILABLE,
)

# Recognizer factory (SFace/ArcFace switching)
from .recognizer_factory import (
    get_recognizer,
    reset_recognizer,
)

# Index factory (recognizer-aware HNSW)
from .index_factory import (
    get_index,
    reset_index,
    get_index_for_recognizer,
    get_index_stats,
)

# Model download utility
from .download_models import (
    download_model,
)

__all__ = [
    # Inference functions
    'detect_faces',
    'extract_face_features',
    'compare_face_features',
    'draw_face_rectangles',
    'get_face_landmarks',
    'get_face_detector',
    'get_face_recognizer',
    # Recognizer factory
    'get_recognizer',
    'reset_recognizer',
    # HNSW index
    'HNSWIndexManager',
    'get_index',
    'reset_index',
    'get_index_for_recognizer',
    'get_index_stats',
    # Model download
    'download_model',
    # Constants - YuNet
    'YUNET_FILENAME',
    'YUNET_INPUT_SIZE',
    'YUNET_SCORE_THRESHOLD',
    'YUNET_NMS_THRESHOLD',
    'YUNET_TOP_K',
    # Constants - SFace
    'SFACE_FILENAME',
    'SFACE_SIMILARITY_THRESHOLD',
    # Constants - HNSW
    'DEFAULT_DIMENSION',
    'DEFAULT_M',
    'DEFAULT_EF_CONSTRUCTION',
    'DEFAULT_EF_SEARCH',
    'DEFAULT_MAX_ELEMENTS',
    'HNSW_AVAILABLE',
]
