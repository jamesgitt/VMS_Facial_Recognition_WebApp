"""
Pydantic Schemas for Face Recognition API

Contains request/response models for all API endpoints.
"""

from .detection import (
    DetectRequest,
    FaceDetection,
    DetectionResponse,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
)
from .recognition import (
    RecognizeRequest,
    VisitorMatch,
    VisitorRecognitionResponse,
)
from .comparison import (
    CompareRequest,
    CompareResponse,
)
from .common import (
    HealthResponse,
    ModelStatusResponse,
    ModelInfo,
    ModelInfoResponse,
    HNSWStatusResponse,
    ValidateImageRequest,
    ValidateImageResponse,
    ErrorResponse,
    WebSocketMessage,
    # HNSW Management
    HNSWAddVisitorRequest,
    HNSWAddVisitorFeatureRequest,
    HNSWAddVisitorResponse,
    HNSWRebuildRequest,
    HNSWRebuildResponse,
    HNSWSyncRequest,
    HNSWSyncResponse,
)

__all__ = [
    # Detection
    "DetectRequest",
    "FaceDetection",
    "DetectionResponse",
    "FeatureExtractionRequest",
    "FeatureExtractionResponse",
    # Recognition
    "RecognizeRequest",
    "VisitorMatch",
    "VisitorRecognitionResponse",
    # Comparison
    "CompareRequest",
    "CompareResponse",
    # Common
    "HealthResponse",
    "ModelStatusResponse",
    "ModelInfo",
    "ModelInfoResponse",
    "HNSWStatusResponse",
    "ValidateImageRequest",
    "ValidateImageResponse",
    "ErrorResponse",
    "WebSocketMessage",
    # HNSW Management
    "HNSWAddVisitorRequest",
    "HNSWAddVisitorFeatureRequest",
    "HNSWAddVisitorResponse",
    "HNSWRebuildRequest",
    "HNSWRebuildResponse",
    "HNSWSyncRequest",
    "HNSWSyncResponse",
]
