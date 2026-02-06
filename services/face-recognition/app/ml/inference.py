"""
Face Detection and Recognition Inference Utilities

Provides face detection (YuNet) and recognition (SFace/ArcFace) capabilities.
The recognizer is selected based on RECOGNIZER_TYPE configuration.

Exports:
- detect_faces: Face detection with YuNet
- extract_face_features: Feature extraction with configured recognizer
- compare_face_features: Cosine similarity comparison
- draw_face_rectangles: Visualization utility
- get_face_landmarks: Extract 5-point landmarks from detection
"""

import os
from typing import Optional, List, Tuple, Union

import cv2
import numpy as np

from core.logger import get_logger
from core.config import settings

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# YuNet constants (for backward compatibility)
YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
SFACE_FILENAME = 'face_recognition_sface_2021dec.onnx'

# Model parameters
YUNET_INPUT_SIZE = (640, 640)
YUNET_SCORE_THRESHOLD = 0.7
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
SFACE_SIMILARITY_THRESHOLD = 0.55


def _find_models_dir() -> str:
    """Find the models directory from environment or common locations."""
    # First try settings
    try:
        return settings.models.models_path
    except Exception:
        pass
    
    # Fallback to environment
    env_path = os.environ.get("MODELS_PATH")
    if env_path:
        return env_path
    
    # Search common paths
    search_paths = [
        os.path.join(_SCRIPT_DIR, 'models'),
        os.path.join(os.path.dirname(_SCRIPT_DIR), 'models'),
        os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), 'models'),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return os.path.join(_SCRIPT_DIR, 'models')


DEFAULT_MODELS_DIR = _find_models_dir()
YUNET_PATH = os.path.join(DEFAULT_MODELS_DIR, YUNET_FILENAME)
SFACE_PATH = os.path.join(DEFAULT_MODELS_DIR, SFACE_FILENAME)


# =============================================================================
# FACE DETECTOR (YuNet)
# =============================================================================

_detector = None


def _get_detector():
    """Get or create the YuNet face detector."""
    global _detector
    
    if _detector is not None:
        return _detector
    
    yunet_path = settings.models.yunet_path if settings else YUNET_PATH
    
    if not os.path.exists(yunet_path):
        raise FileNotFoundError(f"YuNet model not found at {yunet_path}")
    
    try:
        _detector = cv2.FaceDetectorYN.create(
            model=yunet_path,
            config='',
            input_size=YUNET_INPUT_SIZE,
            score_threshold=settings.models.yunet_score_threshold if settings else YUNET_SCORE_THRESHOLD,
            nms_threshold=settings.models.yunet_nms_threshold if settings else YUNET_NMS_THRESHOLD,
            top_k=settings.models.yunet_top_k if settings else YUNET_TOP_K
        )
        logger.info(f"YuNet detector loaded: {yunet_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize YuNet: {e}")
    
    return _detector


# Initialize detector on first use
detector = property(lambda self: _get_detector())


# =============================================================================
# FACE DETECTION
# =============================================================================

def detect_faces(
    frame: np.ndarray,
    input_size: Tuple[int, int] = None,
    score_threshold: float = None,
    nms_threshold: float = YUNET_NMS_THRESHOLD,
    return_landmarks: bool = False
) -> Optional[Union[np.ndarray, List[Tuple[int, int, int, int]]]]:
    """
    Detect faces in a BGR image using YuNet.

    Args:
        frame: Input BGR image
        input_size: Input size for detector (uses frame size if None for accurate landmarks)
        score_threshold: Detection confidence threshold (uses config default if None)
        nms_threshold: Non-max suppression threshold
        return_landmarks: If True, return full face data with landmarks

    Returns:
        If return_landmarks=True: np.ndarray shape [num_faces, 15]
        If return_landmarks=False: List of (x, y, w, h) tuples
        None if no faces detected
    """
    if frame is None or not hasattr(frame, 'shape'):
        raise ValueError("Frame is None or invalid")
    
    det = _get_detector()
    
    # Use config threshold if not specified
    if score_threshold is None:
        score_threshold = settings.models.yunet_score_threshold if settings else YUNET_SCORE_THRESHOLD
    
    # Use frame dimensions for accurate landmark detection (no distortion)
    # This avoids the aspect ratio mismatch that causes landmark misalignment
    if input_size is None:
        # Use actual frame dimensions (width, height)
        frame_h, frame_w = frame.shape[:2]
        input_size = (frame_w, frame_h)
    
    # Only resize if input_size differs from frame size
    frame_h, frame_w = frame.shape[:2]
    if input_size[0] != frame_w or input_size[1] != frame_h:
        resized = cv2.resize(frame, input_size)
        needs_rescale = True
    else:
        resized = frame
        needs_rescale = False
    
    det.setInputSize(input_size)

    try:
        _, faces = det.detect(resized)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return None

    if faces is None or len(faces) == 0:
        return None

    # Rescale to original frame size if we resized
    if needs_rescale:
        sx = frame_w / input_size[0]
        sy = frame_h / input_size[1]
        
        faces_rescaled = faces.astype(np.float32).copy()
        faces_rescaled[:, [0, 2, 5, 7, 9, 11, 13]] *= sx  # x coords and width
        faces_rescaled[:, [1, 3, 6, 8, 10, 12, 14]] *= sy  # y coords and height
    else:
        faces_rescaled = faces.astype(np.float32)

    if return_landmarks:
        return faces_rescaled
    
    bboxes = faces_rescaled[:, :4].astype(int)
    return [(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) for bbox in bboxes]


# =============================================================================
# FEATURE EXTRACTION (Uses Recognizer Factory)
# =============================================================================

def extract_face_features(frame: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract face feature vector using the configured recognizer (SFace or ArcFace).

    Args:
        frame: Full BGR image
        face_row: Face detection row from YuNet [x, y, w, h, score, ...landmarks]

    Returns:
        Feature vector (128-dim for SFace, 512-dim for ArcFace) or None on failure
    """
    if frame is None or face_row is None:
        raise ValueError("Input frame or face_row is missing")
    
    try:
        from .recognizer_factory import get_recognizer
        recognizer = get_recognizer()
        return recognizer.extract_features(frame, face_row)
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return None


def compare_face_features(
    feature1: np.ndarray,
    feature2: np.ndarray,
    threshold: float = None
) -> Tuple[float, bool]:
    """
    Compare two face features using cosine similarity.

    Args:
        feature1: First feature vector
        feature2: Second feature vector
        threshold: Similarity threshold for match (uses config default if None)

    Returns:
        Tuple of (similarity_score, is_match)
    """
    if feature1 is None or feature2 is None:
        raise ValueError("Both features must be provided")
    
    try:
        from .recognizer_factory import get_recognizer
        recognizer = get_recognizer()
        
        if threshold is None:
            threshold = recognizer.default_threshold
        
        return recognizer.match(feature1, feature2, threshold)
    except Exception as e:
        logger.error(f"Face comparison error: {e}")
        return 0.0, False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_feature_dimension() -> int:
    """Get feature dimension of the active recognizer."""
    from .recognizer_factory import get_recognizer
    return get_recognizer().feature_dim


def get_similarity_threshold() -> float:
    """Get default similarity threshold of the active recognizer."""
    from .recognizer_factory import get_recognizer
    return get_recognizer().default_threshold


def get_recognizer_name() -> str:
    """Get name of the active recognizer."""
    from .recognizer_factory import get_recognizer
    return get_recognizer().name


def draw_face_rectangles(
    frame: np.ndarray,
    faces: Union[np.ndarray, List],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: Optional[List[str]] = None
) -> None:
    """
    Draw rectangles and optional labels on detected faces.

    Args:
        frame: Image to draw on (modified in place)
        faces: List of [x, y, w, h] or ndarray
        color: BGR color tuple
        thickness: Line thickness
        labels: Optional list of labels for each face
    """
    if faces is None:
        return
    
    faces_arr = np.array(faces)
    for i, face in enumerate(faces_arr):
        x, y, w, h = map(int, face[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        if labels and i < len(labels):
            label = str(labels[i])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_face_landmarks(face_row: np.ndarray) -> np.ndarray:
    """
    Extract 5-point landmarks from YuNet detection row.

    Args:
        face_row: 15-element array [x, y, w, h, score, l0x, l0y, ..., l4y]

    Returns:
        np.ndarray shape [5, 2] for landmark points:
        [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    if face_row is None or len(face_row) < 15:
        raise ValueError("Invalid face_row for landmarks extraction")
    
    return np.array([
        [face_row[5], face_row[6]],    # left eye
        [face_row[7], face_row[8]],    # right eye
        [face_row[9], face_row[10]],   # nose
        [face_row[11], face_row[12]],  # left mouth
        [face_row[13], face_row[14]],  # right mouth
    ], dtype=np.float32)


def get_face_detector():
    """Get the loaded YuNet face detector model."""
    return _get_detector()


def get_face_recognizer():
    """
    Get the active face recognizer.
    
    Deprecated: Use get_recognizer() from recognizer_factory instead.
    """
    from .recognizer_factory import get_recognizer
    return get_recognizer()


__all__ = [
    'detect_faces',
    'extract_face_features',
    'compare_face_features',
    'draw_face_rectangles',
    'get_face_landmarks',
    'get_face_detector',
    'get_face_recognizer',
    'get_feature_dimension',
    'get_similarity_threshold',
    'get_recognizer_name',
    'YUNET_PATH',
    'SFACE_PATH',
    'YUNET_INPUT_SIZE',
    'SFACE_SIMILARITY_THRESHOLD',
]
