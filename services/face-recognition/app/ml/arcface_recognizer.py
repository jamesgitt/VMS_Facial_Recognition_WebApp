"""ArcFace recognizer implementation using ONNX Runtime."""

import cv2
import numpy as np

from .recognizer_base import FaceRecognizerBase
from core.logger import get_logger

logger = get_logger(__name__)

# ArcFace alignment template (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class ArcFaceRecognizer(FaceRecognizerBase):
    """ArcFace face recognition using ONNX Runtime."""
    
    def __init__(self, model_path: str, feature_dim: int = 512):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        
        self._session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self._input_name = self._session.get_inputs()[0].name
        self._feature_dim = feature_dim
        logger.info(f"ArcFace loaded: {model_path} ({feature_dim}-dim)")
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    @property
    def default_threshold(self) -> float:
        return 0.45
    
    @property
    def name(self) -> str:
        return "ArcFace"
    
    def _get_landmarks(self, face_row: np.ndarray) -> np.ndarray:
        """Extract 5-point landmarks from YuNet detection."""
        return np.array([
            [face_row[5], face_row[6]],    # left eye
            [face_row[7], face_row[8]],    # right eye
            [face_row[9], face_row[10]],   # nose
            [face_row[11], face_row[12]],  # left mouth
            [face_row[13], face_row[14]],  # right mouth
        ], dtype=np.float32)
    
    def _align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face to ArcFace template."""
        tform = cv2.estimateAffinePartial2D(landmarks, ARCFACE_DST)[0]
        aligned = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)
        return aligned
    
    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """Preprocess aligned face for ArcFace input."""
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 127.5
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face
    
    def extract_features(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        landmarks = self._get_landmarks(face_row)
        aligned = self._align_face(image, landmarks)
        input_tensor = self._preprocess(aligned)
        
        embedding = self._session.run(None, {self._input_name: input_tensor})[0]
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
        
        return embedding.astype(np.float32)