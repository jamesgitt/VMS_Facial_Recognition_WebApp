"""SFace recognizer implementation using OpenCV."""

import cv2
import numpy as np

from .recognizer_base import FaceRecognizerBase
from core.logger import get_logger

logger = get_logger(__name__)


class SFaceRecognizer(FaceRecognizerBase):
    """SFace face recognition using OpenCV's FaceRecognizerSF."""
    
    def __init__(self, model_path: str):
        self._model = cv2.FaceRecognizerSF.create(
            model=model_path,
            config='',
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        logger.info(f"SFace loaded: {model_path}")
    
    @property
    def feature_dim(self) -> int:
        return 128
    
    @property
    def default_threshold(self) -> float:
        return 0.55
    
    @property
    def name(self) -> str:
        return "SFace"
    
    def extract_features(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self._model.alignCrop(image, face_row)
        feature = self._model.feature(aligned)
        return feature.flatten().astype(np.float32)
    
    def compare(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        # Use OpenCV's built-in comparison
        return float(self._model.match(
            feature1.reshape(1, -1), 
            feature2.reshape(1, -1), 
            cv2.FaceRecognizerSF_FR_COSINE
        ))
