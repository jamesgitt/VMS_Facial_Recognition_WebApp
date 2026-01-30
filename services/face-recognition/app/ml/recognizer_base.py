"""Base interface for face recognizers."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class FaceRecognizerBase(ABC):
    """Abstract base class for face recognition models."""
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of feature vectors (e.g., 128 or 512)."""
        pass
    
    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """Return the default similarity threshold for matching."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the recognizer name for logging."""
        pass
    
    @abstractmethod
    def extract_features(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a detected face.
        
        Args:
            image: Full BGR image
            face_row: YuNet detection row [x, y, w, h, score, landmarks...]
        
        Returns:
            Normalized feature vector
        """
        pass
    
    def compare(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Compare two feature vectors using cosine similarity.
        
        Default implementation - can be overridden if needed.
        """
        return float(np.dot(feature1.flatten(), feature2.flatten()))
    
    def match(
        self, 
        feature1: np.ndarray, 
        feature2: np.ndarray, 
        threshold: float = None
    ) -> Tuple[float, bool]:
        """Compare features and return (score, is_match)."""
        threshold = threshold or self.default_threshold
        score = self.compare(feature1, feature2)
        return score, score >= threshold