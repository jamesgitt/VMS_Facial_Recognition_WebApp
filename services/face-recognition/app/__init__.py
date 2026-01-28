"""
Face Recognition API Application Package

This package provides a FastAPI-based face detection and recognition service
using YuNet and SFace ONNX models.

Usage:
    # Import the FastAPI app
    from app import app
    
    # Or run directly
    python -m app.main
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
__version__ = "1.0.0"
