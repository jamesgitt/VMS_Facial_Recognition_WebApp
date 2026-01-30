"""
Model Download Script
Downloads YuNet, SFace, and ArcFace ONNX models for face detection and recognition.

Supports:
- YuNet: Face detection
- SFace: Face recognition (128-dim, fast)
- ArcFace: Face recognition (512-dim, more accurate)
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

from core.logger import get_logger
logger = get_logger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _SCRIPT_DIR = Path(__file__).parent
    env_file = _SCRIPT_DIR.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    pass

# Model configuration
MODELS_DIR = os.environ.get("MODELS_PATH", "models")
MIN_MODEL_SIZE_BYTES = 1_000_000  # ONNX models should be at least 1MB

MODEL_INFO = {
    'yunet': {
        'url': "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        'filename': 'face_detection_yunet_2023mar.onnx',
        'hash': None,
        'required': True,
        'description': 'Face detection model'
    },
    'sface': {
        'url': "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        'filename': 'face_recognition_sface_2021dec.onnx',
        'hash': None,
        'required': True,
        'description': 'Face recognition model (128-dim)'
    },
    # ArcFace models from InsightFace
    'arcface_r50': {
        'url': "https://huggingface.co/aarnphm/insightface-onnx/resolve/main/buffalo_l/w600k_r50.onnx",
        'filename': 'arcface_r50.onnx',
        'hash': None,
        'required': False,
        'description': 'ArcFace ResNet-50 (512-dim, recommended)'
    },
    'arcface_r100': {
        'url': "https://huggingface.co/aarnphm/insightface-onnx/resolve/main/buffalo_l/w600k_mbf.onnx",
        'filename': 'arcface_r100.onnx',
        'hash': None,
        'required': False,
        'description': 'ArcFace MobileFaceNet (512-dim, faster)'
    },
}

# ArcFace model aliases for easier access
ARCFACE_MODELS = ['arcface_r50', 'arcface_r100']

# HTTP headers to avoid 403 errors
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def _format_size(size_bytes: int) -> str:
    """Format byte size as MB string."""
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def download_file(url: str, filepath: str, model_name: str) -> bool:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        filepath: Path to save the file
        model_name: Name of model (for display)
    
    Returns:
        True if download successful, False otherwise
    """
    logger.info(f"Downloading {model_name} from {url}...")
    
    try:
        request = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(request) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            if file_size > 0:
                logger.info(f"File size: {_format_size(file_size)}")
            
            with open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        
        downloaded_size = os.path.getsize(filepath)
        logger.info(f"Downloaded {model_name} to {filepath} ({_format_size(downloaded_size)})")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def verify_file(filepath: str, expected_hash: Optional[str] = None) -> bool:
    """
    Verify downloaded file exists and optionally check hash.
    
    Args:
        filepath: Path to file
        expected_hash: Optional SHA256 hash for verification
    
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} does not exist")
        return False
    
    if expected_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_hash:
            logger.error(f"File {filepath} hash mismatch")
            return False
    
    return True


def download_model(model_key: str, models_dir: str = MODELS_DIR) -> bool:
    """
    Download a model if not already present.
    
    Args:
        model_key: Model identifier ('yunet' or 'sface')
        models_dir: Directory to save models
    
    Returns:
        True if model is available (downloaded or already exists), False otherwise
    """
    if model_key not in MODEL_INFO:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    info = MODEL_INFO[model_key]
    filepath = os.path.join(models_dir, info['filename'])
    
    # Check if model already exists and is valid
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size >= MIN_MODEL_SIZE_BYTES:
            logger.info(f"Model {model_key} already exists ({_format_size(file_size)})")
            return True
        logger.warning(f"Model {model_key} appears invalid ({file_size} bytes). Re-downloading...")
        os.remove(filepath)
    
    # Download and verify
    if not download_file(info['url'], filepath, model_key):
        return False
    
    if not verify_file(filepath, info['hash']):
        logger.error(f"Failed to verify model {model_key}")
        return False
    
    return True


def download_required_models(models_dir: str = MODELS_DIR) -> bool:
    """
    Download only required models (YuNet and SFace).
    
    Args:
        models_dir: Directory to save models
    
    Returns:
        True if all required models are available
    """
    os.makedirs(models_dir, exist_ok=True)
    
    success = True
    for model_key, info in MODEL_INFO.items():
        if info.get('required', False):
            if download_model(model_key, models_dir):
                logger.info(f"{model_key.upper()} model ready")
            else:
                logger.error(f"Failed to download {model_key.upper()} model")
                success = False
    
    return success


def download_arcface(model_variant: str = 'arcface_r50', models_dir: str = MODELS_DIR) -> bool:
    """
    Download ArcFace model.
    
    Args:
        model_variant: 'arcface_r50' or 'arcface_r100'
        models_dir: Directory to save models
    
    Returns:
        True if download successful
    """
    if model_variant not in ARCFACE_MODELS:
        logger.error(f"Unknown ArcFace variant: {model_variant}. Use one of: {ARCFACE_MODELS}")
        return False
    
    os.makedirs(models_dir, exist_ok=True)
    return download_model(model_variant, models_dir)


def list_models() -> None:
    """Print available models and their status."""
    print("\nAvailable models:")
    print("-" * 60)
    for model_key, info in MODEL_INFO.items():
        filepath = os.path.join(MODELS_DIR, info['filename'])
        status = "✓ Downloaded" if os.path.exists(filepath) else "✗ Not downloaded"
        required = "(required)" if info.get('required', False) else "(optional)"
        print(f"  {model_key:15} {required:12} {status}")
        print(f"                   {info.get('description', '')}")
    print()


def main() -> int:
    """
    Download models based on command line arguments.
    
    Usage:
        python download_models.py           # Download required models only
        python download_models.py --all     # Download all models
        python download_models.py --arcface # Download ArcFace (arcface_r50)
        python download_models.py --list    # List available models
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download face recognition models')
    parser.add_argument('--all', action='store_true', help='Download all models including ArcFace')
    parser.add_argument('--arcface', action='store_true', help='Download ArcFace model (arcface_r50)')
    parser.add_argument('--arcface-variant', choices=ARCFACE_MODELS, default='arcface_r50',
                        help='ArcFace variant to download')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--models-dir', default=MODELS_DIR, help='Models directory')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return 0
    
    logger.info("Downloading models...")
    os.makedirs(args.models_dir, exist_ok=True)
    
    success = True
    
    # Always download required models
    if not download_required_models(args.models_dir):
        success = False
    
    # Download ArcFace if requested
    if args.arcface or args.all:
        if not download_arcface(args.arcface_variant, args.models_dir):
            success = False
    
    # Download all optional models if --all
    if args.all:
        for model_key in ARCFACE_MODELS:
            if not download_model(model_key, args.models_dir):
                logger.warning(f"Failed to download optional model: {model_key}")
    
    if success:
        logger.info("All requested models downloaded successfully")
    else:
        logger.error("Some models failed to download")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
