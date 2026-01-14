"""
Model Download Script
Downloads YuNet and Sface ONNX models for face detection and recognition
"""

# - urllib.request for downloading files
# - os/pathlib for file path handling
# - Optional: hashlib for file verification

import urllib.request
import os
import hashlib

# - YuNet model URL and filename
# - Sface model URL and filename

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download="
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?download="

# - Define models directory path
# - Get or create models directory
# - Handle relative/absolute paths

models_dir = "test_facial_recognitin_web_app/models"
os.makedirs(models_dir, exist_ok=True)
yunet_filepath = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
sface_filepath = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

def download_file(url, filepath, model_name):
    """
    Download a file from URL
    
    Args:
        url: URL to download from
        filepath: Path to save the file
        model_name: Name of model (for display)
    
    Returns:
        bool: True if successful
    """
    
    # Add User-Agent header to avoid 403 errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # Show download progress
    print(f"Downloading {model_name} from {url}...")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        with open(filepath, 'wb') as out_file:
            out_file.write(response.read())
    print(f"Downloaded {model_name} to {filepath}")
    return True

def verify_file(filepath, expected_hash=None):
    """
    Verify downloaded file exists and optionally check hash
    
    Args:
        filepath: Path to file
        expected_hash: Optional SHA256 hash for verification
    
    Returns:
        bool: True if file is valid
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist")
        return False
    # If expected_hash provided, calculate file hash and compare
    if expected_hash:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_hash:
            print(f"File {filepath} hash mismatch")
            return False
    return True

def download_model(model_key, models_dir):
    """
    Download models from multiple URLs
    
    Args:
        model_key: model name
        models_dir: Directory to save models
    
    Returns:
        bool: True if successful
    """
    # TODO: Get model info (URL, filename) from configuration
    
    # TODO: Check if model already exists, skip if valid
    
    # TODO: Try primary URL first
    
    # TODO: If primary fails, try alternative URLs
    
    # TODO: Verify downloaded file
    
    # TODO: Return True on success, False on failure
    pass

def main():
    """Main function to download all models"""
    # TODO: Print header/start message
    
    # TODO: Get/create models directory
    
    # TODO: Download each model (yunet, sface)
    
    # TODO: Print summary of results
    
    # TODO: Return success status
    pass

if __name__ == "__main__":
    # TODO: Call main() and exit with appropriate code
    pass
