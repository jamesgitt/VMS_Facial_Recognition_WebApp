# Development Guide

## 🚀 Running the Project in Development Mode

### Prerequisites

1. **Python 3.11 or 3.12** (required for OpenCV and dependencies)
2. **pip** (Python package manager)
3. **Virtual environment** (recommended)

### Step 1: Navigate to Project Directory

```bash
cd sevices/face-recognition
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install API dependencies (minimal set for development)
pip install -r requirements.txt

# Or if you have a separate requirements-api.txt
pip install fastapi uvicorn[standard] python-multipart pydantic pillow opencv-python numpy
```

### Step 4: Download Models (if not already present)

```bash
# Check if models exist
ls models/
# Should show:
# face_detection_yunet_2023mar.onnx
# face_recognition_sface_2021dec.onnx

# If models are missing, download them
python app/download_models.py
```

### Step 5: Run the Development Server

**Option A: Using main.py (Recommended)**

```bash
# From the face-recognition directory
python app/main.py --reload
```

**Option B: Using uvicorn directly**

```bash
# From the face-recognition/app directory
cd app
uvicorn face_recog_api:app --host 0.0.0.0 --port 8000 --reload
```

**Option C: Using Python module**

```bash
# From the face-recognition directory
python -m uvicorn app.face_recog_api:app --host 0.0.0.0 --port 8000 --reload
```

### Development Server Options

```bash
# Basic run
python app/main.py

# With auto-reload (recommended for development)
python app/main.py --reload

# Custom port
python app/main.py --port 8001

# Custom host
python app/main.py --host 127.0.0.1

# Debug logging
python app/main.py --log-level debug

# Skip model check (if models are guaranteed to exist)
python app/main.py --skip-model-check
```

### Verify It's Running

1. **Check the console output** - You should see:
   ```
   Face Recognition ML Microservice
   ============================================================
   Host:        0.0.0.0
   Port:        8000
   Workers:     1
   Reload:      True
   Log Level:   info
   ============================================================
   
   API will be available at: http://0.0.0.0:8000
   API Documentation: http://0.0.0.0:8000/docs
   Health Check: http://0.0.0.0:8000/api/v1/health
   ```

2. **Test the health endpoint:**
   ```bash
   curl http://localhost:8000/api/v1/health
   # Should return: {"status":"ok","time":"..."}
   ```

3. **Open API documentation:**
   - Navigate to: http://localhost:8000/docs
   - Interactive Swagger UI for testing endpoints

---

## 🔧 Development Workflow

### File Structure

```
sevices/face-recognition/
├── app/
│   ├── main.py              # Entry point (use this to run)
│   ├── face_recog_api.py    # FastAPI app (main API code)
│   ├── inference.py         # ML inference logic
│   ├── download_models.py   # Model downloader
│   └── model_testing.py     # Testing utilities
├── models/                  # ONNX models (must exist)
├── requirements.txt         # Python dependencies
└── DEV_GUIDE.md            # This file
```

### Making Changes

1. **Edit code** in `app/` directory
2. **Save file** - With `--reload` flag, server auto-restarts
3. **Test changes** via http://localhost:8000/docs

### Environment Variables

You can set these environment variables:

```bash
# Windows PowerShell
$env:MODELS_PATH="models"
$env:CORS_ORIGINS="http://localhost:3000,http://localhost:3001"

# Windows CMD
set MODELS_PATH=models
set CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Linux/Mac
export MODELS_PATH=models
export CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

Or create a `.env` file (requires `python-dotenv`):

```env
MODELS_PATH=models
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

---

## 🧪 Testing Endpoints

### Using cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Detect faces (replace BASE64_IMAGE with actual base64 string)
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_IMAGE", "score_threshold": 0.6}'
```

### Using Python

```python
import requests
import base64

# Read image and encode
with open("test_images/test_image1.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Detect faces
response = requests.post(
    "http://localhost:8000/api/v1/detect",
    json={"image": image_b64, "score_threshold": 0.6}
)
print(response.json())
```

### Using the Interactive Docs

1. Go to http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in the request body
5. Click "Execute"

---

## 🐛 Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Models Not Found

**Error**: `FileNotFoundError: YuNet model not found`

**Solution**: Download models
```bash
python app/download_models.py
```

Or set `MODELS_PATH` environment variable:
```bash
export MODELS_PATH=/path/to/models
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Use a different port
```bash
python app/main.py --port 8001
```

### OpenCV Errors

**Error**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution** (Linux): Install system dependencies
```bash
sudo apt-get update
sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### CORS Errors in Browser

**Error**: `Access to fetch at 'http://localhost:8000' has been blocked by CORS policy`

**Solution**: Set CORS_ORIGINS environment variable
```bash
export CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

---

## 📝 Development Tips

1. **Use `--reload` flag** for automatic server restart on code changes
2. **Use `--log-level debug`** for detailed logging
3. **Test endpoints** using the interactive docs at `/docs`
4. **Check model status** at `/models/status` endpoint
5. **Monitor logs** in the console for errors

---

## 🔄 Hot Reload

The `--reload` flag enables automatic server restart when you modify Python files. This is perfect for development:

```bash
python app/main.py --reload
```

Changes to these files will trigger a restart:
- `app/face_recog_api.py`
- `app/inference.py`
- Any imported Python modules

---

## 🚀 Quick Start Script

Create a `dev.sh` (Linux/Mac) or `dev.bat` (Windows) for quick startup:

**dev.sh** (Linux/Mac):
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python app/main.py --reload
```

**dev.bat** (Windows):
```batch
@echo off
cd /d %~dp0
call venv\Scripts\activate.bat 2>nul
python app\main.py --reload
```

Make executable (Linux/Mac):
```bash
chmod +x dev.sh
./dev.sh
```

---

## 📚 Next Steps

- Read the [README.md](README.md) for API documentation
- Check [IMPLEMENTATION_GAP_ANALYSIS.md](IMPLEMENTATION_GAP_ANALYSIS.md) for missing features
- Explore the API at http://localhost:8000/docs
