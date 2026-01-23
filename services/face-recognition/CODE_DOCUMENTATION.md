# VMS Facial Recognition System
## Complete Code Documentation

---

**Version:** 1.1  
**Last Updated:** January 2026  
**Author:** Development Team

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Structure](#3-file-structure)
4. [Core Modules](#4-core-modules)
5. [API Endpoints](#5-api-endpoints)
6. [Configuration](#6-configuration)
7. [Deployment](#7-deployment)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Overview

### Purpose
A facial recognition system that detects, extracts, and matches faces against a local index of enrolled identities using deep learning models and approximate nearest neighbor search. *Note: No external database dependency (all identities are managed locally in files).*

### Key Features
- **Face Detection**: YuNet neural network for real-time face detection
- **Feature Extraction**: SFace model for 128-dimensional facial embeddings
- **Fast Search**: HNSW algorithm for O(log n) similarity search
- **Identity Management**: Identities enrolled/removed via REST API endpoints
- **REST API**: FastAPI with automatic OpenAPI docs
- **WebSocket**: Real-time face detection streaming

### Technology Stack

| Component         | Technology        | Purpose                    |
|-------------------|------------------|----------------------------|
| Backend Framework | FastAPI           | Async REST API             |
| Face Detection    | YuNet (ONNX)      | Detect faces and landmarks |
| Face Recognition  | SFace (ONNX)      | 128-dim feature extraction |
| ANN Search        | hnswlib           | Fast approximate matching  |
| Storage           | Local disk (JSON, binary) | Identities, features, index |
| Image Processing  | OpenCV, Pillow    | Image manipulation         |
| Frontend          | Next.js           | Web interface              |

---

## 2. Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Web App    │  │  Mobile App │  │  IoT Device │             │
│  │  (Next.js)  │  │             │  │  (Camera)   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                  │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │   │
│  │  │ REST API   │ │ WebSocket  │ │ CORS Middleware    │   │   │
│  │  │ Endpoints  │ │ Handler    │ │ Error Handler      │   │   │
│  │  └────────────┘ └────────────┘ └────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│  │ Image       │  │ Inference   │  │ HNSW Index          │     │
│  │ Loader      │  │ Engine      │  │ Manager             │     │
│  │             │  │             │  │                     │     │
│  │ - Base64    │  │ - YuNet     │  │ - Build index       │     │
│  │ - File      │  │ - SFace     │  │ - Search k-NN       │     │
│  │ - URL       │  │ - Compare   │  │ - Persist to disk   │     │
│  └─────────────┘  └─────────────┘  └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          Local Filesystem (no database)                 │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │ identities.json (*.png, *.jpg etc.)             │    │   │
│  │  │ - id, firstName, lastName, imagePath, features  │    │   │
│  │  │ hnsw_visitor_index.bin, hnsw_visitor_metadata.pkl   │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Client sends image (base64/file)
           │
           ▼
2. Image Loader validates and decodes
           │
           ▼
3. YuNet detects faces + landmarks
           │
           ▼
4. SFace extracts 128-dim features
           │
           ▼
5. HNSW searches for k nearest neighbors in local index
           │
           ▼
6. Return matched identity/identities with confidence
```

---

## 3. File Structure

```
VMS_Facial_Recognition_TEST/
│
├── services/face-recognition/
│   ├── app/
│   │   ├── face_recog_api.py          # Main FastAPI application
│   │   ├── inference.py               # Face detection & recognition
│   │   ├── hnsw_index.py              # HNSW ANN index manager
│   │   ├── image_loader.py            # Multi-format image loading
│   │   ├── identity_store.py          # Local disk identity management
│   │   ├── download_models.py         # Model download utility
│   │   └── models/                    # ONNX model files
│   │       ├── face_detection_yunet_2023mar.onnx
│   │       └── face_recognition_sface_2021dec.onnx
│   ├── test_images/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.test
│   └── venvback/
│
├── apps/facial_recog_web_app/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx
│   │   │   ├── camera/page.tsx
│   │   │   └── _components/
│   │   │       └── camera.tsx
│   │   └── ...
│   ├── Dockerfile
│   └── package.json
│
├── docker-compose.yml
├── .env
└── documentation/
```

---

## 4. Core Modules

### 4.1 face_recog_api.py (Main Application)

**Purpose:** FastAPI application with REST endpoints and WebSocket for face recognition and local identity management.

**Key Components:**

```python
# Application initialization
app = FastAPI(title="Face Recognition API", version="1.1.0")

# Startup event - loads models and builds HNSW index from local storage
@app.on_event("startup")
def load_models():
    # 1. Load YuNet face detector
    # 2. Load SFace feature extractor
    # 3. Initialize HNSW index manager
    # 4. Load identities from disk (JSON/PKL)
    # 5. Build HNSW index from local features
```

**Key endpoints**

- POST /api/v1/detect – Detect faces in image
- POST /api/v1/extract-features – Extract face feature vector(s)
- POST /api/v1/compare – Compare two faces
- POST /api/v1/recognize – Recognize or match to enrolled identities in local index
- POST /api/v1/identities/enroll – Add new identity to local store/index
- POST /api/v1/identities/remove – Remove identity from local store/index
- GET  /api/v1/identities – List enrolled identities
- GET  /api/v1/health – Health check
- GET  /api/v1/hnsw/status – HNSW index status
- WS   /ws/realtime – Real-time face detection

**Configuration Variables:**

| Variable        | Default             | Description                        |
|-----------------|---------------------|-------------------------------------|
| `MODELS_PATH`   | `models`            | ONNX models directory               |
| `DATA_DIR`      | `data`              | Local data directory for identities |
| `IDENTITIES_FILE` | identities.json    | File mapping identities             |
| `INDEX_DIR`     | `models`            | HNSW index directory                |
| `CORS_ORIGINS`  | `*`                 | Allowed origins                     |

---

### 4.2 inference.py (Face Detection & Recognition)

**Purpose:** Handles all face detection and feature extraction via ONNX models.

**Key Functions:**

```python
def get_face_detector(models_path: str) -> cv2.FaceDetectorYN: ...
def get_face_recognizer(models_path: str) -> cv2.FaceRecognizerSF: ...
def detect_faces(image: np.ndarray, return_landmarks=True) -> List: ...
def extract_face_features(image: np.ndarray, face: np.ndarray) -> np.ndarray: ...
def compare_faces(feature1: np.ndarray, feature2: np.ndarray) -> float: ...
```

**Model Specifications:**

| Model  | File                                   | Input           | Output            |
|--------|----------------------------------------|-----------------|-------------------|
| YuNet  | face_detection_yunet_2023mar.onnx      | 320x320 image   | Faces + landmarks |
| SFace  | face_recognition_sface_2021dec.onnx    | Aligned face    | 128-dim vector    |

---

### 4.3 hnsw_index.py (HNSW Index Manager)

**Purpose:** High-performance approximate nearest neighbor search for face matching (all identities managed locally, not in a database).

**Key Class:**

```python
class HNSWIndexManager:
    def __init__(self, ...)
    def add_identity(self, identity_id: str, feature: np.ndarray, metadata: Dict = None) -> bool: ...
    def add_identities_batch(self, identities: List[Tuple[str, np.ndarray, Dict]]) -> int: ...
    def search(self, query_feature: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]: ...
    def rebuild_from_local(self, identities_file: str) -> int: ...
    def save(self): ...
    def load(self) -> bool: ...
```

**Index Files:**

| File                     | Content                      | Example Size        |
|--------------------------|------------------------------|---------------------|
| hnsw_visitor_index.bin   | HNSW graph structure         | ~1–70 MB            |
| hnsw_visitor_metadata.pkl| Identity IDs and metadata    | ~KBs                |

---

### 4.4 identity_store.py (Local Disk Identity Management)

**Purpose:** Manage identity list, images, and features in local files (JSON/PKL).

**Key Functions:**

```python
def load_identities(filepath: str) -> List[Dict]:
    # Load identities from file (JSON)
def save_identities(filepath: str, identities: List[Dict]): ...
def enroll_identity(info: Dict, feature: np.ndarray, image_path: str): ...
def remove_identity(identity_id: str): ...
def list_identities(): ...
```

**Example identity record:**

```json
{
  "id": "john-doe-123",
  "firstName": "John",
  "lastName": "Doe",
  "imagePath": "data/john-doe-123.png",
  "features": [0.25, 0.13, ...]  // 128 floats
}
```

---

### 4.5 image_loader.py (Image Loading)

**Purpose:** Centralized image loading with support for multiple formats.

**Key Functions:**  
See original; unchanged except all loads, validation are from images in file system or uploaded, not from DB.

---

## 5. API Endpoints

### 5.1 Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-23T10:30:00Z"
}
```

---

### 5.2 Face Detection

```http
POST /api/v1/detect
Content-Type: multipart/form-data

image: <file>
threshold: 0.6 (optional)
```

**Response:**
```json
{
  "faces": [
    [x, y, width, height, landmark1_x, landmark1_y, ..., confidence]
  ],
  "count": 1
}
```

---

### 5.3 Feature Extraction

```http
POST /api/v1/extract-features
Content-Type: multipart/form-data

image: <file>
```

**Response:**
```json
{
  "features": [[0.123, -0.456, ...]],
  "num_faces": 1
}
```

---

### 5.4 Face Comparison

```http
POST /api/v1/compare
Content-Type: multipart/form-data

image1: <file>
image2: <file>
threshold: 0.363 (optional)
```

**Response:**
```json
{
  "similarity_score": 0.85,
  "is_match": true
}
```

---

### 5.5 Face Recognition

```http
POST /api/v1/recognize
Content-Type: multipart/form-data

image: <file>
threshold: 0.363 (optional)
```

**Response:**
```json
{
  "identity_id": "john-doe-123",
  "confidence": 0.85,
  "matched": true,
  "firstName": "John",
  "lastName": "Doe",
  "matches": [
    {
      "identity_id": "john-doe-123",
      "match_score": 0.85,
      "is_match": true,
      "firstName": "John",
      "lastName": "Doe"
    }
  ]
}
```

---

### 5.6 Enroll Identity

```http
POST /api/v1/identities/enroll
Content-Type: multipart/form-data

image: <file>
firstName: John
lastName: Doe
id: (optional custom id)
```

**Response:**
```json
{
  "identity_id": "john-doe-123",
  "status": "enrolled"
}
```

---

### 5.7 Remove Identity

```http
POST /api/v1/identities/remove
Content-Type: application/json

{
  "identity_id": "john-doe-123"
}
```

**Response:**
```json
{
  "identity_id": "john-doe-123",
  "status": "removed"
}
```

---

### 5.8 List Identities

```http
GET /api/v1/identities
```

**Response:**
```json
[
  {
    "identity_id": "john-doe-123",
    "firstName": "John",
    "lastName": "Doe"
    // ...
  },
  // ...
]
```

---

### 5.9 HNSW Status

```http
GET /api/v1/hnsw/status
```

**Response:**
```json
{
  "available": true,
  "initialized": true,
  "total_vectors": 87,
  "dimension": 128,
  "index_type": "HNSW",
  "m": 8,
  "ef_construction": 40,
  "ef_search": 10,
  "identities_indexed": 87
}
```

---

### 5.10 WebSocket Real-time Detection

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/realtime");
// ... see original for details ...
```

---

## 6. Configuration

### Environment Variables

#### Backend (.env.test)

```env
# No database config required!
# Identity/Index storage
DATA_DIR=data
IDENTITIES_FILE=identities.json
INDEX_DIR=models

# Models
MODELS_PATH=models

# API
CORS_ORIGINS=*
```

#### Docker (.env)

```env
API_PORT=8000

# Frontend
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 7. Deployment

### Local Development

```bash
# Backend
cd services/face-recognition
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd app
uvicorn face_recog_api:app --reload --port 8000

# Frontend
cd apps/facial_recog_web_app
pnpm install
pnpm dev
```

### Docker

```bash
docker compose up -d
docker compose logs -f backend
docker compose down
```

---

## 8. Troubleshooting

### Common Issues

| Issue               | Cause                        | Solution                |
|---------------------|-----------------------------|-------------------------|
| "No face found"     | Poor lighting, angle         | Improve image quality   |
| "Model not found"   | Missing ONNX files           | Run `download_models.py`|
| "Index missing"     | No identities enrolled yet   | Enroll an identity      |
| "Low similarity"    | Different people             | Threshold is correct    |
| "Memory error"      | Too many identities in RAM   | Fewer identities or add RAM |

### Performance Tuning

| Parameter         | Lower Value       | Higher Value       |
|-------------------|------------------|-------------------|
| `ef_search`       | Faster, less accurate | Slower, more accurate |
| `ef_construction` | Faster build      | Better quality index  |
| `M`               | Less memory      | More connections   |

### Logs

```bash
docker logs facial_recog_backend -f
python face_recog_api.py 2>&1 | tee app.log
```

---

## Quick Reference

### Start Backend Locally

```bash
cd services/face-recognition
.\venvback\Scripts\Activate  # Windows
source venv/bin/activate     # Linux
cd app
uvicorn face_recog_api:app --host 0.0.0.0 --port 8000
```

### Test API

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/hnsw/status
curl http://localhost:8000/api/v1/recognize -F "image=@photo.jpg"
```

### API Documentation

Interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

**End of Documentation**
