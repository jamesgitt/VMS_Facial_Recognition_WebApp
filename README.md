# Face Recognition API

A production-ready FastAPI microservice for face detection and recognition using ONNX models with HNSW approximate nearest neighbor search.

**Features:**
- Face detection using YuNet
- Face recognition using SFace (recommended) or ArcFace
- HNSW index for fast similarity search (~71k visitors)
- API key authentication
- Automatic fallback between recognizers
- PostgreSQL database integration

---

## Quick Start

### Prerequisites

- Python 3.11+ or Docker
- Model files in `app/models/`:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`
  - `arcface.onnx` (optional, for fallback)

### Run with Docker

```bash
# Build and start
docker compose up --build

# Start in background
docker compose up -d
```

### Run Locally

```bash
cd services/face-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API available at: **http://localhost:8000**

### API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **OpenAPI Schema:** http://localhost:8000/openapi.json

---

## Model Comparison Results

Based on testing with ~71,500 indexed visitors:

| Model | Top-1 Accuracy | Best F1 | Speed | Recommended Threshold |
|-------|---------------|---------|-------|----------------------|
| **SFace** | **81.71%** | **90.27%** | **2.1x faster** | **0.70** |
| ArcFace | 73.71% | 84.87% | 1.0x | 0.90 |

**Recommendation:** Use SFace (default) for better accuracy, speed, and storage efficiency.

See [docs/MODEL_COMPARISON_RESULTS.pdf](docs/MODEL_COMPARISON_RESULTS.pdf) for detailed analysis.

---

## API Endpoints

### Public Endpoints (No Auth Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/api/v1/health` | Health with recognizer info |
| GET | `/models/status` | Model loading status |
| GET | `/models/info` | Model metadata |
| GET | `/api/v1/hnsw/status` | HNSW index status |

### Protected Endpoints (API Key Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/detect` | Detect faces in image |
| POST | `/api/v1/extract-features` | Extract face feature vectors |
| POST | `/api/v1/compare` | Compare two faces |
| POST | `/api/v1/recognize` | Recognize visitor from database |
| POST | `/api/v1/validate-image` | Validate image before processing |

### HNSW Index Management (API Key Required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/hnsw/add-visitor` | Add single visitor to index (with image) |
| POST | `/api/v1/hnsw/add-visitor-feature` | Add visitor with pre-extracted feature |
| POST | `/api/v1/hnsw/rebuild` | Full rebuild from database |
| POST | `/api/v1/hnsw/sync` | Sync new visitors (incremental update) |
| DELETE | `/api/v1/hnsw/visitor/{id}` | Remove visitor from index |

### Authentication

Include API key in request header:

```bash
curl -X POST "http://localhost:8000/api/v1/recognize" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data"}'
```

---

## Configuration

### Environment Variables

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Authentication
API_KEY=your-secure-api-key
AUTH_ENABLED=true

# CORS
CORS_ORIGINS=https://your-frontend.com

# Model Settings
RECOGNIZER_TYPE=sface              # sface or arcface
SFACE_SIMILARITY_THRESHOLD=0.70    # Recommended for best F1
ARCFACE_SIMILARITY_THRESHOLD=0.90

# Database
USE_DATABASE=true
DATABASE_URL=postgresql://user:password@host:5432/dbname

# HNSW Index
HNSW_M=32
HNSW_EF_CONSTRUCTION=400
HNSW_EF_SEARCH=400
```

### Generate API Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Project Structure

```
face-recognition/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── api/                    # Routes & dependencies
│   │   ├── routes.py          # HTTP endpoints
│   │   ├── deps.py            # Auth & dependencies
│   │   └── websocket.py       # WebSocket detection
│   ├── core/                   # Core infrastructure
│   │   ├── config.py          # Pydantic settings
│   │   ├── logger.py          # Centralized logging
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── state.py           # App state management
│   ├── ml/                     # Machine learning
│   │   ├── inference.py       # YuNet/SFace inference
│   │   ├── hnsw_index.py      # HNSW index manager
│   │   ├── sface_recognizer.py
│   │   ├── arcface_recognizer.py
│   │   ├── recognizer_factory.py  # With fallback support
│   │   └── index_factory.py
│   ├── db/                     # Database layer
│   ├── pipelines/              # Business logic
│   ├── schemas/                # Pydantic models
│   ├── utils/                  # Utilities
│   └── models/                 # ONNX model files
├── scripts/
│   ├── extract_features_to_db.py   # Feature extraction
│   ├── rebuild_for_recognizer.py   # Index rebuild
│   └── convert_md_to_pdf.py        # Documentation
├── tests/
│   ├── compare_recognizers.py      # Model comparison
│   ├── optimize_thresholds.py      # Threshold tuning
│   ├── test_accuracy.py            # Accuracy testing
│   └── verify_indexes.py           # Index verification
├── docs/
│   ├── MODEL_COMPARISON_RESULTS.md
│   └── MODEL_COMPARISON_RESULTS.pdf
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Fallback Mechanism

The recognizer factory automatically falls back if the primary model fails:

- **RECOGNIZER_TYPE=sface:** SFace → ArcFace (fallback)
- **RECOGNIZER_TYPE=arcface:** ArcFace → SFace (fallback)

Check fallback status via health endpoint:
```json
{
  "status": "ok",
  "recognizer": "ArcFace",
  "feature_dim": 512,
  "is_fallback": true
}
```

---

## Scripts

### Rebuild HNSW Index

```bash
# Rebuild for SFace (128-dim)
python scripts/rebuild_for_recognizer.py --recognizer sface

# Rebuild for ArcFace (512-dim)
python scripts/rebuild_for_recognizer.py --recognizer arcface
```

### Run Accuracy Tests

```bash
# Compare models at optimal thresholds
python tests/compare_recognizers.py --arcface-threshold 0.90 --sface-threshold 0.70

# Optimize thresholds
python tests/optimize_thresholds.py
```

---

## Docker Commands

```bash
docker compose build               # Build image
docker compose up                  # Start (foreground)
docker compose up -d               # Start (background)
docker compose logs -f backend     # View logs
docker compose down                # Stop and remove
docker compose restart backend     # Restart service
```

---

## Integration Example

### JavaScript/TypeScript

```typescript
const API_URL = 'http://localhost:8000';
const API_KEY = 'your-api-key';

async function recognizeVisitor(base64Image: string) {
  const response = await fetch(`${API_URL}/api/v1/recognize-json`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    },
    body: JSON.stringify({
      image: base64Image,
      threshold: 0.70,
      top_k: 5,
    }),
  });
  
  return await response.json();
}
```

### Python

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"

def recognize_visitor(base64_image: str):
    response = requests.post(
        f"{API_URL}/api/v1/recognize-json",
        headers={"X-API-Key": API_KEY},
        json={
            "image": base64_image,
            "threshold": 0.70,
            "top_k": 5,
        },
    )
    return response.json()
```

### Add New Visitor to Index (After DB Insert)

```python
# Call this after inserting a new visitor into your database
def add_visitor_to_index(visitor_id: str, base64_image: str, first_name: str, last_name: str):
    response = requests.post(
        f"{API_URL}/api/v1/hnsw/add-visitor",
        headers={"X-API-Key": API_KEY},
        json={
            "visitor_id": visitor_id,
            "image": base64_image,
            "first_name": first_name,
            "last_name": last_name,
        },
    )
    return response.json()

# Or sync multiple new visitors from database
def sync_new_visitors(visitor_ids: list = None):
    response = requests.post(
        f"{API_URL}/api/v1/hnsw/sync",
        headers={"X-API-Key": API_KEY},
        json={"visitor_ids": visitor_ids},  # None = sync all new
    )
    return response.json()
```

---

## Troubleshooting

### Models Not Found

```bash
# Check models exist
ls app/models/
# Expected: face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx

# Download models
python -m app.ml.download_models
```

### Authentication Errors

```bash
# Check if API_KEY is set
echo $API_KEY

# Test with API key
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/health
```

### HNSW Index Issues

```bash
# Check index status
curl http://localhost:8000/api/v1/hnsw/status

# Rebuild index via API
curl -X POST "http://localhost:8000/api/v1/hnsw/rebuild" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Or via script
python scripts/rebuild_for_recognizer.py --recognizer sface
```

### View Logs

```bash
# Docker
docker compose logs -f backend

# Local
# Logs are printed to stdout with [INFO], [WARNING], [ERROR] prefixes
```

---

## Performance

| Metric | SFace | ArcFace |
|--------|-------|---------|
| Feature Dimension | 128 | 512 |
| Index File Size | ~37 MB | ~146 MB |
| Queries/Second | ~25 | ~12 |
| Top-1 Accuracy | 81.71% | 73.71% |

---

## License

MIT License - See LICENSE file for details.
