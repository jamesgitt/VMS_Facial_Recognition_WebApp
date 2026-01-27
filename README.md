# Face Recognition Backend API

A modern REST API for face detection and recognition using YuNet (face detection) and SFace (face recognition) ONNX models. Features include PostgreSQL integration for persistent visitor storage, HNSW-based fast approximate nearest neighbor search for recognition, and WebSocket endpoints for real-time image processing.

## 🚀 Quick Start

### Prerequisites
- **Python 3.11 or 3.12** (recommended)
- **Docker & Docker Compose** (recommended)
- **PostgreSQL** (optional; falls back to test_images/ directory if not provided)

### Deploy with Docker (Recommended)

Start the backend service:
```bash
docker compose up -d backend
```
Once running, access the API at: **http://localhost:8000**

API documentation:
- Swagger UI: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

### Local Development Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venvback
   ```

2. **Activate the virtual environment:**
   - **Windows (PowerShell):** `.\venvback\Scripts\Activate.ps1`
   - **Windows (CMD):** `venvback\Scripts\activate.bat`
   - **Linux/Mac:** `source venvback/bin/activate`

3. **Install dependencies:**
   ```bash
   cd services/face-recognition
   pip install -r requirements.txt
   ```

4. **Download face models:**
   ```bash
   python app/download_models.py
   ```
   This will download:
   - `face_detection_yunet_2023mar.onnx`
   - `face_recognition_sface_2021dec.onnx`

5. **Define environment variables:**
   Create a file named `.env` in `services/face-recognition/` (example below):
   ```env
   MODELS_PATH=app/models
   CORS_ORIGINS=http://localhost:3000,http://localhost:3001
   USE_DATABASE=true
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/visitors_db
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=visitors_db
   DB_USER=postgres
   DB_PASSWORD=postgres
   DB_TABLE_NAME=visitors
   ```

6. **Launch the API locally:**
   ```bash
   cd app
   uvicorn face_recog_api:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📋 API Endpoints

### Health & Model Status
- `GET /api/v1/health` — Service health check
- `GET /api/v1/models/status` — Model loading status
- `GET /api/v1/models/info` — Model details
- `GET /api/v1/hnsw/status` — HNSW index status

### Face Detection
- `POST /api/v1/detect`
  - **Input:** JSON `{ image: base64 }` or form file upload
  - **Output:** Faces detected (boxes, landmarks, confidence)

### Face Recognition
- `POST /api/v1/recognize`
  - **Input:** JSON `{ image: base64 }` or form file upload
  - **Output:** Visitor ID match, confidence score, and top-N results

### Face Comparison
- `POST /api/v1/compare`
  - **Input:** JSON `{ image1, image2 }` (base64)
  - **Output:** Cosine similarity score and match verdict

### Feature Extraction
- `POST /api/v1/extract-features`
  - **Input:** JSON `{ image: base64 }`
  - **Output:** 128-d feature vector

### Image Validation
- `POST /api/v1/validate-image`
  - **Input:** JSON `{ image: base64 }` or file upload
  - **Output:** Image validation metadata (format, size, etc.)

### Real-Time Processing
- `WebSocket /ws/realtime`
  - Bidirectional real-time detection/recognition (send base64 images, receive results)
  - For live camera or video processing

## 🔧 Configuration

### Environment Variables

#### Core
- `MODELS_PATH`: Path to model files (default `/app/app/models` in Docker, `app/models` locally)
- `CORS_ORIGINS`: CSV of allowed origins for CORS (default: `*`)

#### Database
- `USE_DATABASE`: Enable PostgreSQL visitor database (`true`/`false`, default: `false`)
- `DATABASE_URL`: Full DB connection string
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_TABLE_NAME`: Individual DB settings
- `DB_VISITOR_ID_COLUMN`, `DB_IMAGE_COLUMN`, `DB_FEATURES_COLUMN`: (advanced overrides)
- `DB_VISITOR_LIMIT`: Limit loaded visitors (default: 0 = all)

#### Model Parameters
- `YUNET_SCORE_THRESHOLD`: Face detection (default: `0.6`)
- `SFACE_SIMILARITY_THRESHOLD`: Matching (default: `0.55`)

#### Example API Requests

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Face detection
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_here", "score_threshold": 0.6}'

# Face recognition
curl -X POST http://localhost:8000/api/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_here", "threshold": 0.55}'

# Compare faces
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"image1": "base64_image1_here", "image2": "base64_image2_here"}'
```

## 🗄️ Database Integration

The backend supports saving visitor records in PostgreSQL. If unavailable or not configured, the service defaults to the `test_images/` folder for recognition.

### DB Table Expectation

A `visitors` table should contain:
- `id`: Primary key
- `base64Image`: Visitor image (base64)
- `facefeatures`: (optional) 128-d feature vector (JSON array)

### Batch Feature Extraction

Pre-extract face features for every visitor (speeds up recognition):

```bash
python app/extract_features_to_db.py \
  --db-host localhost \
  --db-name visitors_db \
  --db-user postgres \
  --db-password postgres \
  --image-dir test_images \
  --batch-size 10
```

## 🔍 HNSW Index

HNSW enables fast similarity search (O(log n)), auto-indexes on startup, and provides persistence between restarts.
- If unavailable, service falls back to linear (slow) search.

Check status:
```bash
curl http://localhost:8000/api/v1/hnsw/status
```

## 📦 Project Structure

```
services/face-recognition/
├── app/
│   ├── face_recog_api.py
│   ├── inference.py
│   ├── database.py
│   ├── hnsw_index.py
│   ├── image_loader.py
│   ├── download_models.py
│   ├── extract_features_to_db.py
│   └── models/
│       ├── face_detection_yunet_2023mar.onnx
│       └── face_recognition_sface_2021dec.onnx
├── test_images/
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🐳 Frequently Used Docker Commands

```bash
# Build backend image
docker compose build backend

# Start backend
docker compose up -d backend

# See live logs
docker compose logs -f backend

# Stop
docker compose down backend

# Rebuild after changes
docker compose up --build backend

# Run script in container
docker compose exec backend python app/download_models.py
```

## 🧪 Testing API

**Face Detection**
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@path/to/image.jpg"
```

**Face Recognition**
```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "file=@path/to/image.jpg"
```

**Health Check**
```bash
curl http://localhost:8000/api/v1/health
```

## 🐛 Troubleshooting Tips

### Models Not Found
Make sure models exist in `app/models/`:
```bash
ls services/face-recognition/app/models/
# Should list the two ONNX models

# To download models:
cd services/face-recognition
python app/download_models.py
```

### Database Connection Problems
- Ensure PostgreSQL is running: `docker compose ps postgres`
- Confirm all DB settings in `.env`
- If DB fails, fallback is `test_images/`

### Port Conflicts
Change API port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"
```

### HNSW Indexing Issues
- Verify visitor records have valid images/features
- Check logs: `docker compose logs backend`
- Index build may take a while on large datasets

## 📚 Major Dependencies

See `requirements.txt` for details, but includes:

- `fastapi` (API)
- `uvicorn` (ASGI server)
- `opencv-python` (vision)
- `numpy` (numerics)
- `Pillow` (image)
- `hnswlib` (nearest neighbor search)
- `psycopg2-binary` (PostgreSQL)
- `pydantic` (validation)
- `python-dotenv` (env loader)
- `websockets` (WS support)

> **Note:** DeepFace or TensorFlow are optional—install manually if you need them.

## 🚢 Production Deployment Tips

- Set CORS origins appropriately
- Use secure production DB credentials
- Enable HNSW for performance
- Prefer a reverse proxy (nginx/traefik) for HTTPS
- Add robust health checks and monitoring
- Use Docker Compose or Kubernetes for scaling/HA

### Cloud Options
- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform
- Kubernetes

## 📝 License

[Specify your license here]
