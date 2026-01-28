# Code Improvement Roadmap

> **Overall Rating: 7.5/10**  
> The codebase shows solid engineering practices and modular intent, but would benefit from improved testing, stricter typing, and some modernization.

---

## Review Overview

| Area                  | Score | Notes                                                  |
|-----------------------|-------|--------------------------------------------------------|
| Code Organization     | 8/10  | Modules have clear purpose; main API file too large    |
| Documentation         | 7/10  | Docstrings are present but inconsistent in detail      |
| Error Handling        | 7/10  | Needs custom exceptions and better error responses     |
| Type Hints            | 6/10  | Coverage is partial; missing in key functions          |
| Testing               | 4/10  | No unit or integration tests found                     |
| Configuration         | 8/10  | Well-structured, environment-driven                    |
| Code Duplication      | 6/10  | Some common patterns not yet deduplicated              |
| Naming Conventions    | 8/10  | Consistent and descriptive naming                      |
| Separation of Concerns| 7/10  | Some mixing of logic and routes in main API file       |

---

## Recommended Project Structure

### Current Structure (Flat)
```
services/face-recognition/
├── app/
│   ├── face_recog_api.py      # 836 lines - routes + logic + schemas + config
│   ├── inference.py
│   ├── database.py
│   ├── hnsw_index.py
│   ├── image_loader.py
│   ├── logger.py
│   ├── main.py
│   ├── download_models.py
│   ├── extract_features_to_db.py
│   ├── rebuild_index.py
│   └── models/
├── test_images/
├── requirements.txt
└── Dockerfile
```

### Proposed Structure (Modular)
```
services/face-recognition/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Entry point only
│   │
│   ├── api/                       # API layer (routes only)
│   │   ├── __init__.py
│   │   ├── routes.py              # All FastAPI route definitions
│   │   ├── deps.py                # Dependency injection (get_db, get_index, etc.)
│   │   └── websocket.py           # WebSocket handlers
│   │
│   ├── core/                      # Core configuration & utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings (all env vars)
│   │   ├── logger.py              # Centralized logging
│   │   ├── exceptions.py          # Custom exception classes
│   │   └── state.py               # Load visitors from DB/test_images
│   │
│   ├── ml/                        # ML model layer
│   │   ├── __init__.py
│   │   ├── inference.py           # YuNet/SFace model wrappers
│   │   ├── hnsw_index.py          # HNSW index manager
│   │   └── download_models.py     # Model downloader
│   │
│   ├── db/                        # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py          # Connection pool, get_connection()
│   │   ├── queries.py             # SQL query functions
│   │   └── models.py              # DB models (if using ORM later)
│   │
│   ├── utils/                     # Shared utilities
│   │   ├── __init__.py
│   │   └── image_loader.py        # Image loading/validation
│   │
│   └── models/                    # ONNX model files
│       ├── face_detection_yunet_2023mar.onnx
│       └── face_recognition_sface_2021dec.onnx
│
├── scripts/                       # Standalone CLI scripts
│   ├── extract_features_to_db.py
│   └── rebuild_index.py
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── unit/
│   │   ├── test_inference.py
│   │   ├── test_hnsw_index.py
│   │   └── test_image_loader.py
│   ├── integration/
│   │   ├── test_api_detect.py
│   │   ├── test_api_recognize.py
│   │   └── test_database.py
│   └── fixtures/
│       └── test_face.jpg
│
├── test_images/                   # Fallback visitor images
├── requirements.txt
├── requirements-dev.txt           # pytest, black, mypy, ruff
├── pyproject.toml                 # Tool configs (black, mypy, ruff)
├── Dockerfile
├── .env.example
└── README.md
```# AppState singleton
│   │
│   ├── schemas/                   # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── detection.py           # DetectRequest, DetectionResponse
│   │   ├── recognition.py         # RecognitionRequest, VisitorRecognitionResponse
│   │   ├── comparison.py          # CompareRequest, RecognitionResponse
│   │   └── common.py              # Shared schemas (ModelStatus, Health, etc.)
│   │
│   ├── pipelines/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── detection.py           # Face detection logic
│   │   ├── recognition.py         # Face recognition/matching logic
│   │   ├── comparison.py          # Face comparison logic
│   │   ├── feature_extraction.py  # Feature extraction helpers
│   │   └── visitor_loader.py      

### Key Benefits of Proposed Structure

| Benefit | Description |
|---------|-------------|
| **Separation of Concerns** | Routes, business logic, ML, and DB are clearly separated |
| **Testability** | Each layer can be unit tested in isolation |
| **Scalability** | Easy to add new endpoints, pipelines, or ML models |
| **Maintainability** | Smaller files (~100-200 lines each vs 836) |
| **Dependency Injection** | `deps.py` provides testable dependencies |
| **Clear Imports** | `from app.pipelines.detection import detect_faces` |

### Migration Path

1. **Phase 1**: Create `core/` folder with `config.py`, `logger.py`, `exceptions.py`
2. **Phase 2**: Create `schemas/` folder, move Pydantic models
3. **Phase 3**: Create `pipelines/` folder, extract business logic from routes
4. **Phase 4**: Create `api/` folder, slim down routes to just HTTP handling
5. **Phase 5**: Reorganize `ml/` and `db/` folders
6. **Phase 6**: Add `tests/` folder with pytest infrastructure

---

## 1. Modularization & Structure

- **Split up `face_recog_api.py` (836 lines) into smaller modules**:  
  - Only keep routes in `face_recog_api.py`
  - Move business logic to `detection.py`, `recognition.py`, `comparison.py`
  - Move configuration to `config.py`
  - Pydantic models → `schemas.py`
  - Startup logic → `startup.py`
- **Folder naming:**  
  - If `app/` is already inside `services/`, don’t add another `pipelines/` subfolder.  
  ```
  services/
    └── app/
        ├── face_recog_api.py
        ├── config.py
        ├── schemas.py
        ├── startup.py
        ├── detection.py
        ├── recognition.py
        └── comparison.py
  ```
  *Avoid `services/app/pipelines/` paths if already at the correct depth.*

---

## 2. Global State

- Replace ad hoc global variables with a singleton `AppState` class in `state.py`:
  ```python
  from dataclasses import dataclass, field
  from typing import Dict, Any, Optional

  @dataclass
  class AppState:
      face_detector: Optional[Any] = None
      face_recognizer: Optional[Any] = None
      hnsw_manager: Optional[Any] = None
      visitor_features: Dict[str, Dict] = field(default_factory=dict)
      use_database: bool = False

  app_state = AppState()
  ```
  This enables dependency injection and safer state sharing.

---

## 3. Unified Configuration

- Centralize all settings using Pydantic in `config.py`:
  ```python
  from pydantic_settings import BaseSettings
  from typing import Optional

  class Settings(BaseSettings):
      # Paths, thresholds, DB settings, CORS, HNSW index params etc.
      models_path: str = "models"
      visitor_images_dir: str = "../test_images"
      score_threshold: float = 0.7
      compare_threshold: float = 0.55
      max_image_width: int = 1920
      max_image_height: int = 1920
      use_database: bool = False
      database_url: Optional[str] = None
      db_table_name: str = 'public."Visitor"'
      db_visitor_id_column: str = "id"
      db_image_column: str = "base64Image"
      db_features_column: str = "facefeatures"
      db_visitor_limit: Optional[int] = None
      hnsw_max_elements: int = 100000
      hnsw_m: int = 32
      hnsw_ef_construction: int = 400
      hnsw_ef_search: int = 400
      cors_origins: str = "*"

      class Config:
          env_file = ".env"

  settings = Settings()
  ```
---

## 4. Error Handling Enhancements

- Define domain-specific exceptions in `exceptions.py` (`FaceRecognitionError` etc.).
- Replace generic excepts (e.g., `except Exception: pass`) with explicit logging:
  ```python
  except Exception as e:
      logger.warning(f"Failed to update features for {visitor_id}: {e}")
  ```
- Standardize API error responses, and use custom error classes.

---

## 5. Logging Modernization

- Replace all `print` statements and silent failures with `logging` via a shared logger:
  ```python
  # logger.py:
  import logging
  import sys

  def setup_logger(name: str = "face_recognition") -> logging.Logger:
      logger = logging.getLogger(name)
      logger.setLevel(logging.INFO)
      handler = logging.StreamHandler(sys.stdout)
      formatter = logging.Formatter(
          "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
      )
      handler.setFormatter(formatter)
      if not logger.handlers:
          logger.addHandler(handler)
      return logger

  logger = setup_logger()
  ```
- All modules:  
  ```python
  from .logger import logger
  ```
- Use context in logs: include `visitor_id`, model name, shapes, etc.

---

## 6. Improve Type Hint Coverage

- Add missing function return type annotations. Examples:
  ```python
  def load_models() -> None:
  def health() -> dict:
  def get_db_connection() -> psycopg2.extensions.connection:
  ```
- Use precise types (not just `Any`)—refactor as needed.

---

## 7. Reduce Duplication

- Add helpers for image loading/validation and face checks, e.g.:
  ```python
  def load_and_validate_image(image_data: str, source_type: str = "base64") -> np.ndarray:
      img_np = image_loader.load_image(image_data, source_type=source_type)
      image_loader.validate_image_size((img_np.shape[1], img_np.shape[0]), MAX_IMAGE_SIZE)
      return img_np

  def require_faces(img: np.ndarray, error_message: str = "No face detected") -> np.ndarray:
      faces = inference.detect_faces(img, return_landmarks=True)
      if not faces:
          raise HTTPException(status_code=400, detail=error_message)
      return faces
  ```

---

## 8. Modernize API

- Use FastAPI’s recommended lifespan context for startup:
  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      load_models()
      yield
      # cleanup if needed

  app = FastAPI(lifespan=lifespan)
  ```
- Standardize all endpoint paths under `/api/v1/` and clearly mark deprecated response fields in schemas.

---

## 9. HNSW Enhancements

- Add divide-by-zero protection to `_normalize_feature`:
  ```python
  def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
      norm = np.linalg.norm(feature)
      if norm < 1e-10:
          raise ValueError("Cannot normalize zero or near-zero vector")
      return (feature / norm).astype('float32')
  ```
- Document or implement actual removal for `remove_visitor` method.

---

## 10. Database Improvements

- Validate table names to prevent SQL injection:
  ```python
  import re

  def _validate_table_name(table_name: str) -> str:
      pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\."[a-zA-Z_][a-zA-Z0-9_]*")?$'
      if not re.match(pattern, table_name):
          raise ValueError(f"Invalid table name format: {table_name}")
      return table_name
  ```
- Ensure database connections are safely closed on shutdown using FastAPI lifecycle events.

---

## 11. Script & Batch Improvements

- Update old function signatures in migration scripts:
  ```python
  # OLD (incorrect):
  inference.get_face_detector(MODELS_PATH)
  # NEW:
  inference.get_face_detector()
  ```
- Aggregate DB updates into batch commits for improved performance.

---

## 12. Add Testing Infrastructure

- Introduce unit and integration tests:
  ```
  tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_inference.py
    ├── test_hnsw_index.py
    ├── test_database.py
    ├── test_image_loader.py
    ├── test_api_detect.py
    ├── test_api_recognize.py
    └── fixtures/
        └── test_face.jpg
  ```
---

## Quick Reference

### Files to Refactor

| File                    | Key Changes                                    |
|-------------------------|------------------------------------------------|
| face_recog_api.py       | Split into modular files, update globals, add helpers, update startup logic     |
| hnsw_index.py           | Add safe normalization and structured logging  |
| database.py             | Add table validation and ensure connection cleanup |
| inference.py            | Tighten type hints                            |
| image_loader.py         | Largely clean                                  |
| extract_features_to_db.py| Fix function signatures (see batch note above) |

**New/updated files:** `config.py`, `schemas.py`, `exceptions.py`, `logger.py`, `state.py` (optional), and `tests/` (root level).

---

## Project Strengths

- **Modular file responsibilities**
- **Graceful fallback from DB to test_images**
- **Flexible, environment-based config**
- **Efficient, normalized HNSW k-NN**
- **Clear logging/error prefixes ([OK], [WARNING], [ERROR])**
- **Context managers for safe DB connections**

---

## Quick Wins To Prioritize

- [ ] Add `py.typed` for type checking support
- [ ] Provide linters/config as `.flake8` or in `pyproject.toml`
- [ ] Supply dev requirements (`requirements-dev.txt`: pytest, black, mypy, etc.)
- [ ] Fix function signatures in feature extraction scripts
- [ ] Patch zero-vector risk in HNSW normalization
