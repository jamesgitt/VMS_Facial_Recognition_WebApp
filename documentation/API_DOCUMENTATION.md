# Face Recognition API Documentation

**Version**: 1.0.0  
**Base URL**: `http://localhost:8000` (development)  
**Content-Type**: `application/json` (for JSON endpoints) or `multipart/form-data` (for file upload endpoints)

---

## Table of Contents

1. [Health Check Endpoints](#health-check-endpoints)
2. [Face Detection Endpoints](#face-detection-endpoints)
3. [Feature Extraction Endpoints](#feature-extraction-endpoints)
4. [Face Comparison Endpoints](#face-comparison-endpoints)
5. [Visitor Recognition Endpoints](#visitor-recognition-endpoints)
6. [Model Information Endpoints](#model-information-endpoints)
7. [Utility Endpoints](#utility-endpoints)
8. [WebSocket Endpoint](#websocket-endpoint)
9. [Error Handling](#error-handling)
10. [Request/Response Formats](#requestresponse-formats)

---

## Health Check Endpoints

### GET `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "ok",
  "time": "2024-01-15T10:30:00.000Z"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### GET `/api/v1/health`

Versioned health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "time": "2024-01-15T10:30:00.000Z"
}
```

---

## Face Detection Endpoints

### POST `/api/v1/detect`

Detect faces in an image using YuNet face detector.

**Request Body (JSON):**
```json
{
  "image": "base64_encoded_image_string",
  "score_threshold": 0.6,
  "return_landmarks": false
}
```

**Parameters:**
- `image` (string, required): Base64-encoded image. Can include `data:image/jpeg;base64,` prefix or be plain base64.
- `score_threshold` (float, optional): Minimum confidence score for face detection (0.0-1.0). Default: `0.6`
- `return_landmarks` (boolean, optional): Whether to return facial landmarks. Default: `false`

**Response:**
```json
{
  "faces": [
    [x, y, w, h],
    [x, y, w, h]
  ],
  "count": 2
}
```

**Response Fields:**
- `faces` (array): Array of bounding boxes `[x, y, width, height]` for each detected face
- `count` (integer): Number of faces detected

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "score_threshold": 0.6,
    "return_landmarks": false
  }'
```

**Error Responses:**
- `400 Bad Request`: Invalid image data or missing required fields
- `500 Internal Server Error`: Processing error

---

### POST `/detect`

Legacy endpoint for face detection (supports file upload or form data).

**Request (multipart/form-data):**
- `image` (file, optional): Image file to upload
- `image_base64` (string, optional): Base64-encoded image
- `score_threshold` (float, optional): Default: `0.6`
- `return_landmarks` (boolean, optional): Default: `false`

**Note:** Either `image` or `image_base64` must be provided.

**Response:** Same as `/api/v1/detect`

---

## Feature Extraction Endpoints

### POST `/api/v1/extract-features`

Extract face feature vectors (embeddings) from detected faces in an image.

**Request (multipart/form-data):**
- `image` (file, optional): Image file to upload
- `image_base64` (string, optional): Base64-encoded image

**Note:** Either `image` or `image_base64` must be provided.

**Response:**
```json
{
  "features": [
    [0.123, -0.456, 0.789, ...],  // 512-dimensional feature vector for face 1
    [0.234, -0.567, 0.890, ...]   // 512-dimensional feature vector for face 2
  ],
  "num_faces": 2
}
```

**Response Fields:**
- `features` (array): Array of 512-dimensional feature vectors (one per detected face)
- `num_faces` (integer): Number of faces detected and features extracted

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/extract-features \
  -F "image=@photo.jpg"
```

**Error Responses:**
- `400 Bad Request`: Missing image or invalid image data
- `500 Internal Server Error`: Processing error

---

## Face Comparison Endpoints

### POST `/api/v1/compare`

Compare two faces and determine if they match.

**Request Body (JSON):**
```json
{
  "image1": "base64_encoded_image_string_1",
  "image2": "base64_encoded_image_string_2",
  "threshold": 0.363
}
```

**Parameters:**
- `image1` (string, required): Base64-encoded first image
- `image2` (string, required): Base64-encoded second image
- `threshold` (float, optional): Similarity threshold (0.0-1.0). Default: `0.363`

**Response:**
```json
{
  "similarity_score": 0.85,
  "is_match": true,
  "features1": null,
  "features2": null
}
```

**Response Fields:**
- `similarity_score` (float): Cosine similarity score between the two faces (0.0-1.0)
- `is_match` (boolean): Whether the similarity score exceeds the threshold
- `features1` (null): Feature vectors are not returned for security/performance
- `features2` (null): Feature vectors are not returned for security/performance

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "image1": "iVBORw0KGgoAAAANSUhEUgAA...",
    "image2": "iVBORw0KGgoAAAANSUhEUgAA...",
    "threshold": 0.363
  }'
```

**Error Responses:**
- `400 Bad Request`: Missing images, no face detected in one or both images, or invalid image data
- `500 Internal Server Error`: Processing error

---

### POST `/compare`

Legacy endpoint for face comparison (supports file upload or form data).

**Request (multipart/form-data):**
- `image1` (file, optional): First image file
- `image2` (file, optional): Second image file
- `image1_base64` (string, optional): Base64-encoded first image
- `image2_base64` (string, optional): Base64-encoded second image
- `threshold` (float, optional): Default: `0.363`

**Note:** Both images must be provided (either as files or base64).

**Response:** Same as `/api/v1/compare`

---

## Visitor Recognition Endpoints

### POST `/api/v1/recognize`

Recognize a visitor by matching their face against the database of known visitors.

**Request (multipart/form-data):**
- `image` (file, optional): Image file to upload
- `image_base64` (string, optional): Base64-encoded image
- `threshold` (float, optional): Similarity threshold. Default: `0.363`

**Note:** Either `image` or `image_base64` must be provided.

**Response:**
```json
{
  "visitor_id": "clqnb3f1j000bjo08pusl0t48",
  "confidence": 0.92,
  "matched": true,
  "visitor": "clqnb3f1j000bjo08pusl0t48",
  "match_score": 0.92,
  "matches": [
    {
      "visitor_id": "clqnb3f1j000bjo08pusl0t48",
      "match_score": 0.92,
      "is_match": true
    },
    {
      "visitor_id": "clqnbcuxg000bkv08kdesf1to",
      "match_score": 0.75,
      "is_match": false
    }
  ]
}
```

**Response Fields:**
- `visitor_id` (string|null): Database ID of the matched visitor (if match found)
- `confidence` (float|null): Match confidence score (if match found)
- `matched` (boolean): Whether a match was found above the threshold
- `visitor` (string|null): **Legacy field** - same as `visitor_id` (deprecated)
- `match_score` (float|null): **Legacy field** - same as `confidence` (deprecated)
- `matches` (array): Top 10 matches sorted by score (descending)

**Database Integration:**
- When `USE_DATABASE=true`, queries PostgreSQL database for visitor images
- Extracts features on-the-fly from database images
- Falls back to `test_images/` directory if database unavailable
- Supports up to 9,640+ visitors in production database

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "image=@visitor_photo.jpg" \
  -F "threshold=0.363"
```

**Error Responses:**
- `400 Bad Request`: Missing image or invalid image data
- `500 Internal Server Error`: Processing error or database connection error

**No Match Response:**
```json
{
  "visitor_id": null,
  "confidence": null,
  "matched": false,
  "visitor": null,
  "match_score": null,
  "matches": []
}
```

---

## Model Information Endpoints

### GET `/api/v1/models/status`

Get the loading status of face detection and recognition models.

**Response:**
```json
{
  "loaded": true,
  "details": {
    "face_detector": "<class 'cv2.dnn_Net'>",
    "face_recognizer": "<class 'cv2.dnn_Net'>"
  }
}
```

**Response Fields:**
- `loaded` (boolean): Whether both models are loaded
- `details` (object|null): Model type information (if loaded)

**Example:**
```bash
curl http://localhost:8000/api/v1/models/status
```

---

### GET `/api/v1/models/info`

Get detailed information about the face detection and recognition models.

**Response:**
```json
{
  "detector": {
    "type": "YuNet",
    "model_path": "/app/app/models/face_detection_yunet_2023mar.onnx",
    "input_size": [320, 320],
    "loaded": true
  },
  "recognizer": {
    "type": "Sface",
    "model_path": "/app/app/models/face_recognition_sface_2021dec.onnx",
    "similarity_threshold": 0.363,
    "loaded": true
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/models/info
```

---

## Utility Endpoints

### POST `/api/v1/validate-image`

Validate an image before processing (check format and size).

**Request (multipart/form-data):**
- `image` (file, optional): Image file to validate
- `image_base64` (string, optional): Base64-encoded image

**Response:**
```json
{
  "valid": true,
  "format": "jpeg",
  "size": [1920, 1080]
}
```

**Response Fields:**
- `valid` (boolean): Whether the image is valid (format and size)
- `format` (string|null): Image format (`jpg`, `jpeg`, or `png`)
- `size` (array|null): Image dimensions `[width, height]`

**Allowed Formats:** JPG, JPEG, PNG  
**Max Size:** 1920x1920 pixels

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/validate-image \
  -F "image=@photo.jpg"
```

---

## WebSocket Endpoint

### WebSocket `/ws/realtime`

Real-time face detection via WebSocket connection.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');
```

**Message Format (Request):**
```json
{
  "type": "frame",
  "image": "base64_encoded_image_string",
  "score_threshold": 0.6,
  "return_landmarks": false
}
```

**Message Parameters:**
- `type` (string, required): Must be `"frame"`
- `image` (string, required): Base64-encoded image (without `data:` prefix)
- `score_threshold` (float, optional): Default: `0.6`
- `return_landmarks` (boolean, optional): Default: `false`

**Response Format (Success):**
```json
{
  "type": "results",
  "faces": [
    {
      "bbox": [x, y, w, h],
      "confidence": 0.95,
      "landmarks": [x1, y1, x2, y2, ...]  // Only if return_landmarks=true
    }
  ],
  "count": 1
}
```

**Response Format (Error):**
```json
{
  "type": "error",
  "error": "Error message description"
}
```

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
  console.log('WebSocket connected');
  
  // Send frame
  ws.send(JSON.stringify({
    type: 'frame',
    image: base64ImageString,
    score_threshold: 0.6,
    return_landmarks: false
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  
  if (response.type === 'results') {
    console.log(`Detected ${response.count} faces`);
    response.faces.forEach((face, index) => {
      console.log(`Face ${index + 1}:`, face.bbox, `confidence: ${face.confidence}`);
    });
  } else if (response.type === 'error') {
    console.error('Error:', response.error);
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};
```

**Error Responses:**
- `{ type: "error", error: "Invalid JSON input" }`: Invalid JSON message
- `{ type: "error", error: "Invalid request type. Must be { type: 'frame', image: ... }" }`: Wrong message type
- `{ type: "error", error: "Missing required field: image (base64)" }`: Missing image
- `{ type: "error", error: "..." }`: Other processing errors

---

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request data (missing fields, invalid image, etc.)
- `500 Internal Server Error`: Server processing error

### Error Response Format

```json
{
  "error": "Error message description",
  "type": "ValueError"  // or "FileNotFoundError", etc.
}
```

### Common Errors

1. **Missing Image:**
   ```json
   {
     "error": "An image (file or base64) must be provided.",
     "type": "ValueError"
   }
   ```

2. **Invalid Image Format:**
   ```json
   {
     "error": "Invalid base64 image.",
     "type": "ValueError"
   }
   ```

3. **Image Too Large:**
   ```json
   {
     "error": "Image dimensions too large: 3000x2000, max is (1920, 1920)",
     "type": "ValueError"
   }
   ```

4. **No Face Detected:**
   - Detection endpoints return empty results: `{ "faces": [], "count": 0 }`
   - Recognition endpoints return: `{ "visitor_id": null, "matched": false, ... }`

---

## Request/Response Formats

### Image Formats

**Supported Formats:**
- JPEG/JPG
- PNG

**Max Dimensions:** 1920x1920 pixels

**Encoding:**
- Base64 strings can include `data:image/jpeg;base64,` prefix or be plain base64
- File uploads via `multipart/form-data`

### Base64 Image Example

```javascript
// With data URL prefix (automatically stripped)
const imageWithPrefix = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...";

// Plain base64 (also accepted)
const imagePlain = "/9j/4AAQSkZJRgABAQAAAQ...";
```

### Bounding Box Format

Faces are returned as bounding boxes:
```json
[x, y, width, height]
```

Where:
- `x`: X coordinate of top-left corner
- `y`: Y coordinate of top-left corner
- `width`: Width of bounding box
- `height`: Height of bounding box

### Feature Vector Format

Face features are 512-dimensional vectors:
```json
[0.123, -0.456, 0.789, ...]  // 512 floating-point numbers
```

---

## Configuration

### Environment Variables

The API can be configured using environment variables:

**Database Configuration:**
- `USE_DATABASE`: Enable PostgreSQL database (default: `false`)
- `DATABASE_URL`: PostgreSQL connection string
- `DB_TABLE_NAME`: Table name (default: `visitors`)
- `DB_VISITOR_ID_COLUMN`: ID column name (default: `id`)
- `DB_IMAGE_COLUMN`: Image column name (default: `base64Image`)
- `DB_VISITOR_LIMIT`: Limit number of visitors queried (default: `0` = no limit)

**Model Configuration:**
- `MODELS_PATH`: Path to model files (default: `models`)

**CORS Configuration:**
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: `*`)

### Default Thresholds

- **Detection Score Threshold**: `0.6` (minimum confidence for face detection)
- **Recognition/Comparison Threshold**: `0.363` (minimum similarity for face match)

---

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production deployments.

---

## Authentication

Currently, no authentication is required. Consider adding API keys or OAuth for production deployments.

---

## Examples

### Complete Detection Workflow

```bash
# 1. Check API health
curl http://localhost:8000/api/v1/health

# 2. Validate image
curl -X POST http://localhost:8000/api/v1/validate-image \
  -F "image=@photo.jpg"

# 3. Detect faces
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'$(base64 -i photo.jpg)'",
    "score_threshold": 0.6
  }'

# 4. Extract features
curl -X POST http://localhost:8000/api/v1/extract-features \
  -F "image=@photo.jpg"

# 5. Recognize visitor
curl -X POST http://localhost:8000/api/v1/recognize \
  -F "image=@visitor.jpg" \
  -F "threshold=0.363"
```

### Python Example

```python
import requests
import base64

# Read image and encode
with open('photo.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Detect faces
response = requests.post(
    'http://localhost:8000/api/v1/detect',
    json={
        'image': image_b64,
        'score_threshold': 0.6
    }
)

result = response.json()
print(f"Detected {result['count']} faces")
for face in result['faces']:
    x, y, w, h = face
    print(f"Face at ({x}, {y}) with size {w}x{h}")
```

### JavaScript/TypeScript Example

```typescript
async function detectFaces(imageFile: File) {
  // Convert file to base64
  const base64 = await new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix if present
      const base64Data = result.includes(',') 
        ? result.split(',')[1] 
        : result;
      resolve(base64Data);
    };
    reader.readAsDataURL(imageFile);
  });

  // Call API
  const response = await fetch('http://localhost:8000/api/v1/detect', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64,
      score_threshold: 0.6,
    }),
  });

  const result = await response.json();
  return result;
}
```

---

## Changelog

### Version 1.0.0
- Initial API release
- Face detection with YuNet
- Face recognition with Sface
- PostgreSQL database integration
- WebSocket support for real-time processing
- 9,640+ visitors in production database

---

## Support

For issues or questions, please refer to:
- Implementation Gap Analysis: `documentation/IMPLEMENTATION_GAP_ANALYSIS.md`
- Database Setup: `database/README.md`
- ML Service README: `services/face-recognition/README.md`
