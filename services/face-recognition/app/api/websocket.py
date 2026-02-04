"""
WebSocket Routes

Real-time face detection via WebSocket with API key authentication.
"""

import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from core.logger import get_logger
from core.config import settings

from pipelines import detect_faces_in_image
from utils import image_loader

logger = get_logger(__name__)

router = APIRouter()


async def authenticate_websocket(
    websocket: WebSocket,
    api_key: Optional[str] = None,
) -> bool:
    """
    Authenticate WebSocket connection using API key.
    
    Args:
        websocket: WebSocket connection
        api_key: API key from query parameter
    
    Returns:
        True if authenticated, False otherwise
    """
    # If auth is disabled, allow all connections
    if not settings.auth.is_enabled:
        return True
    
    # Check API key from query parameter
    if api_key and api_key == settings.auth.api_key:
        return True
    
    return False


@router.websocket("/ws/realtime")
async def websocket_face_endpoint(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None, alias="api_key"),
):
    """
    WebSocket endpoint for real-time face detection with authentication.
    
    Authentication:
    - Pass API key as query parameter: /ws/realtime?api_key=your-key
    - Or send auth message first: {"type": "auth", "api_key": "your-key"}
    - If AUTH_ENABLED=false, no authentication required
    
    Expected message format:
    {
        "type": "frame",
        "image": "<base64_image>",
        "score_threshold": 0.7,  // optional
        "return_landmarks": false  // optional
    }
    
    Response format:
    {
        "type": "results",
        "faces": [
            {
                "bbox": [x, y, width, height],
                "confidence": 0.95,
                "landmarks": [...]  // if requested
            }
        ],
        "count": 1
    }
    
    Error format:
    {
        "type": "error",
        "error": "Error message"
    }
    
    Auth success response:
    {
        "type": "auth_success",
        "message": "Authenticated successfully"
    }
    """
    # Check authentication via query parameter first
    is_authenticated = await authenticate_websocket(websocket, api_key)
    
    # Accept connection (we need to accept before we can send/receive)
    await websocket.accept()
    
    # If not authenticated via query param and auth is enabled,
    # wait for auth message
    if not is_authenticated and settings.auth.is_enabled:
        logger.info("WebSocket awaiting authentication...")
        try:
            # Wait for auth message (with timeout handled by client)
            auth_data = await websocket.receive_text()
            auth_req = json.loads(auth_data)
            
            if auth_req.get("type") == "auth":
                provided_key = auth_req.get("api_key")
                if provided_key and provided_key == settings.auth.api_key:
                    is_authenticated = True
                    await websocket.send_json({
                        "type": "auth_success",
                        "message": "Authenticated successfully"
                    })
                    logger.info("WebSocket authenticated via message")
                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid API key"
                    })
                    logger.warning("WebSocket authentication failed: invalid API key")
                    await websocket.close(code=4001, reason="Invalid API key")
                    return
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": "Authentication required. Send: {\"type\": \"auth\", \"api_key\": \"your-key\"}"
                })
                await websocket.close(code=4001, reason="Authentication required")
                return
                
        except json.JSONDecodeError:
            await websocket.send_json({
                "type": "error",
                "error": "Invalid JSON. Authentication required."
            })
            await websocket.close(code=4001, reason="Invalid auth message")
            return
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during authentication")
            return
    
    logger.info("WebSocket connection authenticated and accepted")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Parse JSON
            try:
                req = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON"
                })
                continue
            
            # Validate request type
            if req.get("type") != "frame":
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid request type. Expected 'frame'."
                })
                continue
            
            # Get image data
            image_b64 = req.get("image")
            if not image_b64:
                await websocket.send_json({
                    "type": "error",
                    "error": "Missing 'image' field"
                })
                continue
            
            # Process frame
            try:
                # Load image
                img_np = image_loader.load_image(image_b64, source_type="base64")
                image_loader.validate_image_size(
                    (img_np.shape[1], img_np.shape[0]),
                    settings.image.max_size
                )
                
                # Get detection parameters
                score_threshold = float(req.get(
                    "score_threshold",
                    settings.models.yunet_score_threshold
                ))
                return_landmarks = bool(req.get("return_landmarks", False))
                
                # Detect faces using pipeline
                result = detect_faces_in_image(
                    img_np,
                    score_threshold=score_threshold,
                    return_landmarks=return_landmarks,
                )
                
                if not result.success:
                    await websocket.send_json({
                        "type": "error",
                        "error": result.error or "Detection failed"
                    })
                    continue
                
                # Build response
                faces_list: list[dict] = []
                for face in result.faces:
                    face_obj: dict = {
                        "bbox": list(face.bbox)
                    }
                    if face.confidence is not None:
                        face_obj["confidence"] = face.confidence
                    if return_landmarks and face.landmarks:
                        face_obj["landmarks"] = face.landmarks
                    faces_list.append(face_obj)
                
                await websocket.send_json({
                    "type": "results",
                    "faces": faces_list,
                    "count": len(faces_list)
                })
                
            except Exception as ex:
                logger.warning(f"WebSocket frame error: {ex}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(ex)
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except Exception:
            pass


__all__ = ["router"]
