"""
API Routes

HTTP endpoints for face detection, recognition, and comparison.
"""

import io
import base64
import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from PIL import Image

from core.logger import get_logger
from core.config import settings
from core.state import app_state
from api.deps import verify_api_key

from schemas import (
    # Detection
    DetectRequest,
    DetectionResponse,
    FaceDetection,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    # Recognition
    RecognizeRequest,
    VisitorMatch,
    VisitorRecognitionResponse,
    # Comparison
    CompareRequest,
    CompareResponse,
    # Common
    HealthResponse,
    ModelStatusResponse,
    ModelInfo,
    ModelInfoResponse,
    HNSWStatusResponse,
    ValidateImageRequest,
    ValidateImageResponse,
    # HNSW Management
    HNSWAddVisitorRequest,
    HNSWAddVisitorFeatureRequest,
    HNSWAddVisitorResponse,
    HNSWRebuildRequest,
    HNSWRebuildResponse,
    HNSWSyncRequest,
    HNSWSyncResponse,
)

from pipelines import (
    detect_faces_in_image,
    extract_features_from_image,
    compare_from_base64,
    recognize_from_image,
)

from utils import image_loader
from ml import inference

logger = get_logger(__name__)

router = APIRouter()


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
@router.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint with recognizer and index info."""
    from ml.recognizer_factory import get_recognizer, is_fallback_active
    
    # Get recognizer info
    recognizer_name = None
    feature_dim = None
    is_fallback = False
    try:
        recognizer = get_recognizer()
        recognizer_name = recognizer.name
        feature_dim = recognizer.feature_dim
        is_fallback = is_fallback_active()
    except Exception:
        pass
    
    # Get index size
    index_size = None
    if app_state.hnsw_manager:
        try:
            index_size = app_state.hnsw_manager.ntotal
        except Exception:
            pass
    
    return HealthResponse(
        status="ok",
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        recognizer=recognizer_name,
        feature_dim=feature_dim,
        index_size=index_size,
        is_fallback=is_fallback
    )


# =============================================================================
# DETECTION ENDPOINTS
# =============================================================================

@router.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_faces_api(request: DetectRequest, _: str = Depends(verify_api_key)):
    """
    Detect faces in an image.
    
    Returns bounding boxes and optionally landmarks for all detected faces.
    """
    try:
        # Load and validate image
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        # Detect faces using pipeline
        result = detect_faces_in_image(
            img_np,
            score_threshold=request.score_threshold,
            return_landmarks=request.return_landmarks,
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert to response format
        faces_list = result.to_response_format(include_landmarks=request.return_landmarks)
        
        return DetectionResponse(faces=faces_list, count=result.count)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/api/v1/detect-structured", tags=["Detection"])
async def detect_faces_structured_api(request: DetectRequest, _: str = Depends(verify_api_key)):
    """
    Detect faces in an image with structured response.
    
    Returns FaceDetection objects with bbox, confidence, and landmarks.
    """
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        result = detect_faces_in_image(
            img_np,
            score_threshold=request.score_threshold,
            return_landmarks=request.return_landmarks,
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert to structured FaceDetection objects
        faces = [
            FaceDetection(
                bbox=list(face.bbox),
                confidence=face.confidence,
                landmarks=face.landmarks,
            )
            for face in result.faces
        ]
        
        return {"faces": [f.model_dump() for f in faces], "count": len(faces)}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


# =============================================================================
# FEATURE EXTRACTION ENDPOINTS
# =============================================================================

@router.post("/api/v1/extract-features", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    _: str = Depends(verify_api_key),
):
    """
    Extract face feature vectors from an image.
    
    Accepts either file upload or base64 encoded image.
    Returns 128-dimensional feature vectors for all detected faces.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    try:
        # Load image
        if image is not None:
            img_np = image_loader.load_from_upload(image)
        else:
            img_np = image_loader.load_image(image_base64, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        # Extract features using pipeline
        result = extract_features_from_image(img_np)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Convert numpy arrays to lists
        features_list = [
            feat.tolist() if hasattr(feat, 'tolist') else list(feat)
            for feat in result.features
        ]
        
        return FeatureExtractionResponse(
            features=features_list,
            num_faces=result.num_faces
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/api/v1/extract-features-json", response_model=FeatureExtractionResponse, tags=["Features"])
async def extract_features_json_api(request: FeatureExtractionRequest, _: str = Depends(verify_api_key)):
    """
    Extract face feature vectors from an image (JSON body).
    
    Accepts base64-encoded image in JSON request body.
    Returns 128-dimensional feature vectors for all detected faces.
    """
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        result = extract_features_from_image(img_np)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        features_list = [
            feat.tolist() if hasattr(feat, 'tolist') else list(feat)
            for feat in result.features
        ]
        
        return FeatureExtractionResponse(
            features=features_list,
            num_faces=result.num_faces
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


# =============================================================================
# COMPARISON ENDPOINTS
# =============================================================================

@router.post("/api/v1/compare", response_model=CompareResponse, tags=["Recognition"])
async def compare_faces_api(request: CompareRequest, _: str = Depends(verify_api_key)):
    """
    Compare faces between two images.
    
    Returns similarity score and match status based on threshold.
    """
    try:
        # Use comparison pipeline
        result = compare_from_base64(
            request.image1,
            request.image2,
            threshold=request.threshold,
            return_features=False,
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return CompareResponse(
            similarity_score=result.similarity_score,
            is_match=result.is_match,
            threshold=result.threshold,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


# =============================================================================
# RECOGNITION ENDPOINTS
# =============================================================================

@router.post("/api/v1/recognize", response_model=VisitorRecognitionResponse, tags=["Recognition"])
async def recognize_visitor_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    threshold: float = Form(None),
    _: str = Depends(verify_api_key),
):
    """
    Recognize a visitor by matching against the database.
    
    Uses HNSW index for fast approximate search, with linear search fallback.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    threshold = threshold or settings.models.sface_similarity_threshold
    
    try:
        # Load image
        if image is not None:
            img_np = image_loader.load_from_upload(image)
        else:
            img_np = image_loader.load_image(image_base64, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")
    
    # Use recognition pipeline
    result = recognize_from_image(img_np, threshold=threshold, top_k=50)
    
    # Convert to response format using VisitorMatch schema
    matches = [
        VisitorMatch(
            visitor_id=m.visitor_id,
            match_score=m.match_score,
            is_match=m.is_match,
            firstName=m.firstName,
            lastName=m.lastName,
        )
        for m in result.matches[:10]
    ]
    
    return VisitorRecognitionResponse(
        matched=result.matched,
        visitor_id=result.best_match.visitor_id if result.best_match else None,
        confidence=result.best_match.match_score if result.best_match else None,
        firstName=result.best_match.firstName if result.best_match else None,
        lastName=result.best_match.lastName if result.best_match else None,
        matches=[m.model_dump() for m in matches],
    )


@router.post("/api/v1/recognize-json", response_model=VisitorRecognitionResponse, tags=["Recognition"])
async def recognize_visitor_json_api(request: RecognizeRequest, _: str = Depends(verify_api_key)):
    """
    Recognize a visitor by matching against the database (JSON body).
    
    Accepts base64-encoded image in JSON request body.
    Uses HNSW index for fast approximate search, with linear search fallback.
    """
    try:
        img_np = image_loader.load_image(request.image, source_type="base64")
        
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")
    
    # Use recognition pipeline with request parameters
    result = recognize_from_image(
        img_np,
        threshold=request.threshold,
        top_k=request.top_k
    )
    
    # Convert to response format using VisitorMatch schema
    matches = [
        VisitorMatch(
            visitor_id=m.visitor_id,
            match_score=m.match_score,
            is_match=m.is_match,
            firstName=m.firstName,
            lastName=m.lastName,
        )
        for m in result.matches[:request.top_k]
    ]
    
    return VisitorRecognitionResponse(
        matched=result.matched,
        visitor_id=result.best_match.visitor_id if result.best_match else None,
        confidence=result.best_match.match_score if result.best_match else None,
        firstName=result.best_match.firstName if result.best_match else None,
        lastName=result.best_match.lastName if result.best_match else None,
        matches=[m.model_dump() for m in matches],
    )


# =============================================================================
# MODEL STATUS ENDPOINTS
# =============================================================================

@router.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def model_status():
    """Get ML model loading status."""
    loaded = app_state.models_loaded
    
    return ModelStatusResponse(
        loaded=loaded,
        details={
            "face_detector": str(type(app_state.face_detector)),
            "face_recognizer": str(type(app_state.face_recognizer))
        } if loaded else None
    )


@router.get("/models/info", response_model=ModelInfoResponse, tags=["Models"])
async def model_info():
    """Get ML model metadata."""
    return ModelInfoResponse(
        detector=ModelInfo(
            type="YuNet",
            model_path=str(inference.YUNET_PATH),
            input_size=list(inference.YUNET_INPUT_SIZE),
            loaded=app_state.face_detector is not None,
        ),
        recognizer=ModelInfo(
            type="SFace",
            model_path=str(inference.SFACE_PATH),
            similarity_threshold=inference.SFACE_SIMILARITY_THRESHOLD,
            loaded=app_state.face_recognizer is not None,
        ),
    )


# =============================================================================
# DEBUG TEST ENDPOINT
# =============================================================================

@router.get("/api/v1/test-code-version")
async def test_code_version():
    """Test if code changes are loaded."""
    from ml.recognizer_factory import get_recognizer
    try:
        rec = get_recognizer()
        return {"code_version": "v4", "recognizer": rec.name, "dim": rec.feature_dim}
    except Exception as e:
        return {"code_version": "v4", "error": str(e)}


# =============================================================================
# HNSW INDEX ENDPOINTS
# =============================================================================

@router.get("/api/v1/hnsw/status", response_model=HNSWStatusResponse, tags=["HNSW"])
async def hnsw_status():
    """Get HNSW index status and statistics."""
    from ml.recognizer_factory import get_recognizer
    from ml.index_factory import get_index
    
    # Use index factory directly instead of app_state
    try:
        hnsw_manager = get_index()
    except Exception as e:
        logger.error(f"Failed to get index: {e}")
        hnsw_manager = app_state.hnsw_manager
    
    # Get current recognizer info
    try:
        recognizer = get_recognizer()
        recognizer_name = recognizer.name
        recognizer_dim = recognizer.feature_dim
        logger.info(f"HNSW status: recognizer={recognizer_name}, dim={recognizer_dim}")
    except Exception as e:
        logger.error(f"Failed to get recognizer: {e}")
        recognizer_name = "Unknown"
        recognizer_dim = 128
    
    if hnsw_manager is None:
        return HNSWStatusResponse(
            available=False,
            initialized=False,
            total_vectors=0,
            dimension=recognizer_dim,
            recognizer_name=recognizer_name,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": "HNSW not initialized"}
        )
    
    try:
        stats = hnsw_manager.get_stats()
        # Add debug info to details
        debug_info = {
            **stats,
            '_debug_recognizer': recognizer_name,
            '_debug_dim': recognizer_dim,
            '_debug_manager_dim': hnsw_manager.dimension if hnsw_manager else None,
            '_code_version': 'v2'  # Marker to confirm this code is running
        }
        return HNSWStatusResponse(
            available=True,
            initialized=True,
            total_vectors=stats.get('total_vectors', 0),
            dimension=stats.get('dimension', recognizer_dim),
            recognizer_name=stats.get('recognizer_name', recognizer_name),
            index_type=stats.get('index_type', 'HNSW'),
            m=stats.get('m'),
            ef_construction=stats.get('ef_construction'),
            ef_search=stats.get('ef_search'),
            visitors_indexed=stats.get('visitors_indexed', 0),
            details=debug_info
        )
    except Exception as e:
        return HNSWStatusResponse(
            available=True,
            initialized=False,
            total_vectors=0,
            dimension=recognizer_dim,
            recognizer_name=recognizer_name,
            index_type="HNSW",
            visitors_indexed=0,
            details={"error": str(e)}
        )


@router.post("/api/v1/hnsw/add-visitor", response_model=HNSWAddVisitorResponse, tags=["HNSW"])
async def hnsw_add_visitor(request: HNSWAddVisitorRequest, _: str = Depends(verify_api_key)):
    """
    Add a single visitor to the HNSW index.
    
    Extracts face features from the provided image and adds to the index.
    Call this endpoint after creating a new visitor in your database.
    """
    import numpy as np
    from ml.index_factory import get_index
    from ml import inference
    
    try:
        # Load and validate image
        img_np = image_loader.load_image(request.image, source_type="base64")
        image_loader.validate_image_size(
            (img_np.shape[1], img_np.shape[0]),
            settings.image.max_size
        )
        
        # Detect face
        faces = inference.detect_faces(img_np, return_landmarks=True)
        if faces is None or len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Extract features
        feature = inference.extract_face_features(img_np, np.asarray(faces[0]))
        if feature is None:
            raise HTTPException(status_code=500, detail="Failed to extract face features")
        
        feature = np.asarray(feature).flatten().astype(np.float32)
        
        # Get HNSW manager
        hnsw_manager = get_index()
        if hnsw_manager is None:
            raise HTTPException(status_code=500, detail="HNSW index not initialized")
        
        # Check if visitor already exists
        if request.visitor_id in hnsw_manager.visitor_id_to_index:
            return HNSWAddVisitorResponse(
                success=False,
                visitor_id=request.visitor_id,
                message="Visitor already exists in index",
                index_size=hnsw_manager.ntotal
            )
        
        # Add to index
        metadata = {
            "firstName": request.first_name,
            "lastName": request.last_name,
        }
        
        success = hnsw_manager.add_visitor(request.visitor_id, feature, metadata)
        
        if success:
            hnsw_manager.save()
            logger.info(f"Added visitor {request.visitor_id} to HNSW index")
        
        return HNSWAddVisitorResponse(
            success=success,
            visitor_id=request.visitor_id,
            message="Visitor added to HNSW index" if success else "Failed to add visitor",
            index_size=hnsw_manager.ntotal
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    except Exception as e:
        logger.error(f"Error adding visitor to HNSW: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/api/v1/hnsw/add-visitor-feature", response_model=HNSWAddVisitorResponse, tags=["HNSW"])
async def hnsw_add_visitor_feature(request: HNSWAddVisitorFeatureRequest, _: str = Depends(verify_api_key)):
    """
    Add a visitor with pre-extracted feature vector.
    
    Use this when you already have the feature vector (e.g., stored in database).
    """
    import numpy as np
    from ml.index_factory import get_index
    
    try:
        # Get HNSW manager
        hnsw_manager = get_index()
        if hnsw_manager is None:
            raise HTTPException(status_code=500, detail="HNSW index not initialized")
        
        # Validate feature dimension
        feature = np.asarray(request.feature).flatten().astype(np.float32)
        if feature.shape[0] != hnsw_manager.dimension:
            raise HTTPException(
                status_code=400, 
                detail=f"Feature dimension mismatch: got {feature.shape[0]}, expected {hnsw_manager.dimension}"
            )
        
        # Check if visitor already exists
        if request.visitor_id in hnsw_manager.visitor_id_to_index:
            return HNSWAddVisitorResponse(
                success=False,
                visitor_id=request.visitor_id,
                message="Visitor already exists in index",
                index_size=hnsw_manager.ntotal
            )
        
        # Add to index
        metadata = {
            "firstName": request.first_name,
            "lastName": request.last_name,
        }
        
        success = hnsw_manager.add_visitor(request.visitor_id, feature, metadata)
        
        if success:
            hnsw_manager.save()
            logger.info(f"Added visitor {request.visitor_id} to HNSW index (from feature)")
        
        return HNSWAddVisitorResponse(
            success=success,
            visitor_id=request.visitor_id,
            message="Visitor added to HNSW index" if success else "Failed to add visitor",
            index_size=hnsw_manager.ntotal
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding visitor feature to HNSW: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@router.post("/api/v1/hnsw/rebuild", response_model=HNSWRebuildResponse, tags=["HNSW"])
async def hnsw_rebuild(request: HNSWRebuildRequest = None, _: str = Depends(verify_api_key)):
    """
    Rebuild the entire HNSW index from database.
    
    This is a heavy operation that clears the existing index and rebuilds
    it from all visitors in the database. Use sparingly.
    """
    import time
    from ml.index_factory import get_index
    from pipelines.visitor_loader import load_visitors_from_database
    
    start_time = time.time()
    
    try:
        hnsw_manager = get_index()
        if hnsw_manager is None:
            raise HTTPException(status_code=500, detail="HNSW index not initialized")
        
        # Check if rebuild is needed
        if not (request and request.force) and hnsw_manager.ntotal > 0:
            return HNSWRebuildResponse(
                success=False,
                message="Index already has data. Use force=true to rebuild.",
                visitors_indexed=hnsw_manager.ntotal,
                duration_seconds=0.0
            )
        
        # Clear existing index
        logger.info("Starting HNSW rebuild from database...")
        hnsw_manager.clear()
        
        # Reload from database
        result = load_visitors_from_database(hnsw_manager)
        
        duration = time.time() - start_time
        
        if result.success:
            logger.info(f"HNSW rebuild complete: {result.count} visitors in {duration:.2f}s")
            return HNSWRebuildResponse(
                success=True,
                message=f"HNSW index rebuilt successfully from {result.source}",
                visitors_indexed=result.count,
                duration_seconds=round(duration, 2)
            )
        else:
            return HNSWRebuildResponse(
                success=False,
                message=f"Rebuild failed: {result.error}",
                visitors_indexed=0,
                duration_seconds=round(duration, 2)
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding HNSW index: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild error: {e}")


@router.post("/api/v1/hnsw/sync", response_model=HNSWSyncResponse, tags=["HNSW"])
async def hnsw_sync(request: HNSWSyncRequest = None, _: str = Depends(verify_api_key)):
    """
    Sync new visitors from database to HNSW index (incremental update).
    
    Only adds visitors that are not already in the index.
    Much faster than full rebuild for adding new entries.
    """
    import numpy as np
    from ml.index_factory import get_index
    from db import database
    from pipelines.feature_extraction import extract_feature_from_visitor_data
    
    try:
        hnsw_manager = get_index()
        if hnsw_manager is None:
            raise HTTPException(status_code=500, detail="HNSW index not initialized")
        
        db_config = settings.database
        
        # Get visitors from database
        if request and request.visitor_ids:
            # Sync specific visitors - use efficient query by IDs
            logger.info(f"Syncing specific visitors: {request.visitor_ids}")
            visitors = database.get_visitors_by_ids(
                visitor_ids=request.visitor_ids,
                table_name=db_config.table_name,
                visitor_id_column=db_config.visitor_id_column,
                image_column=db_config.image_column,
                features_column=db_config.features_column,
            )
        else:
            # Sync all visitors not in index
            visitors = database.get_visitor_images_from_db(
                table_name=db_config.table_name,
                visitor_id_column=db_config.visitor_id_column,
                image_column=db_config.image_column,
                features_column=db_config.features_column,
                limit=db_config.visitor_limit,
            )
        
        # Filter to only new visitors (not already in index)
        existing_ids = set(hnsw_manager.visitor_id_to_index.keys())
        new_visitors = []
        skipped = 0
        
        for visitor in visitors:
            visitor_id = str(visitor.get('id', visitor.get('visitor_id', 'unknown')))
            if visitor_id in existing_ids:
                skipped += 1
            else:
                new_visitors.append(visitor)
        
        if not new_visitors:
            return HNSWSyncResponse(
                success=True,
                message="No new visitors to sync",
                visitors_added=0,
                visitors_skipped=skipped,
                index_size=hnsw_manager.ntotal
            )
        
        # Extract features and add to index
        batch_data = []
        for visitor in new_visitors:
            visitor_id = str(visitor.get('id', visitor.get('visitor_id', 'unknown')))
            try:
                feature = extract_feature_from_visitor_data(visitor)
                if feature is not None:
                    feature = np.asarray(feature).flatten().astype(np.float32)
                    if feature.shape[0] == hnsw_manager.dimension:
                        metadata = {
                            'firstName': visitor.get('firstName', ''),
                            'lastName': visitor.get('lastName', ''),
                        }
                        batch_data.append((visitor_id, feature, metadata))
            except Exception as e:
                logger.warning(f"Failed to extract feature for {visitor_id}: {e}")
        
        # Add batch to index
        added = hnsw_manager.add_visitors_batch(batch_data)
        
        if added > 0:
            hnsw_manager.save()
            logger.info(f"Synced {added} new visitors to HNSW index")
        
        return HNSWSyncResponse(
            success=True,
            message=f"Synced {added} new visitors to HNSW index",
            visitors_added=added,
            visitors_skipped=skipped,
            index_size=hnsw_manager.ntotal
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing HNSW index: {e}")
        raise HTTPException(status_code=500, detail=f"Sync error: {e}")


@router.delete("/api/v1/hnsw/visitor/{visitor_id}", tags=["HNSW"])
async def hnsw_remove_visitor(visitor_id: str, _: str = Depends(verify_api_key)):
    """
    Remove a visitor from the HNSW index.
    
    Note: HNSW doesn't support true deletion. This marks the entry as removed
    in metadata. For complete removal, rebuild the index.
    """
    from ml.index_factory import get_index
    
    try:
        hnsw_manager = get_index()
        if hnsw_manager is None:
            raise HTTPException(status_code=500, detail="HNSW index not initialized")
        
        if visitor_id not in hnsw_manager.visitor_id_to_index:
            raise HTTPException(status_code=404, detail=f"Visitor {visitor_id} not found in index")
        
        success = hnsw_manager.remove_visitor(visitor_id)
        
        if success:
            hnsw_manager.save()
            logger.info(f"Removed visitor {visitor_id} from HNSW index")
        
        return {
            "success": success,
            "visitor_id": visitor_id,
            "message": "Visitor removed from index" if success else "Failed to remove visitor",
            "index_size": hnsw_manager.ntotal
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing visitor from HNSW: {e}")
        raise HTTPException(status_code=500, detail=f"Remove error: {e}")


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

def _validate_image_data(image_data: bytes) -> ValidateImageResponse:
    """
    Internal helper to validate image bytes.
    
    Returns:
        ValidateImageResponse with validation result
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        
        fmt = (img.format or "").lower()
        size = img.size
        max_size = settings.image.max_size
        allowed = settings.image.allowed_formats
        
        valid = (
            fmt in allowed and
            size[0] <= max_size[0] and
            size[1] <= max_size[1]
        )
        
        return ValidateImageResponse(valid=valid, format=fmt, size=size)
        
    except Exception:
        return ValidateImageResponse(valid=False, format=None, size=None)


@router.post("/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_api(
    image: UploadFile = File(None),
    image_base64: str = Form(None),
    _: str = Depends(verify_api_key),
):
    """
    Validate an image before processing (multipart form).
    
    Accepts file upload or base64 form field.
    Checks format and size constraints.
    """
    if image is None and not image_base64:
        raise HTTPException(status_code=400, detail="Image required")
    
    if image is not None:
        contents = await image.read()
        await image.seek(0)
        return _validate_image_data(contents)
    else:
        # Handle data URI prefix
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',', 1)[1]
        img_bytes = base64.b64decode(image_base64)
        return _validate_image_data(img_bytes)


@router.post("/api/v1/validate-image", response_model=ValidateImageResponse, tags=["Utility"])
async def validate_image_json_api(request: ValidateImageRequest, _: str = Depends(verify_api_key)):
    """
    Validate an image before processing (JSON body).
    
    Accepts base64-encoded image in JSON request body.
    Checks format and size constraints.
    """
    image_data = request.image
    
    # Handle data URI prefix
    if image_data.startswith('data:'):
        image_data = image_data.split(',', 1)[1]
    
    img_bytes = base64.b64decode(image_data)
    return _validate_image_data(img_bytes)


__all__ = ["router"]
