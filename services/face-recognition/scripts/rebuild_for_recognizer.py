#!/usr/bin/env python
"""
Rebuild HNSW Index for Current Recognizer

This script re-extracts face features using the currently configured recognizer
(SFace or ArcFace) and rebuilds the HNSW index.

Use this script when:
- Switching from SFace to ArcFace (or vice versa)
- Index files are corrupted or missing
- Features need to be re-extracted

Usage:
    # Set recognizer in .env first, then run:
    python scripts/rebuild_for_recognizer.py
    
    # Or specify recognizer:
    python scripts/rebuild_for_recognizer.py --recognizer arcface
    python scripts/rebuild_for_recognizer.py --recognizer sface
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add app to path
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

# Load environment
from dotenv import load_dotenv
load_dotenv(APP_DIR.parent / ".env")

import numpy as np

from core.logger import get_logger
from core.config import settings
from ml.recognizer_factory import get_recognizer, reset_recognizer
from ml.index_factory import get_index_for_recognizer, reset_index
from ml import inference
from db import database
from utils import image_loader

logger = get_logger("rebuild_index")


def extract_feature_for_visitor(visitor: dict, recognizer) -> tuple:
    """
    Extract face feature for a single visitor.
    
    Returns:
        (visitor_id, feature, metadata) or (visitor_id, None, None) on failure
    """
    visitor_id = str(visitor.get('id', 'unknown'))
    base64_image = visitor.get('base64Image')
    
    if not base64_image:
        return visitor_id, None, {'error': 'no_image'}
    
    try:
        # Load image
        image = image_loader.load_from_base64(base64_image)
        if image is None:
            return visitor_id, None, {'error': 'invalid_image'}
        
        # Detect face with landmarks
        faces = inference.detect_faces(image, return_landmarks=True)
        if faces is None or len(faces) == 0:
            return visitor_id, None, {'error': 'no_face'}
        
        # Extract feature using recognizer
        feature = recognizer.extract_features(image, np.asarray(faces[0]))
        if feature is None:
            return visitor_id, None, {'error': 'extraction_failed'}
        
        # Prepare metadata
        metadata = {
            'firstName': visitor.get('firstName', ''),
            'lastName': visitor.get('lastName', ''),
        }
        
        return visitor_id, feature, metadata
        
    except Exception as e:
        return visitor_id, None, {'error': str(e)}


def rebuild_index(recognizer_type: str = None, batch_size: int = 100, workers: int = 4):
    """
    Rebuild HNSW index for specified recognizer.
    
    Args:
        recognizer_type: 'sface' or 'arcface' (uses config default if None)
        batch_size: Number of visitors to process before saving
        workers: Number of parallel workers for feature extraction
    """
    start_time = time.time()
    
    # Override recognizer type if specified
    if recognizer_type:
        os.environ['RECOGNIZER_TYPE'] = recognizer_type
        reset_recognizer()
    
    # Get recognizer
    recognizer = get_recognizer()
    logger.info(f"Using recognizer: {recognizer.name} ({recognizer.feature_dim}-dim)")
    
    # Get or create index for this recognizer
    index = get_index_for_recognizer(recognizer.name, recognizer.feature_dim)
    
    # Clear existing index
    logger.info(f"Clearing existing index for {recognizer.name}...")
    index.clear_and_delete()
    
    # Get all visitors from database
    logger.info("Loading visitors from database...")
    try:
        visitors = database.get_visitor_images_from_db()
    except Exception as e:
        logger.error(f"Failed to load visitors: {e}")
        return False
    
    if not visitors:
        logger.warning("No visitors found in database")
        return True
    
    logger.info(f"Processing {len(visitors)} visitors...")
    
    # Process visitors
    success_count = 0
    fail_count = 0
    batch_data = []
    errors_by_type = {}
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_feature_for_visitor, v, recognizer): v 
            for v in visitors
        }
        
        for i, future in enumerate(as_completed(futures)):
            visitor_id, feature, metadata = future.result()
            
            if feature is not None:
                batch_data.append((visitor_id, feature, metadata))
                success_count += 1
            else:
                fail_count += 1
                error_type = metadata.get('error', 'unknown') if metadata else 'unknown'
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            
            # Add batch to index
            if len(batch_data) >= batch_size:
                index.add_visitors_batch(batch_data)
                batch_data = []
                logger.info(f"Progress: {i+1}/{len(visitors)} ({success_count} success, {fail_count} failed)")
    
    # Add remaining batch
    if batch_data:
        index.add_visitors_batch(batch_data)
    
    # Save index
    index.save()
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Rebuild complete for {recognizer.name}")
    logger.info(f"  Total visitors: {len(visitors)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info(f"  Time: {elapsed:.1f}s ({elapsed/len(visitors)*1000:.1f}ms per visitor)")
    logger.info(f"  Index stats: {index.get_stats()}")
    
    if errors_by_type:
        logger.info("  Errors by type:")
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            logger.info(f"    {error_type}: {count}")
    
    logger.info("=" * 60)
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild HNSW index for face recognition'
    )
    parser.add_argument(
        '--recognizer', '-r',
        choices=['sface', 'arcface'],
        default=None,
        help='Recognizer type (uses RECOGNIZER_TYPE from .env if not specified)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    success = rebuild_index(
        recognizer_type=args.recognizer,
        batch_size=args.batch_size,
        workers=args.workers
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
