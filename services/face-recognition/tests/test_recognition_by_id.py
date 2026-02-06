#!/usr/bin/env python3
"""
Test Recognition by Visitor ID

Fetches a visitor by ID from the database, extracts their feature,
and finds the top matching visitors in the HNSW index.

Usage:
    python scripts/test_recognition_by_id.py <visitor_id> [--top-k 10]
"""

import sys
import argparse
from pathlib import Path

# Add app directory to path
script_dir = Path(__file__).parent.resolve()
app_dir = script_dir.parent / "app"
sys.path.insert(0, str(app_dir))

import numpy as np

from core.config import settings
from core.logger import get_logger
from db import database
from db.queries import get_visitor_details
from pipelines.feature_extraction import decode_feature_from_base64, extract_single_feature
from ml.index_factory import get_index
from ml.recognizer_factory import get_recognizer
from ml import inference
from utils import image_loader

logger = get_logger(__name__)


def get_visitor_by_id(visitor_id: str) -> dict | None:
    """Fetch a single visitor by ID."""
    db_config = settings.database
    
    # Initialize database connection
    if not database.test_connection():
        print("ERROR: Could not connect to database")
        return None
    
    visitor = get_visitor_details(
        visitor_id=visitor_id,
        table_name=db_config.table_name,
        visitor_id_column=db_config.visitor_id_column
    )
    
    return visitor


def get_feature_for_visitor(visitor: dict) -> np.ndarray | None:
    """Extract or decode feature vector for a visitor."""
    db_config = settings.database
    
    # Try stored features first
    stored_features = visitor.get(db_config.features_column) or visitor.get('facefeatures')
    if stored_features:
        feature = decode_feature_from_base64(stored_features)
        if feature is not None:
            print(f"  Using stored feature vector ({feature.shape[0]}-dim)")
            return feature
    
    # Extract from image
    base64_data = visitor.get(db_config.image_column) or visitor.get('base64Image')
    if not base64_data:
        print("  ERROR: No image data found")
        return None
    
    print("  Extracting feature from image...")
    img_cv = image_loader.load_from_base64(base64_data)
    feature = extract_single_feature(img_cv)
    
    if feature is not None:
        print(f"  Extracted feature vector ({feature.shape[0]}-dim)")
    
    return feature


def search_similar_visitors(query_feature: np.ndarray, top_k: int = 20) -> list:
    """Search HNSW index for similar visitors."""
    hnsw_manager = get_index()
    
    if hnsw_manager is None or hnsw_manager.ntotal == 0:
        print("ERROR: HNSW index not available or empty")
        return []
    
    print(f"\nSearching HNSW index ({hnsw_manager.ntotal} vectors)...")
    
    # Get more candidates for re-ranking
    results = hnsw_manager.search(query_feature, k=top_k * 2)
    
    # Re-rank using precise comparison
    reranked = []
    recognizer = get_recognizer()
    
    for visitor_id, approx_sim, metadata in results[:top_k]:
        try:
            idx = hnsw_manager.visitor_id_to_index.get(visitor_id)
            if idx is not None and hnsw_manager.index is not None:
                stored_feature = hnsw_manager.index.get_items([idx])[0]
                stored_feature = np.array(stored_feature, dtype=np.float32)
                
                # Precise comparison
                precise_score = recognizer.compare(query_feature, stored_feature)
                reranked.append({
                    'visitor_id': visitor_id,
                    'approx_score': approx_sim,
                    'precise_score': float(precise_score),
                    'firstName': metadata.get('firstName', ''),
                    'lastName': metadata.get('lastName', ''),
                })
        except Exception as e:
            logger.debug(f"Re-rank failed for {visitor_id}: {e}")
            reranked.append({
                'visitor_id': visitor_id,
                'approx_score': approx_sim,
                'precise_score': approx_sim,
                'firstName': metadata.get('firstName', ''),
                'lastName': metadata.get('lastName', ''),
            })
    
    # Sort by precise score
    reranked.sort(key=lambda x: x['precise_score'], reverse=True)
    
    return reranked


def main():
    parser = argparse.ArgumentParser(description="Test recognition by visitor ID")
    parser.add_argument("visitor_id", help="Visitor ID to test")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top matches to return")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold")
    args = parser.parse_args()
    
    visitor_id = args.visitor_id
    top_k = args.top_k
    threshold = args.threshold or settings.models.sface_similarity_threshold
    
    print("=" * 60)
    print("Test Recognition by Visitor ID")
    print("=" * 60)
    print(f"\nVisitor ID: {visitor_id}")
    print(f"Top K: {top_k}")
    print(f"Threshold: {threshold}")
    
    # Get recognizer info
    recognizer = get_recognizer()
    print(f"Recognizer: {recognizer.name} ({recognizer.feature_dim}-dim)")
    
    # Step 1: Fetch visitor
    print(f"\n[1/3] Fetching visitor from database...")
    visitor = get_visitor_by_id(visitor_id)
    
    if visitor is None:
        print(f"ERROR: Visitor '{visitor_id}' not found in database")
        return 1
    
    first_name = visitor.get('firstName', 'Unknown')
    last_name = visitor.get('lastName', 'Unknown')
    print(f"  Found: {first_name} {last_name}")
    
    # Step 2: Get feature vector
    print(f"\n[2/3] Getting feature vector...")
    query_feature = get_feature_for_visitor(visitor)
    
    if query_feature is None:
        print("ERROR: Could not get feature vector for visitor")
        return 1
    
    # Step 3: Search for similar visitors
    print(f"\n[3/3] Searching for similar visitors...")
    results = search_similar_visitors(query_feature, top_k=top_k)
    
    if not results:
        print("No results found")
        return 1
    
    # Display results
    print("\n" + "=" * 80)
    print(f"{'Rank':<5} {'Visitor ID':<30} {'Name':<25} {'Score':>10} {'Match':>8}")
    print("=" * 80)
    
    for i, match in enumerate(results, 1):
        name = f"{match['firstName']} {match['lastName']}".strip() or "Unknown"
        score = match['precise_score']
        is_match = "YES" if score >= threshold else "no"
        is_self = " (SELF)" if match['visitor_id'] == visitor_id else ""
        
        print(f"{i:<5} {match['visitor_id']:<30} {name:<25} {score:>10.4f} {is_match:>8}{is_self}")
    
    print("=" * 80)
    
    # Summary
    matches_above_threshold = sum(1 for m in results if m['precise_score'] >= threshold)
    print(f"\nSummary:")
    print(f"  - Total results: {len(results)}")
    print(f"  - Matches above threshold ({threshold}): {matches_above_threshold}")
    
    # Find self-match
    self_matches = [m for m in results if m['visitor_id'] == visitor_id]
    if self_matches:
        self_rank = next(i for i, m in enumerate(results, 1) if m['visitor_id'] == visitor_id)
        print(f"  - Self-match rank: #{self_rank} (score: {self_matches[0]['precise_score']:.4f})")
    else:
        print(f"  - Self-match: NOT FOUND in top {top_k}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
