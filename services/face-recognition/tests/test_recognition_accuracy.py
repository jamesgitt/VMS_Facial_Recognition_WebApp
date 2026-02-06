#!/usr/bin/env python3
"""
Recognition Accuracy Test

Tests the accuracy and consistency of face recognition by running
multiple recognition attempts and measuring:
- Detection rate of known faces
- Average confidence scores
- Score variance/consistency

Usage:
    python tests/test_recognition_accuracy.py
    
    # Or inside Docker:
    docker exec facial_recog_api python /app/app/../tests/test_recognition_accuracy.py
"""

import sys
from pathlib import Path

# Add app directory to path (works both locally and in Docker)
sys.path.insert(0, '/app/app')
script_dir = Path(__file__).parent.resolve()
app_dir = script_dir.parent / "app"
if app_dir.exists():
    sys.path.insert(0, str(app_dir))

import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

from app.core.config import settings
from app.core.logger import get_logger
from app.db import database
from app.db.queries import get_visitor_details
from app.ml.hnsw_index import HNSWIndexManager
from app.ml import inference
from app.utils import image_loader

logger = get_logger(__name__)

# Configuration
TEST_FACE_ID = "cmkm8mfii0003l504q9h761fe"  # Your test face
YOUR_OTHER_IDS = [
    "cm9avs34z000bjy0cb9xagdok",   # James Errol laptop
    "cmkuy7sea0003l704x2tpwwjl",   # James Errol tablet 2
    "cml7lt7i4000nl50405lidluh",   # James test (NEW)
]
NUM_ITERATIONS = 1000
TOP_K = 10  # How many results to check per search


def extract_feature_from_visitor(visitor_id: str) -> Tuple[str, np.ndarray]:
    """Extract feature from a visitor's image."""
    visitor = get_visitor_details(
        visitor_id,
        settings.database.table_name,
        settings.database.visitor_id_column
    )
    if not visitor:
        raise ValueError(f"Visitor {visitor_id} not found")
    
    name = f"{visitor.get('firstName', '')} {visitor.get('lastName', '')}"
    img = image_loader.load_from_base64(visitor.get('base64Image'))
    
    if img is None:
        raise ValueError(f"Could not load image for {visitor_id}")
    
    faces = inference.detect_faces(img, return_landmarks=True)
    if faces is None or len(faces) == 0:
        raise ValueError(f"No face detected for {visitor_id}")
    
    feature = inference.extract_face_features(img, faces[0])
    if feature is None:
        raise ValueError(f"Could not extract feature for {visitor_id}")
    
    return name, feature


def run_accuracy_test():
    """Run the accuracy test."""
    print("=" * 70)
    print("FACE RECOGNITION ACCURACY TEST")
    print("=" * 70)
    
    # Initialize
    database.test_connection()
    print("[OK] Database connected")
    
    # Load HNSW index
    hnsw = HNSWIndexManager(
        dimension=128,
        recognizer_name='sface',
        index_dir=str(settings.models_path)
    )
    print(f"[OK] HNSW index loaded: {hnsw.ntotal} vectors")
    
    # Extract test face feature
    print(f"\n[1] Extracting test face: {TEST_FACE_ID}")
    test_name, test_feature = extract_feature_from_visitor(TEST_FACE_ID)
    print(f"    Name: {test_name}")
    print(f"    Feature: {test_feature.shape[0]}-dim, norm={np.linalg.norm(test_feature):.2f}")
    
    # Verify your other faces are in the index
    print(f"\n[2] Verifying your other faces are in the index:")
    your_ids_in_index = []
    for vid in YOUR_OTHER_IDS:
        if vid in hnsw.visitor_id_to_index:
            your_ids_in_index.append(vid)
            print(f"    ✓ {vid}")
        else:
            print(f"    ✗ {vid} (NOT in index)")
    
    if not your_ids_in_index:
        print("\nERROR: None of your other faces are in the index!")
        return
    
    # Run iterations
    print(f"\n[3] Running {NUM_ITERATIONS} recognition iterations...")
    print("-" * 70)
    
    # Tracking metrics
    your_face_found_count = 0
    your_face_rank_1_count = 0
    your_face_in_top_5_count = 0
    
    scores_for_your_faces: Dict[str, List[float]] = defaultdict(list)
    top_1_scores: List[float] = []
    stranger_scores: List[float] = []
    
    search_times: List[float] = []
    
    start_time = time.time()
    
    for i in range(NUM_ITERATIONS):
        # Time the search
        t0 = time.perf_counter()
        results = hnsw.search(test_feature, k=TOP_K)
        search_times.append(time.perf_counter() - t0)
        
        # Analyze results
        your_faces_in_results = []
        stranger_faces_in_results = []
        
        for rank, (vid, similarity, meta) in enumerate(results, 1):
            if vid in your_ids_in_index or vid == TEST_FACE_ID:
                your_faces_in_results.append((rank, vid, similarity))
                scores_for_your_faces[vid].append(similarity)
            else:
                stranger_faces_in_results.append((rank, vid, similarity))
                stranger_scores.append(similarity)
        
        # Track if your face was found
        if your_faces_in_results:
            your_face_found_count += 1
            best_rank = min(r[0] for r in your_faces_in_results)
            if best_rank == 1:
                your_face_rank_1_count += 1
            if best_rank <= 5:
                your_face_in_top_5_count += 1
        
        # Track top-1 score
        if results:
            top_1_scores.append(results[0][1])
        
        # Progress update
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"    {i + 1:>5}/{NUM_ITERATIONS} | {rate:.1f} iter/s | "
                  f"Your face rank-1: {your_face_rank_1_count}/{i+1} ({100*your_face_rank_1_count/(i+1):.1f}%)")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n[A] DETECTION ACCURACY (out of {NUM_ITERATIONS} iterations):")
    print(f"    Your face found in top-{TOP_K}:  {your_face_found_count:>5} ({100*your_face_found_count/NUM_ITERATIONS:.2f}%)")
    print(f"    Your face at rank #1:           {your_face_rank_1_count:>5} ({100*your_face_rank_1_count/NUM_ITERATIONS:.2f}%)")
    print(f"    Your face in top-5:             {your_face_in_top_5_count:>5} ({100*your_face_in_top_5_count/NUM_ITERATIONS:.2f}%)")
    
    print(f"\n[B] CONFIDENCE SCORES FOR YOUR FACES:")
    for vid, scores in scores_for_your_faces.items():
        if scores:
            arr = np.array(scores)
            # Get name for this ID
            if vid == TEST_FACE_ID:
                label = f"TEST FACE ({vid[:20]}...)"
            else:
                label = f"{vid[:20]}..."
            print(f"    {label}:")
            print(f"        Mean: {arr.mean():.4f} | Std: {arr.std():.4f} | Min: {arr.min():.4f} | Max: {arr.max():.4f}")
    
    # Overall your-face scores
    all_your_scores = []
    for scores in scores_for_your_faces.values():
        all_your_scores.extend(scores)
    
    if all_your_scores:
        arr = np.array(all_your_scores)
        print(f"\n    OVERALL YOUR FACES:")
        print(f"        Mean: {arr.mean():.4f} | Std: {arr.std():.4f} | Min: {arr.min():.4f} | Max: {arr.max():.4f}")
    
    print(f"\n[C] STRANGER SCORES (faces that are NOT you):")
    if stranger_scores:
        arr = np.array(stranger_scores)
        print(f"    Mean: {arr.mean():.4f} | Std: {arr.std():.4f} | Min: {arr.min():.4f} | Max: {arr.max():.4f}")
    else:
        print(f"    No strangers appeared in top-{TOP_K} results!")
    
    print(f"\n[D] SEPARATION (your faces vs strangers):")
    if all_your_scores and stranger_scores:
        your_mean = np.mean(all_your_scores)
        stranger_mean = np.mean(stranger_scores)
        gap = your_mean - stranger_mean
        print(f"    Your face mean:     {your_mean:.4f}")
        print(f"    Stranger mean:      {stranger_mean:.4f}")
        print(f"    Gap (separation):   {gap:.4f}")
        if gap > 0.2:
            print(f"    ✓ EXCELLENT separation - recognition is reliable")
        elif gap > 0.1:
            print(f"    ~ GOOD separation - may need threshold tuning")
        else:
            print(f"    ✗ POOR separation - accuracy issues")
    
    print(f"\n[E] PERFORMANCE:")
    search_arr = np.array(search_times) * 1000  # Convert to ms
    print(f"    Search time: {search_arr.mean():.2f}ms avg | {search_arr.std():.2f}ms std | {search_arr.max():.2f}ms max")
    print(f"    Total time:  {total_time:.1f}s for {NUM_ITERATIONS} iterations")
    print(f"    Throughput:  {NUM_ITERATIONS/total_time:.1f} searches/sec")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_accuracy_test()
