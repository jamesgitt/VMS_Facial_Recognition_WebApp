#!/usr/bin/env python
"""
Cross-Image Face Recognition Accuracy Test

This is a more rigorous accuracy test that uses repeat visitors
(same person with multiple database entries) to test if DIFFERENT
photos of the same person can recognize each other.

Test methodology:
1. Find visitors who have multiple entries (by fullName)
2. For each entry, extract fresh features from the stored image
3. Search the HNSW index 
4. Check if OTHER entries of the same person appear in top matches

This tests real-world accuracy: can the system recognize someone
from a NEW photo when they were registered with a DIFFERENT photo?

Usage:
    python scripts/test_cross_image_accuracy.py
    python scripts/test_cross_image_accuracy.py --min-entries 50 --limit 500
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

# Add app to path
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

# Load environment
from dotenv import load_dotenv
load_dotenv(APP_DIR.parent / ".env.test")

import numpy as np
from psycopg2.extras import RealDictCursor

from core.logger import get_logger
from core.config import settings
from ml.recognizer_factory import get_recognizer
from ml.index_factory import get_index
from ml import inference
from db.connection import get_connection
from utils import image_loader

logger = get_logger("cross_image_test")


@dataclass
class CrossImageResult:
    """Result of cross-image recognition test for one query."""
    query_id: str  # ID of the query entry
    person_name: str  # fullName
    face_detected: bool = False
    feature_extracted: bool = False
    query_succeeded: bool = False
    
    # Did we find ANY other entry of the same person?
    same_person_found: bool = False
    same_person_in_top1: bool = False
    same_person_in_top5: bool = False
    same_person_in_top10: bool = False
    
    # Best rank where same person was found
    best_same_person_rank: int = -1
    best_same_person_score: float = 0.0
    
    # Top-1 match details
    top1_id: Optional[str] = None
    top1_score: float = 0.0
    top1_is_same_person: bool = False
    
    # How many of same person in top-10?
    same_person_count_in_top10: int = 0
    
    error: Optional[str] = None


@dataclass
class CrossImageMetrics:
    """Overall cross-image accuracy metrics."""
    recognizer: str
    threshold: float
    
    # Test parameters
    total_persons: int  # Number of unique persons tested
    total_queries: int  # Total query images
    min_entries_per_person: int
    
    # Detection rates
    face_detected: int
    feature_extracted: int
    query_succeeded: int
    
    # Cross-image accuracy (the key metrics)
    same_person_found: int  # Found any other entry of same person
    same_person_in_top1: int
    same_person_in_top5: int
    same_person_in_top10: int
    
    # Rates
    cross_image_top1_rate: float  # Main accuracy metric
    cross_image_top5_rate: float
    cross_image_top10_rate: float
    
    # Score statistics
    avg_same_person_score: float
    avg_different_person_score: float
    
    test_duration_seconds: float


def get_repeat_visitors(min_entries: int = 100, limit_persons: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Get visitors grouped by fullName, for persons with multiple entries.
    
    Returns:
        Dict mapping fullName -> list of visitor records
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # First find fullNames with enough entries
        cursor.execute('''
            SELECT "fullName", COUNT(*) as cnt
            FROM public."Visitor"
            WHERE "base64Image" IS NOT NULL
              AND "fullName" IS NOT NULL
              AND "fullName" != ''
            GROUP BY "fullName"
            HAVING COUNT(*) >= %s
            ORDER BY cnt DESC
        ''', (min_entries,))
        
        names = cursor.fetchall()
        logger.info(f"Found {len(names)} persons with >= {min_entries} entries")
        
        if limit_persons:
            names = names[:limit_persons]
        
        # Now get all entries for these names
        result = {}
        for name_row in names:
            name = name_row['fullName']
            cursor.execute('''
                SELECT "id", "fullName", "base64Image", "firstName", "lastName"
                FROM public."Visitor"
                WHERE "fullName" = %s
                  AND "base64Image" IS NOT NULL
                ORDER BY "createdAt" DESC
            ''', (name,))
            entries = [dict(row) for row in cursor.fetchall()]
            result[name] = entries
            
        return result


def get_all_ids_for_person(person_entries: List[Dict]) -> Set[str]:
    """Get all visitor IDs for a person's entries."""
    return {str(e['id']) for e in person_entries}


def test_single_query(
    query_entry: Dict,
    person_name: str,
    same_person_ids: Set[str],
    recognizer,
    index,
    threshold: float
) -> CrossImageResult:
    """Test recognition for a single query image."""
    query_id = str(query_entry['id'])
    base64_image = query_entry.get('base64Image')
    
    result = CrossImageResult(
        query_id=query_id,
        person_name=person_name
    )
    
    if not base64_image:
        result.error = "no_image"
        return result
    
    try:
        # Load image
        image = image_loader.load_from_base64(base64_image)
        if image is None:
            result.error = "invalid_image"
            return result
        
        # Detect face
        faces = inference.detect_faces(image, return_landmarks=True)
        if faces is None or len(faces) == 0:
            result.error = "no_face"
            return result
        
        result.face_detected = True
        
        # Extract features
        feature = recognizer.extract_features(image, np.asarray(faces[0]))
        if feature is None:
            result.error = "feature_extraction_failed"
            return result
        
        result.feature_extracted = True
        feature = feature.flatten().astype(np.float32)
        
        # Search index
        raw_matches = index.search(feature, k=20)
        if not raw_matches:
            result.error = "no_matches"
            return result
        
        result.query_succeeded = True
        
        # Convert to (id, score) and filter out self
        matches = [(vid, score) for vid, score, _ in raw_matches if vid != query_id]
        
        if not matches:
            result.error = "only_self_match"
            return result
        
        # Analyze matches
        result.top1_id = matches[0][0]
        result.top1_score = matches[0][1]
        result.top1_is_same_person = matches[0][0] in same_person_ids
        
        # Check top-1, top-5, top-10 for same person
        for rank, (vid, score) in enumerate(matches[:10]):
            is_same = vid in same_person_ids
            
            if is_same:
                result.same_person_count_in_top10 += 1
                
                if not result.same_person_found:
                    result.same_person_found = True
                    result.best_same_person_rank = rank + 1
                    result.best_same_person_score = score
                
                if rank == 0:
                    result.same_person_in_top1 = True
                if rank < 5:
                    result.same_person_in_top5 = True
                if rank < 10:
                    result.same_person_in_top10 = True
        
        return result
        
    except Exception as e:
        result.error = str(e)
        return result


def run_cross_image_test(
    min_entries: int = 100,
    max_queries_per_person: int = 50,
    limit_persons: Optional[int] = None,
    threshold: Optional[float] = None,
    output_file: Optional[str] = None
) -> CrossImageMetrics:
    """
    Run cross-image recognition accuracy test.
    
    Args:
        min_entries: Minimum entries per person to include in test
        max_queries_per_person: Max queries to run per person
        limit_persons: Limit number of persons to test
        threshold: Similarity threshold
        output_file: Path to save detailed results
    """
    start_time = time.time()
    
    # Get recognizer and index
    recognizer = get_recognizer()
    index = get_index()
    
    if threshold is None:
        threshold = recognizer.default_threshold
    
    logger.info(f"Recognizer: {recognizer.name} ({recognizer.feature_dim}-dim)")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Index size: {index.ntotal} vectors")
    
    # Get repeat visitors
    logger.info(f"Finding persons with >= {min_entries} entries...")
    persons = get_repeat_visitors(min_entries=min_entries, limit_persons=limit_persons)
    
    if not persons:
        logger.error("No repeat visitors found")
        return None
    
    logger.info(f"Testing {len(persons)} persons")
    
    # Run tests
    all_results: List[CrossImageResult] = []
    errors_by_type = defaultdict(int)
    
    total_queries = 0
    for person_name, entries in persons.items():
        same_person_ids = get_all_ids_for_person(entries)
        
        # Sample queries from this person's entries
        query_entries = entries[:max_queries_per_person]
        
        for entry in query_entries:
            result = test_single_query(
                entry, person_name, same_person_ids,
                recognizer, index, threshold
            )
            all_results.append(result)
            total_queries += 1
            
            if result.error:
                errors_by_type[result.error] += 1
            
            # Progress
            if total_queries % 50 == 0:
                elapsed = time.time() - start_time
                rate = total_queries / elapsed
                top1_correct = sum(1 for r in all_results if r.same_person_in_top1)
                logger.info(
                    f"Progress: {total_queries} queries | "
                    f"Cross-Image Top-1: {top1_correct}/{total_queries} "
                    f"({100*top1_correct/total_queries:.1f}%) | "
                    f"Rate: {rate:.1f}/s"
                )
    
    # Calculate metrics
    duration = time.time() - start_time
    
    face_detected = sum(1 for r in all_results if r.face_detected)
    feature_extracted = sum(1 for r in all_results if r.feature_extracted)
    query_succeeded = sum(1 for r in all_results if r.query_succeeded)
    
    same_person_found = sum(1 for r in all_results if r.same_person_found)
    same_person_top1 = sum(1 for r in all_results if r.same_person_in_top1)
    same_person_top5 = sum(1 for r in all_results if r.same_person_in_top5)
    same_person_top10 = sum(1 for r in all_results if r.same_person_in_top10)
    
    # Score statistics
    same_scores = [r.best_same_person_score for r in all_results if r.same_person_found]
    diff_scores = [r.top1_score for r in all_results if r.query_succeeded and not r.top1_is_same_person]
    
    metrics = CrossImageMetrics(
        recognizer=recognizer.name,
        threshold=threshold,
        total_persons=len(persons),
        total_queries=total_queries,
        min_entries_per_person=min_entries,
        face_detected=face_detected,
        feature_extracted=feature_extracted,
        query_succeeded=query_succeeded,
        same_person_found=same_person_found,
        same_person_in_top1=same_person_top1,
        same_person_in_top5=same_person_top5,
        same_person_in_top10=same_person_top10,
        cross_image_top1_rate=same_person_top1 / total_queries if total_queries > 0 else 0,
        cross_image_top5_rate=same_person_top5 / total_queries if total_queries > 0 else 0,
        cross_image_top10_rate=same_person_top10 / total_queries if total_queries > 0 else 0,
        avg_same_person_score=float(np.mean(same_scores)) if same_scores else 0,
        avg_different_person_score=float(np.mean(diff_scores)) if diff_scores else 0,
        test_duration_seconds=duration
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-IMAGE RECOGNITION ACCURACY TEST RESULTS")
    print("=" * 70)
    print(f"Recognizer: {metrics.recognizer}")
    print(f"Threshold: {metrics.threshold}")
    print(f"Duration: {metrics.test_duration_seconds:.1f}s")
    print()
    print("TEST SCOPE:")
    print(f"  Unique Persons Tested: {metrics.total_persons}")
    print(f"  Total Query Images: {metrics.total_queries}")
    print(f"  Min Entries per Person: {metrics.min_entries_per_person}")
    print()
    print("DETECTION RATES:")
    print(f"  Face Detected: {metrics.face_detected}/{metrics.total_queries} ({100*metrics.face_detected/metrics.total_queries:.1f}%)")
    print(f"  Feature Extracted: {metrics.feature_extracted}/{metrics.total_queries} ({100*metrics.feature_extracted/metrics.total_queries:.1f}%)")
    print()
    print("CROSS-IMAGE RECOGNITION ACCURACY:")
    print(f"  *** Top-1 Accuracy: {metrics.same_person_in_top1}/{metrics.total_queries} = {100*metrics.cross_image_top1_rate:.2f}% ***")
    print(f"  Top-5 Accuracy: {metrics.same_person_in_top5}/{metrics.total_queries} = {100*metrics.cross_image_top5_rate:.2f}%")
    print(f"  Top-10 Accuracy: {metrics.same_person_in_top10}/{metrics.total_queries} = {100*metrics.cross_image_top10_rate:.2f}%")
    print()
    print("SCORE STATISTICS:")
    print(f"  Avg Same-Person Match Score: {metrics.avg_same_person_score:.4f}")
    print(f"  Avg Different-Person Score: {metrics.avg_different_person_score:.4f}")
    print()
    
    if errors_by_type:
        print("ERRORS BY TYPE:")
        for error, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")
    
    print("=" * 70)
    
    # Per-person breakdown
    print("\nPER-PERSON BREAKDOWN:")
    print("-" * 70)
    for person_name in list(persons.keys())[:10]:  # Top 10 persons
        person_results = [r for r in all_results if r.person_name == person_name]
        top1 = sum(1 for r in person_results if r.same_person_in_top1)
        total = len(person_results)
        print(f"  {person_name}: {top1}/{total} = {100*top1/total:.1f}% cross-image Top-1")
    
    # Save results if requested
    if output_file:
        output_data = {
            "metrics": asdict(metrics),
            "results": [asdict(r) for r in all_results],
            "errors": dict(errors_by_type)
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Cross-image face recognition accuracy test'
    )
    parser.add_argument(
        '--min-entries', '-m',
        type=int,
        default=100,
        help='Minimum entries per person to include (default: 100)'
    )
    parser.add_argument(
        '--max-queries', '-q',
        type=int,
        default=50,
        help='Max query images per person (default: 50)'
    )
    parser.add_argument(
        '--limit-persons', '-l',
        type=int,
        default=None,
        help='Limit number of persons to test'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='Similarity threshold (default: recognizer default)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for detailed results (JSON)'
    )
    
    args = parser.parse_args()
    
    run_cross_image_test(
        min_entries=args.min_entries,
        max_queries_per_person=args.max_queries,
        limit_persons=args.limit_persons,
        threshold=args.threshold,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
