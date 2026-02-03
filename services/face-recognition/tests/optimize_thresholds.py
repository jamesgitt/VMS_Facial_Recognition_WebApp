#!/usr/bin/env python3
"""
Threshold Optimization Script

Tests multiple thresholds to find the optimal value for each recognizer.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from core.config import settings
from core.logger import logger
from utils import image_loader
from ml import inference


@dataclass
class ThresholdResult:
    """Results for a single threshold test."""
    threshold: float
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


def get_db_connection():
    """Get database connection."""
    # Use DATABASE_URL if available, otherwise use individual params
    if settings.database.database_url:
        return psycopg2.connect(
            settings.database.database_url,
            cursor_factory=RealDictCursor
        )
    else:
        return psycopg2.connect(
            host=settings.database.host,
            port=settings.database.port,
            database=settings.database.name,
            user=settings.database.user,
            password=settings.database.password,
            cursor_factory=RealDictCursor
        )


def get_repeat_visitors(min_entries: int = 100, limit_persons: int = None) -> Dict[str, List[Dict]]:
    """Get visitors with multiple entries for testing."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
    
    if limit_persons:
        names = names[:limit_persons]
    
    result = {}
    for name_row in names:
        name = name_row['fullName']
        cursor.execute('''
            SELECT "id", "fullName", "base64Image"
            FROM public."Visitor"
            WHERE "fullName" = %s
              AND "base64Image" IS NOT NULL
            ORDER BY "createdAt" DESC
        ''', (name,))
        entries = [dict(row) for row in cursor.fetchall()]
        result[name] = entries
    
    conn.close()
    return result


def test_threshold(
    recognizer,
    index,
    persons: Dict[str, List[Dict]],
    threshold: float,
    max_queries_per_person: int = 50
) -> ThresholdResult:
    """Test a single threshold value."""
    
    total_queries = 0
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    
    # For precision/recall calculation
    true_positives = 0  # Correctly identified as same person above threshold
    false_positives = 0  # Wrongly identified as same person above threshold
    true_negatives = 0  # Correctly rejected below threshold
    false_negatives = 0  # Wrongly rejected below threshold (missed same person)
    
    for person_name, entries in persons.items():
        same_person_ids = {str(e['id']) for e in entries}
        query_entries = entries[:max_queries_per_person]
        
        for entry in query_entries:
            query_id = str(entry['id'])
            base64_image = entry.get('base64Image')
            
            if not base64_image:
                continue
            
            try:
                image = image_loader.load_from_base64(base64_image)
                if image is None:
                    continue
                
                # Use inference for face detection (uses YuNet)
                faces = inference.detect_faces(image, return_landmarks=True)
                if faces is None or len(faces) == 0:
                    continue
                
                feature = recognizer.extract_features(image, np.asarray(faces[0]))
                if feature is None:
                    continue
                
                feature = feature.flatten().astype(np.float32)
                total_queries += 1
                
                # Search index
                results = index.search(feature, k=20)
                
                # Filter out self
                matches = [(vid, score) for vid, score, _ in results if vid != query_id]
                
                if not matches:
                    continue
                
                # Check top-1, top-5, top-10
                top1_match_id, top1_score = matches[0]
                is_top1_same_person = top1_match_id in same_person_ids
                
                if is_top1_same_person:
                    top1_correct += 1
                
                # Check if any same person in top-5
                for vid, score in matches[:5]:
                    if vid in same_person_ids:
                        top5_correct += 1
                        break
                
                # Check if any same person in top-10
                for vid, score in matches[:10]:
                    if vid in same_person_ids:
                        top10_correct += 1
                        break
                
                # Precision/Recall calculation based on threshold
                # Check top-1 match against threshold
                if top1_score >= threshold:
                    # We're accepting this match
                    if is_top1_same_person:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    # We're rejecting this match
                    if is_top1_same_person:
                        false_negatives += 1  # Should have accepted
                    else:
                        true_negatives += 1
                        
            except Exception as e:
                continue
    
    # Calculate metrics
    top1_accuracy = top1_correct / total_queries if total_queries > 0 else 0
    top5_accuracy = top5_correct / total_queries if total_queries > 0 else 0
    top10_accuracy = top10_correct / total_queries if total_queries > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return ThresholdResult(
        threshold=threshold,
        top1_accuracy=top1_accuracy,
        top5_accuracy=top5_accuracy,
        top10_accuracy=top10_accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )


def optimize_recognizer(
    recognizer_type: str,
    persons: Dict[str, List[Dict]],
    thresholds: List[float],
    max_queries_per_person: int = 50
) -> List[ThresholdResult]:
    """Test multiple thresholds for a recognizer."""
    
    from ml.hnsw_index import HNSWIndexManager
    
    models_dir = Path(settings.models.models_path)
    index_dir = str(models_dir)
    
    if recognizer_type == 'sface':
        from ml.sface_recognizer import SFaceRecognizer
        model_path = str(models_dir / settings.models.sface_filename)
        recognizer = SFaceRecognizer(model_path=model_path)
        index = HNSWIndexManager(dimension=128, recognizer_name='sface', index_dir=index_dir)
    else:
        from ml.arcface_recognizer import ArcFaceRecognizer
        model_path = str(models_dir / settings.models.arcface_filename)
        recognizer = ArcFaceRecognizer(model_path=model_path)
        index = HNSWIndexManager(dimension=512, recognizer_name='arcface', index_dir=index_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing: {recognizer.name} ({recognizer.feature_dim}-dim)")
    logger.info(f"Index size: {index.ntotal} vectors")
    logger.info(f"Testing {len(thresholds)} thresholds: {thresholds}")
    logger.info(f"{'='*60}")
    
    results = []
    
    for threshold in thresholds:
        logger.info(f"  Testing threshold {threshold:.2f}...")
        result = test_threshold(
            recognizer, index, persons, threshold, max_queries_per_person
        )
        results.append(result)
        logger.info(f"    Top-1: {100*result.top1_accuracy:.1f}%, Precision: {100*result.precision:.1f}%, Recall: {100*result.recall:.1f}%, F1: {100*result.f1_score:.1f}%")
    
    return results


def print_optimization_results(
    recognizer_name: str,
    results: List[ThresholdResult]
):
    """Print optimization results in a table."""
    
    print(f"\n{'='*100}")
    print(f"  {recognizer_name.upper()} THRESHOLD OPTIMIZATION RESULTS")
    print(f"{'='*100}")
    print()
    print(f"{'Threshold':>10} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'Top-10 Acc':>12} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}")
    print("-" * 100)
    
    best_top1 = max(results, key=lambda x: x.top1_accuracy)
    best_f1 = max(results, key=lambda x: x.f1_score)
    best_precision = max(results, key=lambda x: x.precision)
    best_recall = max(results, key=lambda x: x.recall)
    
    for r in results:
        markers = []
        if r.threshold == best_top1.threshold:
            markers.append("TOP1")
        if r.threshold == best_f1.threshold:
            markers.append("F1")
        if r.threshold == best_precision.threshold:
            markers.append("PREC")
        if r.threshold == best_recall.threshold:
            markers.append("REC")
        
        marker_str = " << " + ",".join(markers) if markers else ""
        
        print(f"{r.threshold:>10.2f} {100*r.top1_accuracy:>11.2f}% {100*r.top5_accuracy:>11.2f}% {100*r.top10_accuracy:>11.2f}% {100*r.precision:>11.2f}% {100*r.recall:>11.2f}% {100*r.f1_score:>11.2f}%{marker_str}")
    
    print("-" * 100)
    print()
    print(f"  Best for Top-1 Accuracy: {best_top1.threshold:.2f} ({100*best_top1.top1_accuracy:.2f}%)")
    print(f"  Best for F1 Score:       {best_f1.threshold:.2f} ({100*best_f1.f1_score:.2f}%)")
    print(f"  Best for Precision:      {best_precision.threshold:.2f} ({100*best_precision.precision:.2f}%)")
    print(f"  Best for Recall:         {best_recall.threshold:.2f} ({100*best_recall.recall:.2f}%)")
    print()
    
    return best_top1, best_f1


def main():
    parser = argparse.ArgumentParser(
        description='Optimize thresholds for face recognition models'
    )
    parser.add_argument(
        '--min-entries', '-m',
        type=int,
        default=100,
        help='Minimum entries per person (default: 100)'
    )
    parser.add_argument(
        '--max-queries', '-q',
        type=int,
        default=50,
        help='Max queries per person (default: 50)'
    )
    parser.add_argument(
        '--limit-persons', '-l',
        type=int,
        default=None,
        help='Limit number of persons'
    )
    parser.add_argument(
        '--recognizer', '-r',
        choices=['sface', 'arcface', 'both'],
        default='both',
        help='Which recognizer to optimize (default: both)'
    )
    
    args = parser.parse_args()
    
    # Define thresholds to test
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    # Load test data once
    logger.info("Loading test data...")
    persons = get_repeat_visitors(
        min_entries=args.min_entries,
        limit_persons=args.limit_persons
    )
    
    if not persons:
        logger.error("No repeat visitors found")
        return
    
    total_entries = sum(len(v) for v in persons.values())
    logger.info(f"Testing {len(persons)} persons with {total_entries} total entries")
    
    results = {}
    
    # Test ArcFace
    if args.recognizer in ['arcface', 'both']:
        arcface_results = optimize_recognizer(
            'arcface', persons, thresholds, args.max_queries
        )
        results['arcface'] = arcface_results
        arc_best_top1, arc_best_f1 = print_optimization_results('ArcFace', arcface_results)
    
    # Test SFace
    if args.recognizer in ['sface', 'both']:
        sface_results = optimize_recognizer(
            'sface', persons, thresholds, args.max_queries
        )
        results['sface'] = sface_results
        sf_best_top1, sf_best_f1 = print_optimization_results('SFace', sface_results)
    
    # Summary comparison
    if args.recognizer == 'both':
        print("\n")
        print("=" * 80)
        print("  OPTIMAL THRESHOLD SUMMARY")
        print("=" * 80)
        print()
        print(f"{'Metric':<25} {'ArcFace':>25} {'SFace':>25}")
        print("-" * 80)
        print(f"{'Best Top-1 Threshold':<25} {arc_best_top1.threshold:>25.2f} {sf_best_top1.threshold:>25.2f}")
        print(f"{'Best Top-1 Accuracy':<25} {100*arc_best_top1.top1_accuracy:>24.2f}% {100*sf_best_top1.top1_accuracy:>24.2f}%")
        print()
        print(f"{'Best F1 Threshold':<25} {arc_best_f1.threshold:>25.2f} {sf_best_f1.threshold:>25.2f}")
        print(f"{'Best F1 Score':<25} {100*arc_best_f1.f1_score:>24.2f}% {100*sf_best_f1.f1_score:>24.2f}%")
        print("=" * 80)
        print()
        print("RECOMMENDATIONS:")
        print(f"  - ArcFace: Use threshold {arc_best_f1.threshold:.2f} for balanced precision/recall")
        print(f"  - SFace:   Use threshold {sf_best_f1.threshold:.2f} for balanced precision/recall")
        print()


if __name__ == "__main__":
    main()
