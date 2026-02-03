#!/usr/bin/env python
"""
Compare ArcFace vs SFace Real-World Accuracy

Runs the same cross-image recognition test on both recognizers
and produces a side-by-side comparison.

This directly compares feature extraction quality by using
the same test images with both models.

Usage:
    python scripts/compare_recognizers.py
    python scripts/compare_recognizers.py --min-entries 50 --max-queries 30
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
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
from ml import inference
from db.connection import get_connection
from utils import image_loader

logger = get_logger("compare_recognizers")


@dataclass
class RecognizerResult:
    """Results for a single recognizer."""
    name: str
    dimension: int
    threshold: float
    
    total_queries: int
    face_detected: int
    feature_extracted: int
    
    top1_correct: int
    top5_correct: int
    top10_correct: int
    
    top1_rate: float
    top5_rate: float
    top10_rate: float
    
    avg_same_person_score: float
    avg_different_person_score: float
    
    duration_seconds: float
    
    # Per-person breakdown
    per_person: Dict[str, Dict]


def get_repeat_visitors(min_entries: int = 100, limit_persons: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Get visitors grouped by fullName, for persons with multiple entries."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
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


def test_recognizer(
    recognizer_type: str,
    persons: Dict[str, List[Dict]],
    max_queries_per_person: int,
    threshold: float = None
) -> RecognizerResult:
    """Test a single recognizer on the dataset."""
    
    # Set recognizer type
    os.environ['RECOGNIZER_TYPE'] = recognizer_type
    
    # Force fresh creation of recognizer and index
    from core.config import settings
    from ml.hnsw_index import HNSWIndexManager
    
    models_dir = Path(settings.models.models_path)
    index_dir = str(models_dir)  # Same directory as models
    
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
    
    # Use provided threshold or fall back to default
    test_threshold = threshold if threshold is not None else recognizer.default_threshold
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {recognizer.name} ({recognizer.feature_dim}-dim)")
    logger.info(f"Index size: {index.ntotal} vectors")
    logger.info(f"Threshold: {test_threshold}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    total_queries = 0
    face_detected = 0
    feature_extracted = 0
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    
    same_scores = []
    diff_scores = []
    per_person = {}
    errors = defaultdict(int)
    
    for person_name, entries in persons.items():
        same_person_ids = {str(e['id']) for e in entries}
        query_entries = entries[:max_queries_per_person]
        
        person_top1 = 0
        person_total = 0
        
        for entry in query_entries:
            query_id = str(entry['id'])
            base64_image = entry.get('base64Image')
            
            if not base64_image:
                errors['no_image'] += 1
                continue
            
            total_queries += 1
            person_total += 1
            
            try:
                # Load image
                image = image_loader.load_from_base64(base64_image)
                if image is None:
                    errors['invalid_image'] += 1
                    continue
                
                # Detect face
                faces = inference.detect_faces(image, return_landmarks=True)
                if faces is None or len(faces) == 0:
                    errors['no_face'] += 1
                    continue
                
                face_detected += 1
                
                # Extract features using THIS recognizer
                feature = recognizer.extract_features(image, np.asarray(faces[0]))
                if feature is None:
                    errors['feature_failed'] += 1
                    continue
                
                feature_extracted += 1
                feature = feature.flatten().astype(np.float32)
                
                # Search index
                raw_matches = index.search(feature, k=15)
                if not raw_matches:
                    errors['no_matches'] += 1
                    continue
                
                # Filter out self
                matches = [(vid, score) for vid, score, _ in raw_matches if vid != query_id]
                
                if not matches:
                    errors['only_self'] += 1
                    continue
                
                # Check results
                top1_id, top1_score = matches[0]
                is_same_person = top1_id in same_person_ids
                
                if is_same_person:
                    top1_correct += 1
                    person_top1 += 1
                    same_scores.append(top1_score)
                else:
                    diff_scores.append(top1_score)
                
                # Top-5
                for vid, score in matches[:5]:
                    if vid in same_person_ids:
                        top5_correct += 1
                        break
                
                # Top-10
                for vid, score in matches[:10]:
                    if vid in same_person_ids:
                        top10_correct += 1
                        break
                        
            except Exception as e:
                errors[str(e)[:50]] += 1
        
        per_person[person_name] = {
            'top1': person_top1,
            'total': person_total,
            'rate': person_top1 / person_total if person_total > 0 else 0
        }
        
        # Progress
        logger.info(f"  {person_name}: {person_top1}/{person_total} = {100*person_top1/person_total:.1f}%")
    
    duration = time.time() - start_time
    
    return RecognizerResult(
        name=recognizer.name,
        dimension=recognizer.feature_dim,
        threshold=test_threshold,
        total_queries=total_queries,
        face_detected=face_detected,
        feature_extracted=feature_extracted,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        top10_correct=top10_correct,
        top1_rate=top1_correct / total_queries if total_queries > 0 else 0,
        top5_rate=top5_correct / total_queries if total_queries > 0 else 0,
        top10_rate=top10_correct / total_queries if total_queries > 0 else 0,
        avg_same_person_score=float(np.mean(same_scores)) if same_scores else 0,
        avg_different_person_score=float(np.mean(diff_scores)) if diff_scores else 0,
        duration_seconds=duration,
        per_person=per_person
    )


def print_comparison(arcface: RecognizerResult, sface: RecognizerResult):
    """Print side-by-side comparison."""
    
    def better(a, b, higher_is_better=True):
        """Return indicator for which is better."""
        if higher_is_better:
            if a > b:
                return "<< WINNER", ""
            elif b > a:
                return "", "<< WINNER"
        else:
            if a < b:
                return "<< WINNER", ""
            elif b < a:
                return "", "<< WINNER"
        return "", ""
    
    print("\n")
    print("=" * 80)
    print("  ARCFACE vs SFACE COMPARISON - CROSS-IMAGE RECOGNITION TEST")
    print("=" * 80)
    print()
    print(f"{'METRIC':<35} {'ARCFACE':>18} {'SFACE':>18}")
    print("-" * 80)
    
    # Basic info
    print(f"{'Feature Dimension':<35} {arcface.dimension:>18} {sface.dimension:>18}")
    print(f"{'Test Threshold':<35} {arcface.threshold:>18.2f} {sface.threshold:>18.2f}")
    print(f"{'Index Size':<35} {arcface.total_queries:>18} {sface.total_queries:>18}")
    print()
    
    # Detection
    print(f"{'Face Detection Rate':<35} {100*arcface.face_detected/arcface.total_queries:>17.1f}% {100*sface.face_detected/sface.total_queries:>17.1f}%")
    print(f"{'Feature Extraction Rate':<35} {100*arcface.feature_extracted/arcface.total_queries:>17.1f}% {100*sface.feature_extracted/sface.total_queries:>17.1f}%")
    print()
    
    # Accuracy (the key comparison)
    print("-" * 80)
    arc_b, sf_b = better(arcface.top1_rate, sface.top1_rate)
    print(f"{'*** TOP-1 ACCURACY ***':<35} {100*arcface.top1_rate:>14.2f}% {arc_b:>3} {100*sface.top1_rate:>11.2f}% {sf_b}")
    
    arc_b, sf_b = better(arcface.top5_rate, sface.top5_rate)
    print(f"{'Top-5 Accuracy':<35} {100*arcface.top5_rate:>17.2f}% {100*sface.top5_rate:>17.2f}%")
    
    arc_b, sf_b = better(arcface.top10_rate, sface.top10_rate)
    print(f"{'Top-10 Accuracy':<35} {100*arcface.top10_rate:>17.2f}% {100*sface.top10_rate:>17.2f}%")
    print("-" * 80)
    print()
    
    # Scores
    print(f"{'Avg Same-Person Score':<35} {arcface.avg_same_person_score:>18.4f} {sface.avg_same_person_score:>18.4f}")
    print(f"{'Avg Different-Person Score':<35} {arcface.avg_different_person_score:>18.4f} {sface.avg_different_person_score:>18.4f}")
    
    # Score gap (how separable are same vs different)
    arc_gap = arcface.avg_same_person_score - arcface.avg_different_person_score
    sf_gap = sface.avg_same_person_score - sface.avg_different_person_score
    arc_b, sf_b = better(arc_gap, sf_gap)
    print(f"{'Score Gap (separability)':<35} {arc_gap:>14.4f} {arc_b:>3} {sf_gap:>14.4f} {sf_b}")
    print()
    
    # Speed
    print(f"{'Test Duration (seconds)':<35} {arcface.duration_seconds:>18.1f} {sface.duration_seconds:>18.1f}")
    print()
    
    # Per-person comparison
    print("=" * 80)
    print("  PER-PERSON BREAKDOWN")
    print("=" * 80)
    print(f"{'PERSON':<30} {'ARCFACE':>12} {'SFACE':>12} {'WINNER':>15}")
    print("-" * 80)
    
    arcface_wins = 0
    sface_wins = 0
    ties = 0
    
    for person in arcface.per_person:
        arc_rate = arcface.per_person[person]['rate']
        sf_rate = sface.per_person.get(person, {}).get('rate', 0)
        
        if arc_rate > sf_rate:
            winner = "ARCFACE"
            arcface_wins += 1
        elif sf_rate > arc_rate:
            winner = "SFACE"
            sface_wins += 1
        else:
            winner = "TIE"
            ties += 1
        
        print(f"{person:<30} {100*arc_rate:>11.1f}% {100*sf_rate:>11.1f}% {winner:>15}")
    
    print("-" * 80)
    print(f"{'WINS':<30} {arcface_wins:>12} {sface_wins:>12}")
    print()
    
    # Summary
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    if arcface.top1_rate > sface.top1_rate:
        diff = arcface.top1_rate - sface.top1_rate
        print(f"  ArcFace is {100*diff:.1f}% more accurate than SFace in Top-1 recognition")
    elif sface.top1_rate > arcface.top1_rate:
        diff = sface.top1_rate - arcface.top1_rate
        print(f"  SFace is {100*diff:.1f}% more accurate than ArcFace in Top-1 recognition")
    else:
        print(f"  Both models have identical Top-1 accuracy")
    
    print(f"  ArcFace won {arcface_wins}/{len(arcface.per_person)} persons")
    print(f"  SFace won {sface_wins}/{len(sface.per_person)} persons")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare ArcFace vs SFace accuracy'
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
        '--threshold', '-t',
        type=float,
        default=None,
        help='Threshold to use for both models (overrides individual thresholds)'
    )
    parser.add_argument(
        '--arcface-threshold',
        type=float,
        default=0.40,
        help='Threshold for ArcFace (default: 0.40 - optimal from testing)'
    )
    parser.add_argument(
        '--sface-threshold',
        type=float,
        default=0.45,
        help='Threshold for SFace (default: 0.45 - optimal from testing)'
    )
    
    args = parser.parse_args()
    
    # Determine thresholds - use single threshold if provided, else use individual optimal ones
    arcface_threshold = args.threshold if args.threshold is not None else args.arcface_threshold
    sface_threshold = args.threshold if args.threshold is not None else args.sface_threshold
    
    logger.info(f"Using thresholds: ArcFace={arcface_threshold}, SFace={sface_threshold}")
    
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
    
    # Test ArcFace with its optimal threshold
    logger.info("\n" + "=" * 60)
    logger.info(f"TESTING ARCFACE (threshold={arcface_threshold})")
    logger.info("=" * 60)
    arcface_result = test_recognizer('arcface', persons, args.max_queries, threshold=arcface_threshold)
    
    # Test SFace with its optimal threshold
    logger.info("\n" + "=" * 60)
    logger.info(f"TESTING SFACE (threshold={sface_threshold})")
    logger.info("=" * 60)
    sface_result = test_recognizer('sface', persons, args.max_queries, threshold=sface_threshold)
    
    # Print comparison
    print_comparison(arcface_result, sface_result)


if __name__ == "__main__":
    main()
