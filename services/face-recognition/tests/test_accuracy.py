#!/usr/bin/env python
"""
Face Recognition Accuracy Testing

Tests the accuracy of the face recognition API by:
1. Taking visitor images from the database
2. Using them as query images
3. Checking if the correct visitor is identified

Metrics:
- Top-1 Accuracy: Is the correct visitor the #1 match?
- Top-5 Accuracy: Is the correct visitor in top 5 matches?
- Precision, Recall, F1 Score
- False Accept Rate (FAR) / False Reject Rate (FRR)

Usage:
    python scripts/test_accuracy.py
    python scripts/test_accuracy.py --limit 1000 --threshold 0.5
    python scripts/test_accuracy.py --output results.json
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from collections import defaultdict

# Add app to path
APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

# Load environment
from dotenv import load_dotenv
load_dotenv(APP_DIR.parent / ".env.test")

import numpy as np

from core.logger import get_logger
from core.config import settings
from ml.recognizer_factory import get_recognizer
from ml.index_factory import get_index
from ml import inference
from db import database
from utils import image_loader

logger = get_logger("accuracy_test")


@dataclass
class TestResult:
    """Result of a single recognition test."""
    visitor_id: str
    query_succeeded: bool
    face_detected: bool
    feature_extracted: bool
    top1_correct: bool
    top5_correct: bool
    top1_id: Optional[str] = None
    top1_score: float = 0.0
    correct_rank: int = -1  # -1 if not found
    correct_score: float = 0.0
    error: Optional[str] = None


@dataclass 
class AccuracyMetrics:
    """Overall accuracy metrics."""
    total_tested: int
    face_detected: int
    feature_extracted: int
    
    top1_correct: int
    top5_correct: int
    
    top1_accuracy: float
    top5_accuracy: float
    
    # For threshold-based matching
    true_positives: int  # Correctly identified
    false_positives: int  # Wrong person identified
    false_negatives: int  # Should have matched but didn't
    true_negatives: int  # Correctly rejected (not in index)
    
    precision: float
    recall: float
    f1_score: float
    
    false_accept_rate: float  # FAR
    false_reject_rate: float  # FRR
    
    avg_correct_score: float
    avg_wrong_score: float
    
    test_duration_seconds: float
    recognizer: str
    threshold: float


def extract_feature_for_image(image: np.ndarray, recognizer) -> Optional[np.ndarray]:
    """Extract face feature from image."""
    try:
        faces = inference.detect_faces(image, return_landmarks=True)
        if faces is None or len(faces) == 0:
            return None
        
        feature = recognizer.extract_features(image, np.asarray(faces[0]))
        if feature is None:
            return None
        
        return feature.flatten().astype(np.float32)
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None


def query_index(
    feature: np.ndarray,
    index,
    exclude_id: str,
    k: int = 10
) -> List[Tuple[str, float]]:
    """
    Query the index and return top-k matches.
    
    Returns:
        List of (visitor_id, score) tuples, excluding the query visitor
    """
    try:
        results = index.search(feature, k=k + 1)  # Get extra in case we exclude
        
        # Results are (visitor_id, score, metadata) tuples
        # Filter out the excluded ID and extract just id and score
        filtered = [(vid, score) for vid, score, _ in results if vid != exclude_id]
        
        return filtered[:k]
    except Exception as e:
        logger.debug(f"Search error: {e}")
        return []


def test_single_visitor(
    visitor: dict,
    recognizer,
    index,
    threshold: float
) -> TestResult:
    """Test recognition accuracy for a single visitor."""
    visitor_id = str(visitor.get('id', 'unknown'))
    base64_image = visitor.get('base64Image')
    
    result = TestResult(
        visitor_id=visitor_id,
        query_succeeded=False,
        face_detected=False,
        feature_extracted=False,
        top1_correct=False,
        top5_correct=False
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
        
        # Extract feature
        feature = extract_feature_for_image(image, recognizer)
        if feature is None:
            result.error = "no_face_or_feature"
            return result
        
        result.face_detected = True
        result.feature_extracted = True
        
        # Query index (the visitor IS in the index, so we check if it's returned)
        raw_matches = index.search(feature, k=10)
        
        if not raw_matches:
            result.error = "no_matches"
            return result
        
        # Convert to (id, score) tuples - raw_matches are (id, score, metadata)
        matches = [(vid, score) for vid, score, _ in raw_matches]
        
        result.query_succeeded = True
        
        # Check top-1
        top1_id, top1_score = matches[0]
        result.top1_id = top1_id
        result.top1_score = top1_score
        result.top1_correct = (top1_id == visitor_id)
        
        # Check top-5 and find correct rank
        for rank, (vid, score) in enumerate(matches[:5]):
            if vid == visitor_id:
                result.top5_correct = True
                result.correct_rank = rank + 1
                result.correct_score = score
                break
        
        # Check beyond top-5 if not found
        if result.correct_rank == -1:
            for rank, (vid, score) in enumerate(matches):
                if vid == visitor_id:
                    result.correct_rank = rank + 1
                    result.correct_score = score
                    break
        
        return result
        
    except Exception as e:
        result.error = str(e)
        return result


def calculate_metrics(
    results: List[TestResult],
    threshold: float,
    recognizer_name: str,
    duration: float
) -> AccuracyMetrics:
    """Calculate accuracy metrics from test results."""
    total = len(results)
    
    face_detected = sum(1 for r in results if r.face_detected)
    feature_extracted = sum(1 for r in results if r.feature_extracted)
    
    top1_correct = sum(1 for r in results if r.top1_correct)
    top5_correct = sum(1 for r in results if r.top5_correct)
    
    # For precision/recall, only consider cases where we got a result
    successful = [r for r in results if r.query_succeeded]
    
    # True Positive: Correct match above threshold
    tp = sum(1 for r in successful if r.top1_correct and r.top1_score >= threshold)
    
    # False Positive: Wrong match above threshold
    fp = sum(1 for r in successful if not r.top1_correct and r.top1_score >= threshold)
    
    # False Negative: Should have matched (self) but score below threshold
    fn = sum(1 for r in successful if r.top1_correct and r.top1_score < threshold)
    
    # True Negative: Wrong match correctly rejected (below threshold)
    tn = sum(1 for r in successful if not r.top1_correct and r.top1_score < threshold)
    
    # Calculate rates
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # FAR = FP / (FP + TN) - rate of accepting wrong person
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # FRR = FN / (FN + TP) - rate of rejecting correct person
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Average scores
    correct_scores = [r.correct_score for r in results if r.correct_rank == 1]
    wrong_scores = [r.top1_score for r in results if r.query_succeeded and not r.top1_correct]
    
    avg_correct = np.mean(correct_scores) if correct_scores else 0.0
    avg_wrong = np.mean(wrong_scores) if wrong_scores else 0.0
    
    return AccuracyMetrics(
        total_tested=total,
        face_detected=face_detected,
        feature_extracted=feature_extracted,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        top1_accuracy=top1_correct / total if total > 0 else 0.0,
        top5_accuracy=top5_correct / total if total > 0 else 0.0,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_accept_rate=far,
        false_reject_rate=frr,
        avg_correct_score=float(avg_correct),
        avg_wrong_score=float(avg_wrong),
        test_duration_seconds=duration,
        recognizer=recognizer_name,
        threshold=threshold
    )


def run_accuracy_test(
    limit: Optional[int] = None,
    threshold: Optional[float] = None,
    sample_random: bool = True,
    output_file: Optional[str] = None
) -> AccuracyMetrics:
    """
    Run accuracy test on visitor images.
    
    Args:
        limit: Max number of visitors to test (None = all)
        threshold: Similarity threshold (uses recognizer default if None)
        sample_random: If True and limit is set, randomly sample
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
    
    # Get visitors from database (with limit to avoid slow query)
    logger.info("Loading visitors from database...")
    
    # Fetch with limit at database level for efficiency
    db_limit = limit * 2 if limit else None  # Fetch extra in case some fail
    visitors = database.get_visitor_images_from_db(limit=db_limit)
    
    if not visitors:
        logger.error("No visitors found")
        return None
    
    logger.info(f"Loaded {len(visitors)} visitors from database")
    
    # Sample if limit is set and we have more than needed
    if limit and len(visitors) > limit:
        if sample_random:
            visitors = random.sample(visitors, limit)
        else:
            visitors = visitors[:limit]
    
    logger.info(f"Testing on {len(visitors)} visitors")
    
    # Run tests
    results: List[TestResult] = []
    errors_by_type = defaultdict(int)
    
    for i, visitor in enumerate(visitors):
        result = test_single_visitor(visitor, recognizer, index, threshold)
        results.append(result)
        
        if result.error:
            errors_by_type[result.error] += 1
        
        # Progress logging
        if (i + 1) % 100 == 0 or i == len(visitors) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            correct = sum(1 for r in results if r.top1_correct)
            logger.info(
                f"Progress: {i+1}/{len(visitors)} | "
                f"Top-1 Correct: {correct} ({100*correct/(i+1):.1f}%) | "
                f"Rate: {rate:.1f}/s"
            )
    
    # Calculate metrics
    duration = time.time() - start_time
    metrics = calculate_metrics(results, threshold, recognizer.name, duration)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ACCURACY TEST RESULTS")
    print("=" * 60)
    print(f"Recognizer: {metrics.recognizer}")
    print(f"Threshold: {metrics.threshold}")
    print(f"Total Tested: {metrics.total_tested}")
    print(f"Duration: {metrics.test_duration_seconds:.1f}s")
    print()
    print("DETECTION RATES:")
    print(f"  Face Detected: {metrics.face_detected} ({100*metrics.face_detected/metrics.total_tested:.1f}%)")
    print(f"  Feature Extracted: {metrics.feature_extracted} ({100*metrics.feature_extracted/metrics.total_tested:.1f}%)")
    print()
    print("RECOGNITION ACCURACY:")
    print(f"  Top-1 Accuracy: {metrics.top1_correct}/{metrics.total_tested} = {100*metrics.top1_accuracy:.2f}%")
    print(f"  Top-5 Accuracy: {metrics.top5_correct}/{metrics.total_tested} = {100*metrics.top5_accuracy:.2f}%")
    print()
    print("THRESHOLD-BASED METRICS:")
    print(f"  Precision: {100*metrics.precision:.2f}%")
    print(f"  Recall: {100*metrics.recall:.2f}%")
    print(f"  F1 Score: {100*metrics.f1_score:.2f}%")
    print()
    print("ERROR RATES:")
    print(f"  False Accept Rate (FAR): {100*metrics.false_accept_rate:.2f}%")
    print(f"  False Reject Rate (FRR): {100*metrics.false_reject_rate:.2f}%")
    print()
    print("SCORE STATISTICS:")
    print(f"  Avg Correct Match Score: {metrics.avg_correct_score:.4f}")
    print(f"  Avg Wrong Match Score: {metrics.avg_wrong_score:.4f}")
    print()
    
    if errors_by_type:
        print("ERRORS BY TYPE:")
        for error, count in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")
    
    print("=" * 60)
    
    # Save results if requested
    if output_file:
        output_data = {
            "metrics": asdict(metrics),
            "results": [asdict(r) for r in results],
            "errors": dict(errors_by_type)
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Test face recognition accuracy'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of visitors to test (default: all)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='Similarity threshold (default: recognizer default)'
    )
    parser.add_argument(
        '--no-random',
        action='store_true',
        help='Use first N visitors instead of random sample'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for detailed results (JSON)'
    )
    
    args = parser.parse_args()
    
    run_accuracy_test(
        limit=args.limit,
        threshold=args.threshold,
        sample_random=not args.no_random,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
