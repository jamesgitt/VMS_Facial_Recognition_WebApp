#!/usr/bin/env python
"""Verify that both indexes are using correct dimensions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.test")

from core.config import settings
from ml.sface_recognizer import SFaceRecognizer
from ml.arcface_recognizer import ArcFaceRecognizer
from ml.hnsw_index import HNSWIndexManager
from ml import inference
from utils import image_loader
from db.connection import get_connection
from psycopg2.extras import RealDictCursor
import numpy as np

models_dir = Path(settings.models.models_path)

# Load both recognizers
print("Loading recognizers...")
sface = SFaceRecognizer(str(models_dir / settings.models.sface_filename))
arcface = ArcFaceRecognizer(str(models_dir / settings.models.arcface_filename))

# Load both indexes
print("Loading indexes...")
sface_idx = HNSWIndexManager(dimension=128, recognizer_name='sface', index_dir=str(models_dir))
arcface_idx = HNSWIndexManager(dimension=512, recognizer_name='arcface', index_dir=str(models_dir))

print()
print("=" * 60)
print("INDEX VERIFICATION")
print("=" * 60)
print(f"SFace recognizer: {sface.name}, {sface.feature_dim}-dim")
print(f"SFace index: {sface_idx.ntotal} vectors, {sface_idx.dimension}-dim")
print()
print(f"ArcFace recognizer: {arcface.name}, {arcface.feature_dim}-dim")
print(f"ArcFace index: {arcface_idx.ntotal} vectors, {arcface_idx.dimension}-dim")

# Test feature extraction from a sample image
print()
print("=" * 60)
print("FEATURE EXTRACTION TEST")
print("=" * 60)

with get_connection() as conn:
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('''
        SELECT "id", "base64Image" 
        FROM public."Visitor" 
        WHERE "base64Image" IS NOT NULL 
        LIMIT 1
    ''')
    visitor = dict(cursor.fetchone())

image = image_loader.load_from_base64(visitor['base64Image'])
faces = inference.detect_faces(image, return_landmarks=True)

if faces is not None and len(faces) > 0:
    face = np.asarray(faces[0])
    
    sface_feat = sface.extract_features(image, face)
    arcface_feat = arcface.extract_features(image, face)
    
    print(f"Visitor ID: {visitor['id']}")
    print(f"SFace extracted feature shape: {sface_feat.shape}")
    print(f"ArcFace extracted feature shape: {arcface_feat.shape}")
    
    # Verify dimensions match
    print()
    print("DIMENSION CHECK:")
    print(f"  SFace feature dim ({sface_feat.flatten().shape[0]}) == SFace index dim ({sface_idx.dimension}): {sface_feat.flatten().shape[0] == sface_idx.dimension}")
    print(f"  ArcFace feature dim ({arcface_feat.flatten().shape[0]}) == ArcFace index dim ({arcface_idx.dimension}): {arcface_feat.flatten().shape[0] == arcface_idx.dimension}")
    
    # Search both indexes
    print()
    print("=" * 60)
    print("SEARCH TEST")
    print("=" * 60)
    
    print(f"\nSearching SFace index with SFace features ({sface_feat.flatten().shape[0]}-dim)...")
    sface_results = sface_idx.search(sface_feat.flatten(), k=5)
    print(f"SFace results (top 5):")
    for vid, score, _ in sface_results[:5]:
        match = "SELF" if vid == str(visitor['id']) else ""
        print(f"  {vid}: {score:.4f} {match}")
    
    print(f"\nSearching ArcFace index with ArcFace features ({arcface_feat.flatten().shape[0]}-dim)...")
    arcface_results = arcface_idx.search(arcface_feat.flatten(), k=5)
    print(f"ArcFace results (top 5):")
    for vid, score, _ in arcface_results[:5]:
        match = "SELF" if vid == str(visitor['id']) else ""
        print(f"  {vid}: {score:.4f} {match}")
    
    # Check if self is found
    print()
    print("=" * 60)
    print("SELF-RECOGNITION CHECK")
    print("=" * 60)
    
    sface_self_found = any(vid == str(visitor['id']) for vid, _, _ in sface_results)
    arcface_self_found = any(vid == str(visitor['id']) for vid, _, _ in arcface_results)
    
    print(f"SFace found self in top 5: {sface_self_found}")
    print(f"ArcFace found self in top 5: {arcface_self_found}")

else:
    print("No face detected in sample image")
