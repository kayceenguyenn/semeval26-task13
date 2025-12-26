"""
Cosine similarity features for code classification

Author: Task B Improvement
Difficulty: ⭐⭐ Intermediate
Description: Compute cosine similarity to class centroids for style matching
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict


# Global class centroids - computed from training data
_class_centroids: Dict[int, np.ndarray] = {}
_is_fitted = False


def fit_centroids(tfidf_matrix: np.ndarray, labels: np.ndarray) -> None:
    """
    Compute class centroids from training data TF-IDF vectors

    Args:
        tfidf_matrix: TF-IDF matrix of shape (n_samples, n_features)
        labels: Class labels of shape (n_samples,)
    """
    global _class_centroids, _is_fitted

    print(f"Computing class centroids for {len(np.unique(labels))} classes...")

    _class_centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        centroid = tfidf_matrix[mask].mean(axis=0)
        # Ensure it's a 1D array
        if hasattr(centroid, 'A1'):
            centroid = centroid.A1  # Convert sparse matrix row to array
        _class_centroids[int(label)] = np.asarray(centroid).flatten()
        print(f"  Class {label}: {mask.sum():,} samples")

    _is_fitted = True
    print(f"Computed {len(_class_centroids)} class centroids")


def is_fitted() -> bool:
    """Check if centroids have been computed"""
    return _is_fitted


def get_centroids() -> Dict[int, np.ndarray]:
    """Get computed class centroids"""
    return _class_centroids


def extract_similarity_features(tfidf_vector: np.ndarray) -> dict:
    """
    Compute cosine similarity to each class centroid

    Args:
        tfidf_vector: TF-IDF vector for a single code sample

    Returns:
        dict: Feature name -> similarity value mapping
    """
    if not _is_fitted:
        raise RuntimeError(
            "Class centroids not computed. Call fit_centroids(tfidf_matrix, labels) first."
        )

    features = {}

    # Ensure vector is 2D for cosine_similarity
    if tfidf_vector.ndim == 1:
        tfidf_vector = tfidf_vector.reshape(1, -1)

    for label, centroid in sorted(_class_centroids.items()):
        centroid_2d = centroid.reshape(1, -1)
        sim = cosine_similarity(tfidf_vector, centroid_2d)[0][0]
        features[f'sim_to_class_{label}'] = float(sim)

    return features


def extract_similarity_features_batch(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Compute similarity features for multiple samples efficiently

    Args:
        tfidf_matrix: TF-IDF matrix of shape (n_samples, n_features)

    Returns:
        np.ndarray: Similarity matrix of shape (n_samples, n_classes)
    """
    if not _is_fitted:
        raise RuntimeError(
            "Class centroids not computed. Call fit_centroids(tfidf_matrix, labels) first."
        )

    n_samples = tfidf_matrix.shape[0]
    n_classes = len(_class_centroids)

    # Stack centroids into matrix
    centroid_matrix = np.vstack([
        _class_centroids[label] for label in sorted(_class_centroids.keys())
    ])

    # Compute all similarities at once
    similarities = cosine_similarity(tfidf_matrix, centroid_matrix)

    return similarities


def save_centroids(path: str) -> None:
    """
    Save computed centroids to disk

    Args:
        path: Directory path to save centroids
    """
    if not _is_fitted:
        raise RuntimeError("Centroids not computed yet")

    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'class_centroids.pkl', 'wb') as f:
        pickle.dump(_class_centroids, f)

    print(f"Saved class centroids to {save_dir}")


def load_centroids(path: str) -> None:
    """
    Load computed centroids from disk

    Args:
        path: Directory path containing saved centroids
    """
    global _class_centroids, _is_fitted

    load_dir = Path(path)

    with open(load_dir / 'class_centroids.pkl', 'rb') as f:
        _class_centroids = pickle.load(f)

    _is_fitted = True
    print(f"Loaded {len(_class_centroids)} class centroids from {load_dir}")
