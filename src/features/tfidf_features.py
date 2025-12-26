"""
TF-IDF based features for code classification

Author: Task B Improvement
Difficulty: ⭐⭐ Intermediate
Description: Character and word n-gram TF-IDF features to capture coding style patterns
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from pathlib import Path
from typing import Optional


# Global vectorizers - fitted once on training data
_char_vectorizer: Optional[TfidfVectorizer] = None
_word_vectorizer: Optional[TfidfVectorizer] = None
_is_fitted = False


def get_char_vectorizer() -> TfidfVectorizer:
    """Get or create character n-gram vectorizer"""
    global _char_vectorizer
    if _char_vectorizer is None:
        _char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            max_features=200,
            lowercase=False,  # Preserve case for code
            dtype=np.float32
        )
    return _char_vectorizer


def get_word_vectorizer() -> TfidfVectorizer:
    """Get or create word n-gram vectorizer"""
    global _word_vectorizer
    if _word_vectorizer is None:
        _word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=100,
            token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',  # Code identifiers
            lowercase=False,
            dtype=np.float32
        )
    return _word_vectorizer


def fit_tfidf(codes: list) -> None:
    """
    Fit TF-IDF vectorizers on training data

    Must be called once before extract_tfidf_features() can be used.

    Args:
        codes: List of code strings from training data
    """
    global _is_fitted

    print(f"Fitting TF-IDF vectorizers on {len(codes):,} samples...")

    char_vec = get_char_vectorizer()
    word_vec = get_word_vectorizer()

    char_vec.fit(codes)
    word_vec.fit(codes)

    _is_fitted = True
    print(f"  Char n-gram features: {len(char_vec.get_feature_names_out())}")
    print(f"  Word n-gram features: {len(word_vec.get_feature_names_out())}")


def is_fitted() -> bool:
    """Check if vectorizers have been fitted"""
    return _is_fitted


def extract_tfidf_features(code: str) -> dict:
    """
    Extract TF-IDF features from code

    Requires fit_tfidf() to be called first on training data.

    Args:
        code: Source code string

    Returns:
        dict: Feature name -> value mapping
    """
    if not _is_fitted:
        raise RuntimeError(
            "TF-IDF vectorizers not fitted. Call fit_tfidf(train_codes) first."
        )

    features = {}

    # Character n-gram features
    char_vec = get_char_vectorizer()
    char_features = char_vec.transform([code]).toarray()[0]
    for i, val in enumerate(char_features):
        features[f'tfidf_char_{i}'] = float(val)

    # Word n-gram features
    word_vec = get_word_vectorizer()
    word_features = word_vec.transform([code]).toarray()[0]
    for i, val in enumerate(word_features):
        features[f'tfidf_word_{i}'] = float(val)

    return features


def get_tfidf_vector(code: str) -> np.ndarray:
    """
    Get raw TF-IDF vector (combined char + word)

    Useful for computing similarity features.

    Args:
        code: Source code string

    Returns:
        np.ndarray: Combined TF-IDF vector
    """
    if not _is_fitted:
        raise RuntimeError(
            "TF-IDF vectorizers not fitted. Call fit_tfidf(train_codes) first."
        )

    char_vec = get_char_vectorizer()
    word_vec = get_word_vectorizer()

    char_features = char_vec.transform([code]).toarray()[0]
    word_features = word_vec.transform([code]).toarray()[0]

    return np.concatenate([char_features, word_features])


def get_tfidf_matrix(codes: list) -> np.ndarray:
    """
    Get TF-IDF matrix for multiple codes

    Args:
        codes: List of code strings

    Returns:
        np.ndarray: TF-IDF matrix of shape (n_samples, n_features)
    """
    if not _is_fitted:
        raise RuntimeError(
            "TF-IDF vectorizers not fitted. Call fit_tfidf(train_codes) first."
        )

    char_vec = get_char_vectorizer()
    word_vec = get_word_vectorizer()

    char_matrix = char_vec.transform(codes).toarray()
    word_matrix = word_vec.transform(codes).toarray()

    return np.hstack([char_matrix, word_matrix])


def save_vectorizers(path: str) -> None:
    """
    Save fitted vectorizers to disk

    Args:
        path: Directory path to save vectorizers
    """
    if not _is_fitted:
        raise RuntimeError("Vectorizers not fitted yet")

    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'char_vectorizer.pkl', 'wb') as f:
        pickle.dump(_char_vectorizer, f)

    with open(save_dir / 'word_vectorizer.pkl', 'wb') as f:
        pickle.dump(_word_vectorizer, f)

    print(f"Saved TF-IDF vectorizers to {save_dir}")


def load_vectorizers(path: str) -> None:
    """
    Load fitted vectorizers from disk

    Args:
        path: Directory path containing saved vectorizers
    """
    global _char_vectorizer, _word_vectorizer, _is_fitted

    load_dir = Path(path)

    with open(load_dir / 'char_vectorizer.pkl', 'rb') as f:
        _char_vectorizer = pickle.load(f)

    with open(load_dir / 'word_vectorizer.pkl', 'rb') as f:
        _word_vectorizer = pickle.load(f)

    _is_fitted = True
    print(f"Loaded TF-IDF vectorizers from {load_dir}")
