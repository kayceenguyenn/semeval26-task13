"""
Feature extraction for SemEval Task 13

This package uses a modular structure to avoid merge conflicts:
- Each student creates their own feature file
- No editing the same file simultaneously
- Easy to add/remove features

Usage:
    from src.features import extract_all_features
    features = extract_all_features(code)

For TF-IDF features (requires fitting on training data first):
    from src.features import fit_tfidf_pipeline, extract_all_features_with_tfidf
    fit_tfidf_pipeline(train_codes, train_labels)
    features = extract_all_features_with_tfidf(code)
"""

from .basic import extract_basic_features
from .keywords import extract_keyword_features
from .ast_features import extract_ast_features
from .tfidf_features import (
    extract_tfidf_features,
    fit_tfidf,
    get_tfidf_vector,
    get_tfidf_matrix,
    is_fitted as tfidf_is_fitted,
    save_vectorizers,
    load_vectorizers,
)
from .similarity_features import (
    extract_similarity_features,
    fit_centroids,
    is_fitted as centroids_is_fitted,
    save_centroids,
    load_centroids,
)

import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_all_features(code: str, include_tfidf: bool = False) -> dict:
    """
    Extract all features from code

    This function automatically combines features from all modules.
    When you add a new feature module, just import it above and
    add it to the features dict below.

    Args:
        code: Source code string
        include_tfidf: Whether to include TF-IDF and similarity features
                       (requires fit_tfidf_pipeline to be called first)

    Returns:
        dict: All features combined
    """
    features = {}

    # Basic features (always included)
    features.update(extract_basic_features(code))

    # Keyword features (always included)
    features.update(extract_keyword_features(code))

    # AST features (structural analysis)
    features.update(extract_ast_features(code))

    # TF-IDF and similarity features (only if fitted)
    if include_tfidf:
        if not tfidf_is_fitted():
            raise RuntimeError(
                "TF-IDF not fitted. Call fit_tfidf_pipeline(train_codes, train_labels) first."
            )
        features.update(extract_tfidf_features(code))

        if centroids_is_fitted():
            tfidf_vec = get_tfidf_vector(code)
            features.update(extract_similarity_features(tfidf_vec))

    return features


def fit_tfidf_pipeline(train_codes: list, train_labels: np.ndarray) -> None:
    """
    Fit TF-IDF vectorizers and compute class centroids

    Must be called once on training data before using TF-IDF features.

    Args:
        train_codes: List of code strings from training data
        train_labels: Array of class labels
    """
    print("Fitting TF-IDF pipeline...")

    # Fit TF-IDF vectorizers
    fit_tfidf(train_codes)

    # Get TF-IDF matrix for training data
    tfidf_matrix = get_tfidf_matrix(train_codes)

    # Compute class centroids
    fit_centroids(tfidf_matrix, train_labels)

    print("TF-IDF pipeline fitted successfully!")


def extract_features_from_dataframe(
    df: pd.DataFrame,
    include_tfidf: bool = False,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract features from a dataframe with 'code' column

    OPTIMIZED: TF-IDF is batched for 10-50x speedup.

    Args:
        df: DataFrame with 'code' column
        include_tfidf: Whether to include TF-IDF features
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with feature columns
    """
    codes = df['code'].tolist()
    n_samples = len(codes)

    # Step 1: Extract basic + keyword + AST features (must be per-sample)
    print(f"Extracting basic/keyword/AST features for {n_samples:,} samples...")
    basic_features_list = []
    iterator = tqdm(codes, desc="Basic features") if show_progress else codes
    for code in iterator:
        features = {}
        features.update(extract_basic_features(code))
        features.update(extract_keyword_features(code))
        features.update(extract_ast_features(code))
        basic_features_list.append(features)

    basic_df = pd.DataFrame(basic_features_list)

    # Step 2: Batch TF-IDF extraction (FAST - vectorized)
    if include_tfidf:
        if not tfidf_is_fitted():
            raise RuntimeError(
                "TF-IDF not fitted. Call fit_tfidf_pipeline(train_codes, train_labels) first."
            )

        print("Extracting TF-IDF features (batched)...")
        tfidf_matrix = get_tfidf_matrix(codes)  # Batch transform - much faster!

        # Convert to DataFrame with column names
        n_char_features = 200  # from char_vectorizer max_features
        n_word_features = 100  # from word_vectorizer max_features
        tfidf_cols = [f'tfidf_char_{i}' for i in range(n_char_features)] + \
                     [f'tfidf_word_{i}' for i in range(n_word_features)]
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)

        # Step 3: Batch similarity features
        if centroids_is_fitted():
            print("Extracting similarity features (batched)...")
            from .similarity_features import extract_similarity_features_batch, get_centroids
            sim_matrix = extract_similarity_features_batch(tfidf_matrix)
            sim_cols = [f'sim_to_class_{label}' for label in sorted(get_centroids().keys())]
            sim_df = pd.DataFrame(sim_matrix, columns=sim_cols)

            # Combine all features
            result_df = pd.concat([basic_df, tfidf_df, sim_df], axis=1)
        else:
            result_df = pd.concat([basic_df, tfidf_df], axis=1)
    else:
        result_df = basic_df

    print(f"Done! Total features: {result_df.shape[1]}")
    return result_df


def save_fitted_state(path: str) -> None:
    """
    Save fitted TF-IDF vectorizers and centroids

    Args:
        path: Directory path to save state
    """
    save_vectorizers(path)
    save_centroids(path)


def load_fitted_state(path: str) -> None:
    """
    Load fitted TF-IDF vectorizers and centroids

    Args:
        path: Directory path containing saved state
    """
    load_vectorizers(path)
    load_centroids(path)
