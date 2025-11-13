"""
Feature extraction for SemEval Task 13

This package uses a modular structure to avoid merge conflicts:
- Each student creates their own feature file
- No editing the same file simultaneously
- Easy to add/remove features

Usage:
    from src.features import extract_all_features
    features = extract_all_features(code)
"""

from .basic import extract_basic_features
from .keywords import extract_keyword_features
# Add your feature module here:
# from .your_name import extract_your_features

import pandas as pd


def extract_all_features(code: str) -> dict:
    """
    Extract all features from code
    
    This function automatically combines features from all modules.
    When you add a new feature module, just import it above and
    add it to the features dict below.
    
    Args:
        code: Source code string
        
    Returns:
        dict: All features combined
    """
    features = {}
    
    # Basic features (always included)
    features.update(extract_basic_features(code))
    
    # Keyword features (always included)
    features.update(extract_keyword_features(code))
    
    # Add your features here:
    # features.update(extract_your_features(code))
    
    return features


def extract_features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from a dataframe with 'code' column
    
    Args:
        df: DataFrame with 'code' column
        
    Returns:
        DataFrame with feature columns
    """
    features_list = []
    for code in df['code']:
        features_list.append(extract_all_features(code))
    
    return pd.DataFrame(features_list)
