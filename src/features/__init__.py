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
# TODO: Uncomment when ast_features.py is populated
from .task_a_advanced import extract_task_a_advanced_features
# Add your feature module here:
from .kaycee_ast import extract_kayceeast_features

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
    
    # TODO: Uncomment when ast_features.py is populated
    # AST features (structural analysis)
    features.update(extract_task_a_advanced_features(code))
    
    # Add your features here:
    features.update(extract_kayceeast_features(code))
    
    return features


def extract_features_from_dataframe(df: pd.DataFrame, show_progress: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    """
    Extract features from a dataframe with 'code' column
    
    Uses parallel processing to speed up extraction (2-8x faster).
    
    Args:
        df: DataFrame with 'code' column
        show_progress: Whether to show progress updates
        n_jobs: Number of parallel jobs (-1 = use all CPUs, 1 = sequential)
        
    Returns:
        DataFrame with feature columns
    """
    total = len(df)
    
    # Use parallel processing for large datasets
    if n_jobs != 1 and total > 1000:
        try:
            from joblib import Parallel, delayed
            import multiprocessing
            
            # Determine number of jobs
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            if show_progress:
                print(f"🚀 Using {n_jobs} parallel workers for faster extraction...")
            
            # Parallel feature extraction
            features_list = Parallel(n_jobs=n_jobs, verbose=1 if show_progress else 0)(
                delayed(extract_all_features)(code) 
                for code in df['code']
            )
            
            if show_progress:
                print(f"✅ Processed {total:,}/{total:,} samples (100.0%)")
            
            return pd.DataFrame(features_list)
            
        except ImportError:
            if show_progress:
                print("⚠️  joblib not available, falling back to sequential processing")
                print("   Install with: pip install joblib")
            # Fall through to sequential processing
        except Exception as e:
            if show_progress:
                print(f"⚠️  Parallel processing failed: {e}")
                print("   Falling back to sequential processing")
            # Fall through to sequential processing
    
    # Sequential processing (fallback or for small datasets)
    features_list = []
    
    for idx, code in enumerate(df['code']):
        features_list.append(extract_all_features(code))
        
        # Show progress every 10,000 samples
        if show_progress and (idx + 1) % 10000 == 0:
            progress = (idx + 1) / total * 100
            print(f"  Processed {idx + 1:,}/{total:,} samples ({progress:.1f}%)", end='\r')
    
    if show_progress:
        print(f"  Processed {total:,}/{total:,} samples (100.0%)")
    
    return pd.DataFrame(features_list)

