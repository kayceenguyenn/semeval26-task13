"""Tests for features module"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import extract_all_features, extract_features_from_dataframe
import pandas as pd


def test_extract_features_from_code():
    """Test feature extraction from a single code sample"""
    code = "def hello():\n    print('Hello, World!')"
    features = extract_all_features(code)
    
    assert isinstance(features, dict)
    assert 'code_length' in features
    assert 'num_lines' in features
    assert 'kw_def' in features
    assert features['code_length'] == len(code)
    assert features['kw_def'] >= 1


def test_extract_features_from_dataframe():
    """Test feature extraction from a dataframe"""
    df = pd.DataFrame({
        'code': [
            "def func():\n    pass",
            "x = 5\ny = 10",
            "for i in range(10):\n    print(i)"
        ]
    })
    
    features_df = extract_features_from_dataframe(df)
    
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 3
    assert 'code_length' in features_df.columns
    assert 'num_lines' in features_df.columns


def test_feature_count():
    """Test that we extract the expected number of features"""
    code = "def test():\n    return 42"
    features = extract_all_features(code)
    
    # Should have 17 features (7 basic + 10 keywords)
    assert len(features) == 17


if __name__ == "__main__":
    print("Running features tests...")
    test_extract_features_from_code()
    print("✓ test_extract_features_from_code")
    
    test_extract_features_from_dataframe()
    print("✓ test_extract_features_from_dataframe")
    
    test_feature_count()
    print("✓ test_feature_count")
    
    print("\n✅ All features tests passed!")
