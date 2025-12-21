"""
Kaycee's features

Author: Kaycee
Difficulty: ⭐ / ⭐⭐ / ⭐⭐⭐
Description: Brief description of what these features capture
"""
import ast

def extract_kayceeast_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
        features = {
            'num_functions': sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef)),
            'max_nesting': calculate_max_depth(tree),
            'num_loops': sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.For, ast.While)))
        }
        return features
    except:
        return {}

def extract_[yourname]_features(code: str) -> dict:
    """
    Extract [description] features from code
    
    Args:
        code: Source code string
        
    Returns:
        dict: Feature name -> value mapping
    """
    features = {}
    
    # TODO: Add your feature extraction logic here
    # Example:
    # features['my_feature_1'] = calculate_something(code)
    # features['my_feature_2'] = count_something(code)
    
    return features


# Helper functions (optional)
def calculate_something(code: str) -> float:
    """Helper function for feature calculation"""
    # Your logic here
    return 0.0


def count_something(code: str) -> int:
    """Helper function for counting"""
    # Your logic here
    return 0
