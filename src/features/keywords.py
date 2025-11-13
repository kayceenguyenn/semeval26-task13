"""
Keyword-based features

Author: Baseline
Difficulty: ⭐ Beginner
"""


def extract_keyword_features(code: str) -> dict:
    """
    Count Python keywords in code
    
    Args:
        code: Source code string
        
    Returns:
        dict: Keyword counts
    """
    keywords = ['def', 'class', 'if', 'for', 'while', 'import', 'from', 'return', 'try', 'except']
    
    result = {}
    for keyword in keywords:
        result[f'kw_{keyword}'] = code.count(keyword)
    
    return result
