"""
Basic code features

Author: Baseline
Difficulty: ⭐ Beginner
"""


def extract_basic_features(code: str) -> dict:
    """
    Extract basic statistical features from code
    
    Args:
        code: Source code string
        
    Returns:
        dict: Basic features
    """
    lines = code.split('\n')
    
    return {
        'code_length': len(code),
        'num_lines': len(lines),
        'avg_line_length': len(code) / max(len(lines), 1),
        'num_spaces': code.count(' '),
        'num_tabs': code.count('\t'),
        'num_newlines': code.count('\n'),
        'space_ratio': code.count(' ') / max(len(code), 1),
    }
