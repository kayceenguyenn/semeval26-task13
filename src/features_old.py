"""Feature extraction for code samples"""

import pandas as pd
from loguru import logger


def extract_features_from_dataframe(df: pd.DataFrame, code_column: str = 'code') -> pd.DataFrame:
    """Extract features from a dataframe of code samples"""
    logger.info(f"Extracting features from {len(df)} samples...")
    
    features_list = []
    for idx, code in enumerate(df[code_column]):
        features = extract_features_from_code(str(code))
        features_list.append(features)
        
        if (idx + 1) % 100 == 0:
            logger.debug(f"Processed {idx + 1}/{len(df)} samples")
    
    feature_df = pd.DataFrame(features_list)
    logger.info(f"Extracted {len(feature_df.columns)} features")
    return feature_df


def extract_features_from_code(code: str) -> dict:
    """Extract features from a single code sample"""
    features = {}
    
    # Basic length features
    features['code_length'] = len(code)
    features['num_lines'] = code.count('\n') + 1
    features['avg_line_length'] = len(code) / max(features['num_lines'], 1)
    
    # Character-level features
    features['num_spaces'] = code.count(' ')
    features['num_tabs'] = code.count('\t')
    features['num_newlines'] = code.count('\n')
    features['space_ratio'] = features['num_spaces'] / max(len(code), 1)
    
    # Python syntax features
    features['num_def'] = code.count('def ')
    features['num_class'] = code.count('class ')
    features['num_if'] = code.count('if ')
    features['num_for'] = code.count('for ')
    features['num_while'] = code.count('while ')
    features['num_return'] = code.count('return ')
    features['num_import'] = code.count('import ')
    
    # Punctuation features
    features['num_colons'] = code.count(':')
    features['num_semicolons'] = code.count(';')
    features['num_commas'] = code.count(',')
    features['num_periods'] = code.count('.')
    features['num_parens'] = code.count('(') + code.count(')')
    features['num_brackets'] = code.count('[') + code.count(']')
    features['num_braces'] = code.count('{') + code.count('}')
    
    # Comment features
    features['num_comments'] = code.count('#')
    features['num_docstrings'] = code.count('"""') + code.count("'''")
    
    # Naming convention features
    features['num_underscores'] = code.count('_')
    features['num_uppercase'] = sum(1 for c in code if c.isupper())
    features['num_lowercase'] = sum(1 for c in code if c.islower())
    features['num_digits'] = sum(1 for c in code if c.isdigit())
    
    # Complexity indicators
    features['max_line_length'] = max((len(line) for line in code.split('\n')), default=0)
    features['num_empty_lines'] = sum(1 for line in code.split('\n') if not line.strip())
    features['indentation_levels'] = max((len(line) - len(line.lstrip()) for line in code.split('\n')), default=0)
    
    return features
