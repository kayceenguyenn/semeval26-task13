"""
Kaycee's features

Author: Kaycee
Description: AST features plus word length, keyword frequency, and comment pattern features
"""
import ast
import re

def extract_kayceeast_features(code: str) -> dict:
    """
    Extract all features from code including AST, word length, keywords, and comments
    
    Args:
        code: Source code string
        
    Returns:
        dict: Feature name -> value mapping
    """
    features = {}
    
    # AST features
    try:
        tree = ast.parse(code)
        features['num_functions'] = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        features['max_nesting'] = calculate_max_depth(tree)
        features['num_loops'] = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.For, ast.While)))
    except:
        features['num_functions'] = 0
        features['max_nesting'] = 0
        features['num_loops'] = 0
    
    # Word length feature (word count per code sample)
    features['word_count'] = len(code.split())
    
    # Keyword frequency feature (total count of def, class, if, for)
    keywords = ['def', 'class', 'if', 'for']
    total_keyword_freq = 0
    for keyword in keywords:
        # Use word boundaries to avoid partial matches
        total_keyword_freq += len(re.findall(r'\b' + re.escape(keyword) + r'\b', code))
    features['total_keyword_freq'] = total_keyword_freq
    
    # Comment features
    num_comments = count_comments(code)
    avg_comment_length = avg_comment_length_words(code)
    features['num_comments'] = num_comments
    features['avg_comment_length'] = avg_comment_length
    
    return features

# Helper functions

def calculate_max_depth(node, depth=0):
    """
    Calculate maximum nesting depth in AST
    
    Args:
        node: AST node
        depth: Current depth
        
    Returns:
        int: Maximum nesting depth
    """
    if not isinstance(node, ast.AST):
        return depth
    
    max_depth = depth
    for child in ast.iter_child_nodes(node):
        # Increment depth for control flow structures
        if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, 
                             ast.While, ast.Try, ast.With)):
            child_depth = calculate_max_depth(child, depth + 1)
        else:
            child_depth = calculate_max_depth(child, depth)
        max_depth = max(max_depth, child_depth)
    
    return max_depth


def count_comments(code: str) -> int:
    """
    Count the number of comment blocks in code.
    Groups consecutive comment lines into a single comment block.
    
    Args:
        code: Source code string
        
    Returns:
        int: Number of comment blocks
    """
    lines = code.split('\n')
    comment_count = 0
    in_comment_block = False
    
    for line in lines:
        is_comment_line = line.strip().startswith('#')
        if is_comment_line and not in_comment_block:
            comment_count += 1
            in_comment_block = True
        elif not is_comment_line:
            # End of comment block
            in_comment_block = False
    
    return comment_count


def avg_comment_length_words(code: str) -> float:
    """
    Calculate average comment length (in words) for each code sample.
    Groups consecutive comment lines into comment blocks.
    
    Args:
        code: Source code string
        
    Returns:
        float: Average word count per comment block (0 if no comments)
    """
    lines = code.split('\n')
    comment_blocks = []
    current_comment = []
    
    for line in lines:
        if line.strip().startswith('#'):
            # Remove # and leading whitespace
            current_comment.append(line.strip().lstrip('#'))
        else:
            if current_comment:
                comment_blocks.append(' '.join(current_comment))
                current_comment = []
    
    # Handle comment at end of file
    if current_comment:
        comment_blocks.append(' '.join(current_comment))
    
    if len(comment_blocks) == 0:
        return 0.0
    
    # Calculate word count for each comment block
    comment_word_counts = [len(comment.split()) for comment in comment_blocks]
    return sum(comment_word_counts) / len(comment_word_counts) if comment_word_counts else 0.0
