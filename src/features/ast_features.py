"""
AST-based features for code classification

Author: Task B Improvement
Difficulty: PP Intermediate
Description: Extract structural features from Python code using AST parsing
"""

import ast
from typing import Dict, Any


def extract_ast_features(code: str) -> dict:
    """
    Extract AST-based features from Python code

    Falls back to zero values if parsing fails (e.g., non-Python code or syntax errors).

    Args:
        code: Source code string

    Returns:
        dict: Feature name -> value mapping
    """
    # Default features for when parsing fails
    default_features = {
        'ast_num_functions': 0,
        'ast_num_classes': 0,
        'ast_num_imports': 0,
        'ast_num_loops': 0,
        'ast_num_conditionals': 0,
        'ast_num_try_except': 0,
        'ast_num_with': 0,
        'ast_num_comprehensions': 0,
        'ast_num_decorators': 0,
        'ast_num_assertions': 0,
        'ast_num_yields': 0,
        'ast_num_lambda': 0,
        'ast_max_depth': 0,
        'ast_num_docstrings': 0,
        'ast_num_f_strings': 0,
        'ast_num_type_hints': 0,
        'ast_avg_func_args': 0.0,
        'ast_has_main_guard': 0,
        'ast_num_global_vars': 0,
        'ast_num_string_literals': 0,
    }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return default_features

    features = {}

    # Count node types
    node_counts = count_node_types(tree)
    features['ast_num_functions'] = node_counts.get('FunctionDef', 0) + node_counts.get('AsyncFunctionDef', 0)
    features['ast_num_classes'] = node_counts.get('ClassDef', 0)
    features['ast_num_imports'] = node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)
    features['ast_num_loops'] = node_counts.get('For', 0) + node_counts.get('While', 0) + \
                                 node_counts.get('AsyncFor', 0)
    features['ast_num_conditionals'] = node_counts.get('If', 0)
    features['ast_num_try_except'] = node_counts.get('Try', 0)
    features['ast_num_with'] = node_counts.get('With', 0) + node_counts.get('AsyncWith', 0)
    features['ast_num_comprehensions'] = node_counts.get('ListComp', 0) + \
                                          node_counts.get('DictComp', 0) + \
                                          node_counts.get('SetComp', 0) + \
                                          node_counts.get('GeneratorExp', 0)
    features['ast_num_assertions'] = node_counts.get('Assert', 0)
    features['ast_num_yields'] = node_counts.get('Yield', 0) + node_counts.get('YieldFrom', 0)
    features['ast_num_lambda'] = node_counts.get('Lambda', 0)

    # Count decorators
    features['ast_num_decorators'] = count_decorators(tree)

    # Calculate max nesting depth
    features['ast_max_depth'] = calculate_max_depth(tree)

    # Count docstrings
    features['ast_num_docstrings'] = count_docstrings(tree)

    # Count f-strings (Python 3.6+)
    features['ast_num_f_strings'] = node_counts.get('JoinedStr', 0)

    # Count type hints
    features['ast_num_type_hints'] = count_type_hints(tree)

    # Average function arguments
    features['ast_avg_func_args'] = calculate_avg_func_args(tree)

    # Check for if __name__ == "__main__" guard
    features['ast_has_main_guard'] = has_main_guard(tree)

    # Count global variable assignments (module level)
    features['ast_num_global_vars'] = count_global_vars(tree)

    # Count string literals
    features['ast_num_string_literals'] = node_counts.get('Constant', 0)

    return features


def count_node_types(tree: ast.AST) -> Dict[str, int]:
    """Count occurrences of each AST node type"""
    counts: Dict[str, int] = {}
    for node in ast.walk(tree):
        node_type = type(node).__name__
        counts[node_type] = counts.get(node_type, 0) + 1
    return counts


def count_decorators(tree: ast.AST) -> int:
    """Count total decorators on functions and classes"""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            count += len(node.decorator_list)
    return count


def calculate_max_depth(tree: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth of the AST"""
    max_depth = current_depth

    for node in ast.iter_child_nodes(tree):
        child_depth = calculate_max_depth(node, current_depth + 1)
        max_depth = max(max_depth, child_depth)

    return max_depth


def count_docstrings(tree: ast.AST) -> int:
    """Count docstrings in module, classes, and functions"""
    count = 0

    # Check module docstring
    if (tree.body and isinstance(tree.body[0], ast.Expr) and
        isinstance(tree.body[0].value, ast.Constant) and
        isinstance(tree.body[0].value.value, str)):
        count += 1

    # Check function and class docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                count += 1

    return count


def count_type_hints(tree: ast.AST) -> int:
    """Count type annotations in function signatures and assignments"""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Return type annotation
            if node.returns:
                count += 1
            # Argument annotations
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.annotation:
                    count += 1
        elif isinstance(node, ast.AnnAssign):
            # Annotated assignment
            count += 1
    return count


def calculate_avg_func_args(tree: ast.AST) -> float:
    """Calculate average number of arguments per function"""
    total_args = 0
    num_funcs = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            num_args = len(args.args) + len(args.kwonlyargs)
            if args.vararg:
                num_args += 1
            if args.kwarg:
                num_args += 1
            total_args += num_args
            num_funcs += 1

    return total_args / num_funcs if num_funcs > 0 else 0.0


def has_main_guard(tree: ast.AST) -> int:
    """Check if code has if __name__ == "__main__" guard"""
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check for __name__ == "__main__" pattern
            try:
                test = node.test
                if isinstance(test, ast.Compare):
                    if (isinstance(test.left, ast.Name) and
                        test.left.id == '__name__' and
                        len(test.comparators) == 1 and
                        isinstance(test.comparators[0], ast.Constant) and
                        test.comparators[0].value == '__main__'):
                        return 1
            except AttributeError:
                pass
    return 0


def count_global_vars(tree: ast.AST) -> int:
    """Count module-level variable assignments"""
    count = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            count += len(node.targets)
        elif isinstance(node, ast.AnnAssign):
            count += 1
    return count
