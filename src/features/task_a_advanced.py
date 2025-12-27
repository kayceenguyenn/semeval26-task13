"""
Advanced features for Task A (Human vs AI detection)

Author: [Your Name]
Difficulty: ⭐⭐ Intermediate
Description: Features specifically designed to distinguish human-written from AI-generated code
"""

import ast
import re
from typing import Dict


def extract_task_a_advanced_features(code: str) -> dict:
    """
    Extract advanced features that help distinguish human vs AI code
    
    Args:
        code: Source code string
        
    Returns:
        dict: Feature name -> value mapping
    """
    features = {}
    
    # 1. STYLISTIC FEATURES (High Impact)
    features.update(_extract_naming_style_features(code))
    features.update(_extract_indentation_features(code))
    features.update(_extract_whitespace_consistency(code))
    
    # 2. CODE QUALITY FEATURES (High Impact)
    features.update(_extract_error_handling_features(code))
    features.update(_extract_docstring_features(code))
    features.update(_extract_type_hint_features(code))
    
    # 3. PATTERN FEATURES (Medium-High Impact)
    features.update(_extract_repetition_features(code))
    features.update(_extract_import_pattern_features(code))
    features.update(_extract_variable_usage_features(code))
    
    # 4. COMPLEXITY FEATURES (Medium Impact)
    features.update(_extract_cyclomatic_complexity(code))
    features.update(_extract_function_length_features(code))
    
    # 5. LANGUAGE-SPECIFIC FEATURES (Medium Impact)
    features.update(_extract_python_specific_features(code))
    
    return features


# ============================================================================
# 1. STYLISTIC FEATURES
# ============================================================================

def _extract_naming_style_features(code: str) -> Dict[str, float]:
    """Extract variable/function naming style features"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Variable names
        variable_names = []
        function_names = []
        class_names = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variable_names.append(node.id)
            elif isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                class_names.append(node.name)
        
        # Naming conventions
        snake_case_vars = sum(1 for v in variable_names if '_' in v and v.islower())
        camel_case_vars = sum(1 for v in variable_names if v and v[0].islower() and any(c.isupper() for c in v[1:]))
        all_caps_vars = sum(1 for v in variable_names if v.isupper() and '_' in v)
        
        # Function naming
        snake_case_funcs = sum(1 for f in function_names if '_' in f)
        camel_case_funcs = sum(1 for f in function_names if f and f[0].islower() and any(c.isupper() for c in f[1:]))
        
        # Class naming
        pascal_case_classes = sum(1 for c in class_names if c and c[0].isupper())
        
        features['naming_snake_case_ratio'] = snake_case_vars / max(len(variable_names), 1)
        features['naming_camel_case_ratio'] = camel_case_vars / max(len(variable_names), 1)
        features['naming_all_caps_ratio'] = all_caps_vars / max(len(variable_names), 1)
        features['func_snake_case_ratio'] = snake_case_funcs / max(len(function_names), 1)
        features['class_pascal_case_ratio'] = pascal_case_classes / max(len(class_names), 1)
        
        # Average name length
        if variable_names:
            features['avg_variable_name_length'] = sum(len(v) for v in variable_names) / len(variable_names)
        else:
            features['avg_variable_name_length'] = 0
        
        if function_names:
            features['avg_function_name_length'] = sum(len(f) for f in function_names) / len(function_names)
        else:
            features['avg_function_name_length'] = 0
        
    except:
        features.update({
            'naming_snake_case_ratio': 0,
            'naming_camel_case_ratio': 0,
            'naming_all_caps_ratio': 0,
            'func_snake_case_ratio': 0,
            'class_pascal_case_ratio': 0,
            'avg_variable_name_length': 0,
            'avg_function_name_length': 0,
        })
    
    return features


def _extract_indentation_features(code: str) -> Dict[str, float]:
    """Extract indentation style features"""
    features = {}
    
    lines = code.split('\n')
    indent_sizes = []
    uses_tabs = 0
    uses_spaces = 0
    
    for line in lines:
        if line.strip():  # Non-empty line
            leading = len(line) - len(line.lstrip())
            if leading > 0:
                indent_sizes.append(leading)
                if line[0] == '\t':
                    uses_tabs += 1
                elif line[0] == ' ':
                    uses_spaces += 1
    
    # Indentation consistency
    if indent_sizes:
        # Check if indentation is consistent (usually 2, 4, or 8 spaces)
        indent_set = set(indent_sizes)
        most_common_indent = max(set(indent_sizes), key=indent_sizes.count) if indent_sizes else 0
        indent_consistency = indent_sizes.count(most_common_indent) / len(indent_sizes)
        
        features['indent_consistency'] = indent_consistency
        features['most_common_indent_size'] = most_common_indent
        features['indent_variance'] = len(indent_set) / max(len(indent_sizes), 1)
    else:
        features['indent_consistency'] = 1.0
        features['most_common_indent_size'] = 0
        features['indent_variance'] = 0
    
    features['uses_tabs'] = 1 if uses_tabs > 0 else 0
    features['uses_spaces'] = 1 if uses_spaces > 0 else 0
    features['mixed_indentation'] = 1 if uses_tabs > 0 and uses_spaces > 0 else 0
    
    return features


def _extract_whitespace_consistency(code: str) -> Dict[str, float]:
    """Extract whitespace usage consistency"""
    features = {}
    
    lines = code.split('\n')
    
    # Trailing whitespace
    trailing_whitespace_lines = sum(1 for line in lines if line.rstrip() != line)
    features['trailing_whitespace_ratio'] = trailing_whitespace_lines / max(len(lines), 1)
    
    # Blank lines
    blank_lines = sum(1 for line in lines if not line.strip())
    features['blank_line_ratio'] = blank_lines / max(len(lines), 1)
    
    # Multiple blank lines (often indicates human editing)
    consecutive_blanks = 0
    max_consecutive_blanks = 0
    for line in lines:
        if not line.strip():
            consecutive_blanks += 1
            max_consecutive_blanks = max(max_consecutive_blanks, consecutive_blanks)
        else:
            consecutive_blanks = 0
    
    features['max_consecutive_blank_lines'] = max_consecutive_blanks
    
    return features


# ============================================================================
# 2. CODE QUALITY FEATURES
# ============================================================================

def _extract_error_handling_features(code: str) -> Dict[str, float]:
    """Extract error handling patterns"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Count different error handling patterns
        try_except = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Try))
        try_finally = sum(1 for node in ast.walk(tree) 
                         if isinstance(node, ast.Try) and node.finalbody)
        with_statements = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.With))
        
        # Exception types
        exception_types = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type:
                    if isinstance(node.type, ast.Name):
                        exception_types.append(node.type.id)
                    elif isinstance(node.type, ast.Tuple):
                        exception_types.extend([el.id for el in node.type.elts if isinstance(el, ast.Name)])
        
        features['num_try_except'] = try_except
        features['num_try_finally'] = try_finally
        features['num_with_statements'] = with_statements
        features['num_exception_types'] = len(set(exception_types))
        features['uses_bare_except'] = sum(1 for node in ast.walk(tree) 
                                           if isinstance(node, ast.ExceptHandler) and node.type is None)
        
    except:
        features.update({
            'num_try_except': 0,
            'num_try_finally': 0,
            'num_with_statements': 0,
            'num_exception_types': 0,
            'uses_bare_except': 0,
        })
    
    return features


def _extract_docstring_features(code: str) -> Dict[str, float]:
    """Extract docstring patterns"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Function/class docstrings
        functions_with_docstrings = 0
        classes_with_docstrings = 0
        total_docstring_length = 0
        docstring_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    if isinstance(node, ast.FunctionDef):
                        functions_with_docstrings += 1
                    else:
                        classes_with_docstrings += 1
                    total_docstring_length += len(docstring)
                    docstring_count += 1
        
        total_functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        total_classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
        
        features['func_docstring_ratio'] = functions_with_docstrings / max(total_functions, 1)
        features['class_docstring_ratio'] = classes_with_docstrings / max(total_classes, 1)
        features['avg_docstring_length'] = total_docstring_length / max(docstring_count, 1)
        
    except:
        features.update({
            'func_docstring_ratio': 0,
            'class_docstring_ratio': 0,
            'avg_docstring_length': 0,
        })
    
    return features


def _extract_type_hint_features(code: str) -> Dict[str, float]:
    """Extract type hint usage"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Count type hints
        functions_with_type_hints = 0
        parameters_with_type_hints = 0
        total_parameters = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_return_hint = node.returns is not None
                func_has_hints = has_return_hint
                
                for arg in node.args.args:
                    total_parameters += 1
                    if arg.annotation:
                        parameters_with_type_hints += 1
                        func_has_hints = True
                
                if func_has_hints:
                    functions_with_type_hints += 1
        
        total_functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        
        features['func_type_hint_ratio'] = functions_with_type_hints / max(total_functions, 1)
        features['param_type_hint_ratio'] = parameters_with_type_hints / max(total_parameters, 1)
        
    except:
        features.update({
            'func_type_hint_ratio': 0,
            'param_type_hint_ratio': 0,
        })
    
    return features


# ============================================================================
# 3. PATTERN FEATURES
# ============================================================================

def _extract_repetition_features(code: str) -> Dict[str, float]:
    """Extract code repetition patterns (AI often has more repetition)"""
    features = {}
    
    lines = code.split('\n')
    
    # Exact duplicate lines
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    duplicate_lines = sum(1 for count in line_counts.values() if count > 1)
    features['duplicate_line_ratio'] = duplicate_lines / max(len(set(lines)), 1)
    
    # Similar lines (fuzzy matching - lines that are very similar)
    similar_pairs = 0
    unique_lines = [l.strip() for l in lines if l.strip()]
    for i, line1 in enumerate(unique_lines):
        for line2 in unique_lines[i+1:]:
            # Simple similarity: check if one line is substring of another or very similar length
            if len(line1) > 10 and len(line2) > 10:
                if line1 in line2 or line2 in line1:
                    similar_pairs += 1
                    break
    
    features['similar_line_ratio'] = similar_pairs / max(len(unique_lines), 1)
    
    return features


def _extract_import_pattern_features(code: str) -> Dict[str, float]:
    """Extract import statement patterns"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Import types
        import_statements = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Import))
        from_imports = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ImportFrom))
        
        # Import aliases
        aliased_imports = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliased_imports += sum(1 for alias in node.names if alias.asname)
            elif isinstance(node, ast.ImportFrom):
                aliased_imports += sum(1 for alias in node.names if alias.asname)
        
        # Wildcard imports
        wildcard_imports = sum(1 for node in ast.walk(tree) 
                              if isinstance(node, ast.ImportFrom) and 
                              any(alias.name == '*' for alias in node.names))
        
        total_imports = import_statements + from_imports
        
        features['import_count'] = total_imports
        features['from_import_ratio'] = from_imports / max(total_imports, 1)
        features['aliased_import_ratio'] = aliased_imports / max(total_imports, 1)
        features['wildcard_import_count'] = wildcard_imports
        
    except:
        features.update({
            'import_count': 0,
            'from_import_ratio': 0,
            'aliased_import_ratio': 0,
            'wildcard_import_count': 0,
        })
    
    return features


def _extract_variable_usage_features(code: str) -> Dict[str, float]:
    """Extract variable usage patterns"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Variable assignments vs usages
        assignments = []
        usages = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assignments.append(node.id)
                elif isinstance(node.ctx, ast.Load):
                    usages.append(node.id)
        
        # Unused variables (assigned but never used)
        assigned_set = set(assignments)
        used_set = set(usages)
        unused_vars = assigned_set - used_set
        
        features['unused_variable_ratio'] = len(unused_vars) / max(len(assigned_set), 1)
        features['variable_reuse_ratio'] = len(used_set) / max(len(assigned_set), 1)
        
    except:
        features.update({
            'unused_variable_ratio': 0,
            'variable_reuse_ratio': 0,
        })
    
    return features


# ============================================================================
# 4. COMPLEXITY FEATURES
# ============================================================================

def _extract_cyclomatic_complexity(code: str) -> Dict[str, float]:
    """Extract cyclomatic complexity metrics"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        # Count decision points
        decision_points = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, 
                               ast.With, ast.And, ast.Or)):
                decision_points += 1
        
        features['cyclomatic_complexity'] = decision_points + 1  # Base complexity
        
    except:
        features['cyclomatic_complexity'] = 1
    
    return features


def _extract_function_length_features(code: str) -> Dict[str, float]:
    """Extract function length statistics"""
    features = {}
    
    try:
        tree = ast.parse(code)
        
        function_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = len(ast.get_source_segment(code, node).split('\n'))
                function_lengths.append(func_lines)
        
        if function_lengths:
            features['avg_function_length'] = sum(function_lengths) / len(function_lengths)
            features['max_function_length'] = max(function_lengths)
            features['min_function_length'] = min(function_lengths)
        else:
            features['avg_function_length'] = 0
            features['max_function_length'] = 0
            features['min_function_length'] = 0
        
    except:
        features.update({
            'avg_function_length': 0,
            'max_function_length': 0,
            'min_function_length': 0,
        })
    
    return features


# ============================================================================
# 5. LANGUAGE-SPECIFIC FEATURES
# ============================================================================

def _extract_python_specific_features(code: str) -> Dict[str, float]:
    """Extract Python-specific patterns"""
    features = {}
    
    # List/dict comprehensions (more common in human code)
    list_comp_count = len(re.findall(r'\[.*?for.*?in.*?\]', code))
    dict_comp_count = len(re.findall(r'\{.*?for.*?in.*?\}', code))
    
    features['list_comprehension_count'] = list_comp_count
    features['dict_comprehension_count'] = dict_comp_count
    
    # Lambda usage
    lambda_count = code.count('lambda ')
    features['lambda_count'] = lambda_count
    
    # Decorators
    decorator_count = code.count('@')
    features['decorator_count'] = decorator_count
    
    # F-strings (modern Python)
    fstring_count = len(re.findall(r'f["\']', code))
    features['fstring_count'] = fstring_count
    
    # Walrus operator (Python 3.8+)
    walrus_count = code.count(':=')
    features['walrus_operator_count'] = walrus_count
    
    return features

