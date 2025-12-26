"""
AST-based features for code classification

Author: Task B Improvement
Difficulty: ⭐⭐ Intermediate
Description: Extract structural features from Python code using AST parsing

OPTIMIZED: Single-pass AST visitor (5-6x faster than multiple ast.walk() calls)
"""

import ast
from typing import Dict


class ASTFeatureVisitor(ast.NodeVisitor):
    """
    Single-pass AST visitor that collects all features at once.

    Instead of walking the tree 6-7 times with separate functions,
    this collects everything in one traversal for ~5-6x speedup.
    """

    def __init__(self):
        self.features = {
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
        self.current_depth = 0
        self.max_depth = 0
        self.total_func_args = 0
        self.num_funcs = 0
        self.module_body = None  # Track module body for global vars

    def visit(self, node):
        """Override visit to track depth"""
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        result = super().visit(node)
        self.current_depth -= 1
        return result

    def visit_Module(self, node):
        """Process module - check for module docstring and global vars"""
        self.module_body = node.body

        # Module docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.features['ast_num_docstrings'] += 1

        # Count global variables (module-level assignments)
        for child in node.body:
            if isinstance(child, ast.Assign):
                self.features['ast_num_global_vars'] += len(child.targets)
            elif isinstance(child, ast.AnnAssign):
                self.features['ast_num_global_vars'] += 1

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Process function definition"""
        self.features['ast_num_functions'] += 1
        self.features['ast_num_decorators'] += len(node.decorator_list)

        # Check for docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.features['ast_num_docstrings'] += 1

        # Count type hints
        if node.returns:
            self.features['ast_num_type_hints'] += 1
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.annotation:
                self.features['ast_num_type_hints'] += 1

        # Count arguments
        args = node.args
        num_args = len(args.args) + len(args.kwonlyargs)
        if args.vararg:
            num_args += 1
        if args.kwarg:
            num_args += 1
        self.total_func_args += num_args
        self.num_funcs += 1

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Process async function - same as regular function"""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Process class definition"""
        self.features['ast_num_classes'] += 1
        self.features['ast_num_decorators'] += len(node.decorator_list)

        # Check for docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.features['ast_num_docstrings'] += 1

        self.generic_visit(node)

    def visit_Import(self, node):
        """Process import statement"""
        self.features['ast_num_imports'] += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Process from...import statement"""
        self.features['ast_num_imports'] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """Process for loop"""
        self.features['ast_num_loops'] += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        """Process async for loop"""
        self.features['ast_num_loops'] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        """Process while loop"""
        self.features['ast_num_loops'] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        """Process if statement - also check for main guard"""
        self.features['ast_num_conditionals'] += 1

        # Check for if __name__ == "__main__" pattern
        try:
            test = node.test
            if isinstance(test, ast.Compare):
                if (isinstance(test.left, ast.Name) and
                    test.left.id == '__name__' and
                    len(test.comparators) == 1 and
                    isinstance(test.comparators[0], ast.Constant) and
                    test.comparators[0].value == '__main__'):
                    self.features['ast_has_main_guard'] = 1
        except AttributeError:
            pass

        self.generic_visit(node)

    def visit_Try(self, node):
        """Process try/except block"""
        self.features['ast_num_try_except'] += 1
        self.generic_visit(node)

    def visit_With(self, node):
        """Process with statement"""
        self.features['ast_num_with'] += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        """Process async with statement"""
        self.features['ast_num_with'] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        """Process list comprehension"""
        self.features['ast_num_comprehensions'] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        """Process dict comprehension"""
        self.features['ast_num_comprehensions'] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        """Process set comprehension"""
        self.features['ast_num_comprehensions'] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        """Process generator expression"""
        self.features['ast_num_comprehensions'] += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        """Process assert statement"""
        self.features['ast_num_assertions'] += 1
        self.generic_visit(node)

    def visit_Yield(self, node):
        """Process yield expression"""
        self.features['ast_num_yields'] += 1
        self.generic_visit(node)

    def visit_YieldFrom(self, node):
        """Process yield from expression"""
        self.features['ast_num_yields'] += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        """Process lambda expression"""
        self.features['ast_num_lambda'] += 1
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        """Process f-string"""
        self.features['ast_num_f_strings'] += 1
        self.generic_visit(node)

    def visit_Constant(self, node):
        """Process constant - count string literals"""
        if isinstance(node.value, str):
            self.features['ast_num_string_literals'] += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Process annotated assignment (type hint)"""
        self.features['ast_num_type_hints'] += 1
        self.generic_visit(node)

    def get_features(self) -> Dict[str, float]:
        """Return final features with computed values"""
        self.features['ast_max_depth'] = self.max_depth
        self.features['ast_avg_func_args'] = (
            self.total_func_args / self.num_funcs if self.num_funcs > 0 else 0.0
        )
        return self.features


def extract_ast_features(code: str) -> dict:
    """
    Extract AST-based features from Python code

    Falls back to zero values if parsing fails (e.g., non-Python code or syntax errors).

    OPTIMIZED: Uses single-pass AST visitor (5-6x faster than multiple ast.walk() calls)

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
        visitor = ASTFeatureVisitor()
        visitor.visit(tree)
        return visitor.get_features()
    except SyntaxError:
        return default_features
