# 🎯 Adding Your Features (Zero Conflicts!)

## Quick Start

**Create your own feature file** - No merge conflicts!

```bash
# 1. Create your feature file
cp src/features/_template.py src/features/alice_features.py

# 2. Edit your file
# Add your feature extraction functions

# 3. Register in __init__.py
# Add one import line and one function call
```

---

## Step-by-Step Example

### Step 1: Create Your File

```bash
# Copy template
cp src/features/_template.py src/features/alice_features.py
```

### Step 2: Implement Your Features

```python
# src/features/alice_features.py

"""
Alice's awesome features

Author: Alice
Difficulty: ⭐⭐ Intermediate
"""

def extract_alice_features(code: str) -> dict:
    """My custom features"""
    return {
        'num_comments': code.count('#'),
        'has_docstring': '"""' in code or "'''" in code,
        'num_functions': code.count('def '),
    }
```

### Step 3: Register Your Features

Edit `src/features/__init__.py`:

```python
# Add your import (one line)
from .alice_features import extract_alice_features

# Add to extract_all_features function
def extract_all_features(code: str) -> dict:
    features = {}
    features.update(extract_basic_features(code))
    features.update(extract_keyword_features(code))
    features.update(extract_alice_features(code))  # ← Add this line
    return features
```

---

## Why This Works

✅ **No conflicts** - Each student has their own file  
✅ **Easy to add** - Just create a new file  
✅ **Easy to remove** - Delete file + 2 lines in `__init__.py`  
✅ **Easy to test** - Test your file independently  
✅ **Clear ownership** - Your name in filename

---

## Feature File Template

See `_template.py` for a starter template.

---

## Examples by Difficulty

### ⭐ Beginner: Simple Counts

```python
def extract_simple_features(code: str) -> dict:
    return {
        'num_parentheses': code.count('('),
        'num_brackets': code.count('['),
        'num_equals': code.count('='),
    }
```

### ⭐⭐ Intermediate: AST Features

```python
import ast

def extract_ast_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
        return {
            'ast_depth': calculate_depth(tree),
            'num_functions': len([n for n in ast.walk(tree) 
                                  if isinstance(n, ast.FunctionDef)]),
        }
    except:
        return {'ast_depth': 0, 'num_functions': 0}
```

### ⭐⭐⭐ Advanced: Complexity Metrics

```python
from radon.complexity import cc_visit

def extract_complexity_features(code: str) -> dict:
    try:
        complexity = cc_visit(code)
        return {
            'cyclomatic_complexity': sum(c.complexity for c in complexity),
            'avg_complexity': sum(c.complexity for c in complexity) / max(len(complexity), 1),
        }
    except:
        return {'cyclomatic_complexity': 0, 'avg_complexity': 0}
```

---

## Testing Your Features

```python
# Test your features independently
from src.features.alice_features import extract_alice_features

code = "def hello():\n    print('hi')"
features = extract_alice_features(code)
print(features)
```

---

## Questions?

Ask in Slack `#data-features` channel!
