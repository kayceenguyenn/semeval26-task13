# 🤖 Adding Your Model (Zero Conflicts!)

## Quick Start

**Create your own model file** - No merge conflicts!

```bash
# 1. Copy template
cp src/models/_template.py src/models/alice_xgboost.py

# 2. Implement your model
# Edit alice_xgboost.py

# 3. Register in __init__.py
# Add import and elif clause
```

---

## Step-by-Step Example

### Step 1: Create Your File

```bash
cp src/models/_template.py src/models/alice_xgboost.py
```

### Step 2: Implement Your Model

```python
# src/models/alice_xgboost.py

"""
Alice's XGBoost model

Author: Alice
Difficulty: ⭐⭐⭐ Advanced
"""

from xgboost import XGBClassifier

class AliceXGBoostModel:
    def __init__(self, n_estimators=100, max_depth=5):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.is_trained = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

### Step 3: Register Your Model

Edit `src/models/__init__.py`:

```python
# Add import
from .alice_xgboost import AliceXGBoostModel

# Add to get_model function
def get_model(model_type: str, **kwargs):
    if model_type in ['logistic_regression', 'random_forest', 'gradient_boosting']:
        return BaselineModel(model_type=model_type)
    elif model_type == 'alice_xgboost':  # ← Add this
        return AliceXGBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### Step 4: Use Your Model

```python
from src.models import get_model

model = get_model('alice_xgboost', n_estimators=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Why This Works

✅ **No conflicts** - Each student has their own file  
✅ **Easy to add** - Copy template, implement  
✅ **Easy to test** - Test independently  
✅ **Clear ownership** - Your name in filename  

---

## Examples

### XGBoost
```python
# src/models/bob_xgboost.py
from xgboost import XGBClassifier

class BobXGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(random_state=42)
```

### LightGBM
```python
# src/models/carol_lightgbm.py
from lightgbm import LGBMClassifier

class CarolLightGBMModel:
    def __init__(self):
        self.model = LGBMClassifier(random_state=42)
```

### Neural Network
```python
# src/models/dave_neural.py
from sklearn.neural_network import MLPClassifier

class DaveNeuralModel:
    def __init__(self):
        self.model = MLPClassifier(random_state=42)
```

---

**No merge conflicts!** Everyone has their own file.
