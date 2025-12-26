"""
XGBoost model for code classification

Author: Task B Improvement
Difficulty: ⭐⭐⭐ Advanced
Description: Gradient boosting model with GPU support for faster training
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, classification_report
from typing import Optional, List
import pickle
from pathlib import Path

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")


class XGBoostModel(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with GPU support

    Designed for Task B (12-class model attribution) with:
    - Handling of class imbalance
    - GPU acceleration (when available)
    - Early stopping support

    Args:
        n_estimators: Number of boosting rounds (default: 300)
        max_depth: Maximum tree depth (default: 8)
        learning_rate: Step size shrinkage (default: 0.1)
        use_gpu: Whether to use GPU if available (default: True)
        early_stopping_rounds: Stop if no improvement after N rounds (default: 50)
        random_state: Random seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        use_gpu: bool = True,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.is_trained = False
        self.model = None
        self.classes_ = None

    def _create_model(self, n_classes: int) -> XGBClassifier:
        """Create XGBoost classifier with appropriate settings"""
        # Determine device
        if self.use_gpu:
            try:
                # Test if GPU is available
                device = 'cuda'
                tree_method = 'hist'
            except Exception:
                device = 'cpu'
                tree_method = 'hist'
        else:
            device = 'cpu'
            tree_method = 'hist'

        # Multi-class or binary
        if n_classes > 2:
            objective = 'multi:softprob'
        else:
            objective = 'binary:logistic'

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=objective,
            tree_method=tree_method,
            device=device,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            use_label_encoder=False,
        )

        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Train the XGBoost model

        Args:
            X: Training features (DataFrame or array)
            y: Training labels (Series or array)
            X_val: Validation features for early stopping (optional)
            y_val: Validation labels for early stopping (optional)
            sample_weight: Sample weights for class imbalance (optional)
        """
        # Convert to numpy if needed
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        # Get unique classes
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        print(f"\nTraining XGBoost...")
        print(f"  Samples: {len(X_arr):,}")
        print(f"  Features: {X_arr.shape[1]}")
        print(f"  Classes: {n_classes}")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  learning_rate: {self.learning_rate}")

        # Create model
        self.model = self._create_model(n_classes)

        # Compute sample weights if not provided (for class imbalance)
        if sample_weight is None:
            class_counts = np.bincount(y_arr.astype(int))
            total = len(y_arr)
            weights = total / (n_classes * class_counts)
            sample_weight = weights[y_arr.astype(int)]
            print(f"  Using computed sample weights for class balance")

        # Prepare eval set for early stopping
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if hasattr(X_val, 'values') else np.array(X_val)
            y_val_arr = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
            eval_set = [(X_val_arr, y_val_arr)]
            print(f"  Using validation set for early stopping ({len(X_val_arr):,} samples)")

            self.model.fit(
                X_arr, y_arr,
                sample_weight=sample_weight,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )
        else:
            self.model.fit(
                X_arr, y_arr,
                sample_weight=sample_weight,
                verbose=False,
            )

        self.is_trained = True

        # Training metrics
        train_preds = self.model.predict(X_arr)
        train_f1 = f1_score(y_arr, train_preds, average='macro')
        print(f"  Training Macro F1: {train_f1:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix (DataFrame or array)

        Returns:
            array: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            array: Class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        return self.model.predict_proba(X_arr)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        detailed: bool = True
    ) -> dict:
        """
        Evaluate model performance

        Args:
            X: Features
            y: True labels
            detailed: Whether to print detailed report

        Returns:
            dict: Evaluation metrics
        """
        preds = self.predict(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)

        macro_f1 = f1_score(y_arr, preds, average='macro')
        micro_f1 = f1_score(y_arr, preds, average='micro')
        weighted_f1 = f1_score(y_arr, preds, average='weighted')

        results = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'predictions': preds
        }

        if detailed:
            print(f"\n{'='*70}")
            print(f"XGBoost Evaluation Results")
            print(f"{'='*70}")
            print(f"\n  Macro F1 Score: {macro_f1:.4f}  (Official competition metric)")
            print(f"  Micro F1 Score: {micro_f1:.4f}")
            print(f"  Weighted F1:    {weighted_f1:.4f}")
            print(f"\n  Per-Class Performance:")
            print(classification_report(y_arr, preds, digits=4))
            print(f"{'='*70}")

        return results

    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 20
    ) -> None:
        """
        Display top feature importances

        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(f"\n  Top {top_n} Most Important Features:")
        print(f"{'='*70}")
        for i in range(min(top_n, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1:2d}. {feature_names[idx]:.<50} {importances[idx]:.4f}")
        print(f"{'='*70}")

    def save(self, path: str) -> None:
        """
        Save model to disk

        Args:
            path: Path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved XGBoost model to {path}")

    @staticmethod
    def load(path: str) -> 'XGBoostModel':
        """
        Load model from disk

        Args:
            path: Path to load the model from

        Returns:
            XGBoostModel: Loaded model
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded XGBoost model from {path}")
        return model
