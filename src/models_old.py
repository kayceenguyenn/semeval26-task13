"""
Baseline models for SemEval Task 13

⚠️ IMPORTANT: Performance Expectations
=====================================
These baseline models are designed for LEARNING, not competing.

Expected Performance (Task A):
- Logistic Regression:  50-55% Macro F1
- Random Forest:        55-60% Macro F1
- Gradient Boosting:    58-62% Macro F1

For competitive performance (85-95% F1), you'll need:
1. Transformer models (CodeBERT, GraphCodeBERT)
   → See notebooks/04_transformer_basics.ipynb
2. GPU access (Google Colab free tier works)
3. Fine-tuning on task-specific data

The baseline helps you:
✓ Understand the problem structure
✓ Learn feature engineering
✓ Iterate quickly (trains in seconds)
✓ Run on any laptop (no GPU needed)
✓ Build intuition before using transformers

Start here, then move to transformers for competition-grade results!
=====================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class BaselineModel:
    """
    Simple baseline model for code detection
    
    This is a starting point - you can improve it!
    
    Available models:
    - 'logistic_regression': Fast, interpretable (~50-55% F1)
    - 'random_forest': Balanced, robust (~55-60% F1)
    - 'gradient_boosting': Best baseline (~58-62% F1)
    
    Usage:
        >>> model = BaselineModel('random_forest')
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize baseline model
        
        Args:
            model_type (str): One of:
                - 'logistic_regression': Linear model, fast training
                - 'random_forest': Tree ensemble, most robust
                - 'gradient_boosting': Best baseline performance
        """
        self.model_type = model_type
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,  # Use all CPU cores
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Choose from: logistic_regression, random_forest, gradient_boosting"
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print(f"\nTraining {self.model_type}...")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training accuracy
        train_preds = self.model.predict(X_train)
        train_f1 = f1_score(y_train, train_preds, average='macro')
        print(f"  Training Macro F1: {train_f1:.4f}")
        
        # Performance expectation
        self._print_performance_expectation()
    
    def _print_performance_expectation(self):
        """Print expected performance based on model type"""
        expectations = {
            'logistic_regression': "50-55%",
            'random_forest': "55-60%",
            'gradient_boosting': "58-62%"
        }
        
        expected = expectations.get(self.model_type, "Unknown")
        print(f"\n  💡 Expected validation F1: {expected} (baseline model)")
        print(f"     For 85-95% F1, see notebooks/04_transformer_basics.ipynb")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Class probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, detailed: bool = True) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True labels
            detailed (bool): Whether to print detailed report
            
        Returns:
            dict: Evaluation metrics
        """
        preds = self.predict(X)
        
        # Calculate metrics
        macro_f1 = f1_score(y, preds, average='macro')
        micro_f1 = f1_score(y, preds, average='micro')
        weighted_f1 = f1_score(y, preds, average='weighted')
        
        results = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'predictions': preds
        }
        
        if detailed:
            print(f"\n{'='*70}")
            print(f"Evaluation Results - {self.model_type}")
            print(f"{'='*70}")
            print(f"\n🎯 Macro F1 Score: {macro_f1:.4f}  (Official competition metric)")
            print(f"   Micro F1 Score: {micro_f1:.4f}")
            print(f"   Weighted F1:    {weighted_f1:.4f}")
            
            # Classification report
            print(f"\n📊 Per-Class Performance:")
            print(classification_report(y, preds, digits=4))
            
            # Performance context
            self._print_performance_context(macro_f1)
        
        return results
    
    def _print_performance_context(self, achieved_f1: float):
        """Print context about achieved performance"""
        print(f"\n{'='*70}")
        print(f"Performance Context:")
        print(f"{'='*70}")
        
        if achieved_f1 < 0.65:
            print(f"📌 Current: {achieved_f1:.1%} F1 (Baseline)")
            print(f"📈 Next steps:")
            print(f"   1. Add AST features → Expected: ~70% F1")
            print(f"      See: notebooks/03_intermediate_features.ipynb")
            print(f"   2. Try transformers → Expected: ~88% F1")
            print(f"      See: notebooks/04_transformer_basics.ipynb")
        elif achieved_f1 < 0.75:
            print(f"📌 Current: {achieved_f1:.1%} F1 (Good for baseline!)")
            print(f"📈 To reach 85-90% F1:")
            print(f"   → Move to transformers (CodeBERT, GraphCodeBERT)")
            print(f"   → See: notebooks/04_transformer_basics.ipynb")
        else:
            print(f"✨ Excellent! {achieved_f1:.1%} F1 is strong for baseline models")
            print(f"💡 Consider: Transformers could push you to 90%+ F1")
        
        print(f"{'='*70}\n")
    
    def save(self, path: str):
        """
        Save model to disk
        
        Args:
            path (str): Path to save the model
        """
        import pickle
        from pathlib import Path
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """
        Load model from disk
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            BaselineModel: Loaded model
        """
        import pickle
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 15):
        """
        Get feature importance (for tree-based models)
        
        Args:
            feature_names (list): List of feature names
            top_n (int): Number of top features to show
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\n🔍 Top {top_n} Most Important Features:")
            print(f"{'='*70}")
            for i in range(min(top_n, len(feature_names))):
                idx = indices[i]
                print(f"  {i+1:2d}. {feature_names[idx]:.<50} {importances[idx]:.4f}")
            print(f"{'='*70}")
            
            # Insights
            print(f"\n💡 Feature Engineering Insights:")
            print(f"   - Focus on improving top features")
            print(f"   - Consider removing very low importance features")
            print(f"   - Try creating combinations of important features")
            
        elif self.model_type == 'logistic_regression':
            # For logistic regression, show coefficient magnitudes
            coeffs = np.abs(self.model.coef_[0])
            indices = np.argsort(coeffs)[::-1]
            
            print(f"\n🔍 Top {top_n} Most Influential Features (by coefficient):")
            print(f"{'='*70}")
            for i in range(min(top_n, len(feature_names))):
                idx = indices[i]
                print(f"  {i+1:2d}. {feature_names[idx]:.<50} {coeffs[idx]:.4f}")
            print(f"{'='*70}\n")
        else:
            print("ℹ️  Feature importance not available for this model type")


def train_baseline(X_train, y_train, X_val, y_val, model_type='random_forest'):
    """
    Train and evaluate a baseline model
    
    Quick function to train a model and see results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model to use
        
    Returns:
        BaselineModel: Trained model
        
    Example:
        >>> model = train_baseline(X_train, y_train, X_val, y_val, 'random_forest')
    """
    print(f"\n{'='*70}")
    print(f"Baseline Model Training Pipeline")
    print(f"{'='*70}")
    
    # Initialize model
    model = BaselineModel(model_type=model_type)
    
    # Train
    model.train(X_train, y_train)
    
    # Evaluate on validation set
    print(f"\n📊 Validation Set Performance:")
    results = model.evaluate(X_val, y_val)
    
    # Show feature importance
    if hasattr(X_train, 'columns'):
        model.get_feature_importance(list(X_train.columns))
    
    return model


def compare_models(X_train, y_train, X_val, y_val):
    """
    Compare all baseline models
    
    Trains and evaluates all three baseline models to help you choose.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        dict: Results for each model
    """
    print(f"\n{'='*70}")
    print(f"Comparing All Baseline Models")
    print(f"{'='*70}\n")
    
    models = ['logistic_regression', 'random_forest', 'gradient_boosting']
    results = {}
    
    for model_type in models:
        print(f"\n{'─'*70}")
        print(f"Testing: {model_type}")
        print(f"{'─'*70}")
        
        model = BaselineModel(model_type)
        model.train(X_train, y_train)
        result = model.evaluate(X_val, y_val, detailed=False)
        
        results[model_type] = {
            'model': model,
            'macro_f1': result['macro_f1'],
            'micro_f1': result['micro_f1']
        }
        
        print(f"  → Macro F1: {result['macro_f1']:.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Comparison Summary")
    print(f"{'='*70}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"{emoji} {rank}. {name:.<50} {result['macro_f1']:.4f}")
    
    print(f"{'='*70}")
    print(f"\n💡 Recommendation: Use {sorted_results[0][0]} for best baseline performance")
    print(f"   Then move to transformers for 85-95% F1")
    print(f"{'='*70}\n")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Baseline Models\n")
    
    # Create dummy data for testing
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    
    # Convert to DataFrames
    feature_names = [f'feature_{i}' for i in range(20)]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    y_train_series = pd.Series(y_train)
    y_val_series = pd.Series(y_val)
    
    # Test model training
    print("Testing Random Forest model...\n")
    model = train_baseline(
        X_train_df, y_train_series,
        X_val_df, y_val_series,
        model_type='random_forest'
    )
    
    print("\n" + "="*70)
    print("✅ Baseline model is working!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Try with real data: python src/pipeline.py --task A --train")
    print("  2. Add AST features: notebooks/03_intermediate_features.ipynb")
    print("  3. Move to transformers: notebooks/04_transformer_basics.ipynb")
    print()
