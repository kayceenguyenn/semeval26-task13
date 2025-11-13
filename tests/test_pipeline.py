"""Integration tests for the complete pipeline"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import TaskDataLoader
from features import extract_features_from_dataframe
from models import BaselineModel
from evaluate import evaluate
import numpy as np


def test_end_to_end_pipeline():
    """Test the complete pipeline from data loading to evaluation"""
    # 1. Load data
    loader = TaskDataLoader(task="TEST")
    train_df = loader.load_split("train")
    
    assert len(train_df) > 0
    
    # 2. Extract features
    X_train = extract_features_from_dataframe(train_df)
    y_train = train_df['label'].values
    
    assert len(X_train) == len(y_train)
    
    # 3. Train model
    model = BaselineModel(model_type="logistic_regression")
    model.fit(X_train, y_train)
    
    assert model.is_trained
    
    # 4. Make predictions
    y_pred = model.predict(X_train)
    
    assert len(y_pred) == len(y_train)
    
    # 5. Evaluate
    metrics = evaluate(y_train, y_pred)
    
    assert 'macro_f1' in metrics
    assert metrics['macro_f1'] > 0.0


def test_model_save_load():
    """Test model saving and loading"""
    import tempfile
    import os
    
    # Train a model
    loader = TaskDataLoader(task="TEST")
    train_df = loader.load_split("train")
    X_train = extract_features_from_dataframe(train_df)
    y_train = train_df['label'].values
    
    model = BaselineModel(model_type="random_forest")
    model.fit(X_train, y_train)
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    model.save(temp_path)
    assert os.path.exists(temp_path)
    
    # Load model
    loaded_model = BaselineModel.load(temp_path)
    assert loaded_model.is_trained
    
    # Test predictions match
    pred1 = model.predict(X_train[:10])
    pred2 = loaded_model.predict(X_train[:10])
    assert np.array_equal(pred1, pred2)
    
    # Cleanup
    os.unlink(temp_path)


if __name__ == "__main__":
    print("Running pipeline integration tests...")
    test_end_to_end_pipeline()
    print("✓ test_end_to_end_pipeline")
    
    test_model_save_load()
    print("✓ test_model_save_load")
    
    print("\n✅ All pipeline tests passed!")
