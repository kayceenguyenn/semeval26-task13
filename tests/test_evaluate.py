"""Tests for evaluate module"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluate import evaluate
import numpy as np


def test_evaluate_perfect_predictions():
    """Test evaluation with perfect predictions"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    
    metrics = evaluate(y_true, y_pred)
    
    assert 'macro_f1' in metrics
    assert 'micro_f1' in metrics
    assert 'weighted_f1' in metrics
    assert metrics['macro_f1'] == 1.0
    assert metrics['micro_f1'] == 1.0


def test_evaluate_random_predictions():
    """Test evaluation with random predictions"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    
    metrics = evaluate(y_true, y_pred)
    
    assert 'macro_f1' in metrics
    assert 0.0 <= metrics['macro_f1'] <= 1.0


def test_evaluate_detailed():
    """Test evaluation with detailed metrics"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    
    metrics = evaluate(y_true, y_pred, detailed=True)
    
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'report' in metrics


if __name__ == "__main__":
    print("Running evaluate tests...")
    test_evaluate_perfect_predictions()
    print("✓ test_evaluate_perfect_predictions")
    
    test_evaluate_random_predictions()
    print("✓ test_evaluate_random_predictions")
    
    test_evaluate_detailed()
    print("✓ test_evaluate_detailed")
    
    print("\n✅ All evaluate tests passed!")
