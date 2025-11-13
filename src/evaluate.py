"""Evaluation metrics for SemEval Task 13"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from typing import Dict
from loguru import logger


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, detailed: bool = False) -> Dict:
    """Evaluate predictions using Macro F1 score"""
    logger.debug(f"Evaluating {len(y_true)} predictions")
    
    metrics = {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if detailed:
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['report'] = classification_report(y_true, y_pred)
    
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    return metrics


def print_results(metrics: Dict) -> None:
    """Print evaluation results"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    if 'precision' in metrics:
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
    
    if 'report' in metrics:
        print("\n" + metrics['report'])
    
    print("="*50 + "\n")
