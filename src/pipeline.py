#!/usr/bin/env python3
"""Simple pipeline for Task 13"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import TaskDataLoader
from features import extract_features_from_dataframe
from models import get_model
from evaluate import evaluate
import numpy as np

np.random.seed(42)

def train_baseline(task='A', model_type='random_forest'):
    """Train baseline"""
    print(f"\n{'='*60}")
    print(f"Training Task {task} - {model_type}")
    print(f"{'='*60}\n")
    
    # Load
    print("Loading data...")
    loader = TaskDataLoader(task)
    train_df = loader.load_split('train')
    val_df = loader.load_split('validation')
    print(f"✓ Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Features
    print("\nExtracting features...")
    X_train = extract_features_from_dataframe(train_df)
    X_val = extract_features_from_dataframe(val_df)
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    print(f"✓ Features: {X_train.shape[1]}")
    
    # Train
    print(f"\nTraining {model_type}...")
    model = get_model(model_type)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_val)
    metrics = evaluate(y_val, y_pred, detailed=True)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    # Save
    Path("models").mkdir(exist_ok=True)
    model_path = Path(f"models/task_{task}_{model_type}.pkl")
    model.save(model_path)
    print(f"✅ Saved: {model_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'info'])
    parser.add_argument('--task', '-t', default='A')
    parser.add_argument('--model-type', default='random_forest')
    args = parser.parse_args()
    
    if args.command == 'train':
        train_baseline(args.task, args.model_type)
    elif args.command == 'info':
        loader = TaskDataLoader(args.task)
        train_df = loader.load_split('train')
        print(f"\nTask {args.task}: {len(train_df)} training samples\n")
