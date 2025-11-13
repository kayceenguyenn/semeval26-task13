#!/usr/bin/env python3
"""Data download script for SemEval Task 13"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)


def generate_sample_data(task: str, output_dir: Path) -> None:
    """Generate sample data for testing"""
    print(f"📦 Generating sample data for Task {task}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    human_codes = [
        "def hello():\n    print('Hello, World!')",
        "x = 5\ny = 10\nresult = x + y",
        "for i in range(10):\n    print(i)",
        "class MyClass:\n    def __init__(self):\n        pass",
        "import sys\nprint(sys.version)",
    ]
    
    ai_codes = [
        "def calculate_sum(a, b):\n    '''Calculate sum'''\n    return a + b",
        "result = sum([x**2 for x in range(10)])",
        "data = [item for item in range(100) if item % 2 == 0]",
        "def process_data(data):\n    return [x * 2 for x in data]",
        "from typing import List\ndef func(items: List[int]) -> int:\n    return sum(items)",
    ]
    
    # Training data
    n_train = 1000
    train_data = []
    for i in range(n_train):
        code = np.random.choice(human_codes if i % 2 == 0 else ai_codes)
        train_data.append({'id': i, 'code': code, 'label': i % 2})
    
    train_df = pd.DataFrame(train_data)
    train_path = output_dir / f"train_{task}.parquet"
    train_df.to_parquet(train_path)
    print(f"  ✓ Created: {train_path} ({len(train_df)} samples)")
    
    # Validation data
    n_val = 200
    val_data = []
    for i in range(n_val):
        code = np.random.choice(human_codes if i % 2 == 0 else ai_codes)
        val_data.append({'id': n_train + i, 'code': code, 'label': i % 2})
    
    val_df = pd.DataFrame(val_data)
    val_path = output_dir / f"validation_{task}.parquet"
    val_df.to_parquet(val_path)
    print(f"  ✓ Created: {val_path} ({len(val_df)} samples)")
    
    # Test data (no labels)
    n_test = 100
    test_data = []
    for i in range(n_test):
        code = np.random.choice(human_codes + ai_codes)
        test_data.append({'id': n_train + n_val + i, 'code': code})
    
    test_df = pd.DataFrame(test_data)
    test_path = output_dir / f"test_{task}.parquet"
    test_df.to_parquet(test_path)
    print(f"  ✓ Created: {test_path} ({len(test_df)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Download SemEval Task 13 data")
    parser.add_argument("--task", type=str, default="A", choices=["A", "B", "C", "all"])
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    tasks = ["A", "B", "C"] if args.task == "all" else [args.task]
    
    for task in tasks:
        generate_sample_data(task, output_dir)
    
    print("\n✅ Data generation complete!")
    print(f"📁 Data saved to: {output_dir}")
    print("\n🚀 Next steps:")
    print(f"   python3 src/pipeline.py train --task A")


if __name__ == "__main__":
    main()
