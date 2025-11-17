#!/usr/bin/env python3
"""Data download script for SemEval Task 13"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)


def download_real_data(task: str, output_dir: Path, sample_size: int = None) -> None:
    """
    Download real data from HuggingFace
    
    Args:
        task: Task identifier (A, B, or C)
        output_dir: Directory to save data
        sample_size: Optional - limit dataset size for testing (None = full dataset)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("   Install with: pip install datasets")
        print("   Falling back to sample data...")
        generate_sample_data(task, output_dir)
        return
    
    print(f"📥 Downloading real data for Task {task} from HuggingFace...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("DaniilOr/SemEval-2026-Task13", task)
        
        # Get train split
        if 'train' in dataset:
            train_df = dataset['train'].to_pandas()
            
            # Sample if requested
            if sample_size and len(train_df) > sample_size:
                print(f"   Sampling {sample_size:,} from {len(train_df):,} samples...")
                train_df = train_df.sample(n=sample_size, random_state=SEED)
            
            # Create validation split (10%)
            train_df, val_df = train_test_split(
                train_df, test_size=0.1, random_state=SEED, stratify=train_df['label']
            )
            
            # Save train
            train_path = output_dir / f"train_{task}.parquet"
            train_df.to_parquet(train_path, index=False)
            print(f"  ✓ Train: {train_path} ({len(train_df):,} samples)")
            
            # Save validation
            val_path = output_dir / f"validation_{task}.parquet"
            val_df.to_parquet(val_path, index=False)
            print(f"  ✓ Validation: {val_path} ({len(val_df):,} samples)")
        
        # Get test split (if available)
        if 'test' in dataset:
            test_df = dataset['test'].to_pandas()
            test_path = output_dir / f"test_{task}.parquet"
            test_df.to_parquet(test_path, index=False)
            print(f"  ✓ Test: {test_path} ({len(test_df):,} samples)")
        else:
            print("  ⚠️  No test split available yet (will be released Jan 10, 2026)")
        
        # Print dataset info
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(train_df) + len(val_df):,}")
        print(f"   Label distribution: {train_df['label'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print(f"   Falling back to sample data generation...")
        generate_sample_data(task, output_dir)


def generate_sample_data(task: str, output_dir: Path) -> None:
    """
    Generate small sample data for testing (fallback)
    Use this only if real data download fails
    """
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
    parser.add_argument("--task", type=str, default="A", choices=["A", "B", "C", "all"],
                       help="Task to download (A, B, C, or all)")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for data files")
    parser.add_argument("--sample", type=int, default=None,
                       help="Sample size (for testing). Use None for full dataset")
    parser.add_argument("--use-real-data", action="store_true", default=True,
                       help="Download real data from HuggingFace (default: True)")
    parser.add_argument("--use-sample-data", action="store_true",
                       help="Use synthetic sample data instead of real data")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    tasks = ["A", "B", "C"] if args.task == "all" else [args.task]
    
    # Determine which data source to use
    use_real = args.use_real_data and not args.use_sample_data
    
    for task in tasks:
        if use_real:
            download_real_data(task, output_dir, args.sample)
        else:
            generate_sample_data(task, output_dir)
    
    print("\n✅ Data download complete!")
    print(f"📁 Data saved to: {output_dir}")
    print("\n🚀 Next steps:")
    print(f"   python3 src/pipeline.py train --task A")


if __name__ == "__main__":
    main()
