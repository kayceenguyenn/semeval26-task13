"""Data loader for SemEval Task 13"""

from pathlib import Path
from typing import Dict
import pandas as pd
from loguru import logger
import numpy as np

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

class TaskDataLoader:
    """Load training, validation, and test data"""
    
    def __init__(self, task: str):
        self.task = task
        self.data_dir = Path("data")
        
    def load_split(self, split: str) -> pd.DataFrame:
        """Load a data split (train/validation/test)"""
        logger.info(f"Loading {split} data for Task {self.task}")
        
        file_path = self.data_dir / f"{split}_{self.task}.parquet"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}. Generating sample data...")
            return self._generate_sample_data(split)
        
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    def _generate_sample_data(self, split: str) -> pd.DataFrame:
        """Generate sample data for testing"""
        n = 100 if split == 'train' else 20
        
        human = [
            "def hello():\n    print('Hello')",
            "x = 5\ny = 10\nprint(x + y)",
            "for i in range(10):\n    print(i)",
        ]
        
        ai = [
            "def calculate_sum(a, b):\n    return a + b",
            "result = sum([1, 2, 3, 4, 5])",
            "data = [x**2 for x in range(10)]",
        ]
        
        data = []
        for i in range(n):
            code = np.random.choice(human if i % 2 == 0 else ai)
            data.append({'id': i, 'code': code, 'label': i % 2})
        
        return pd.DataFrame(data)
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get dataset statistics"""
        return {
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        }
