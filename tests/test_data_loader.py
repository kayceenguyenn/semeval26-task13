"""Tests for data_loader module"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import TaskDataLoader
import pandas as pd


def test_data_loader_initialization():
    """Test TaskDataLoader can be initialized"""
    loader = TaskDataLoader(task="A")
    assert loader.task == "A"
    assert loader.data_dir == Path("data")


def test_load_split_generates_sample_data():
    """Test that load_split generates sample data when file doesn't exist"""
    loader = TaskDataLoader(task="TEST")
    df = loader.load_split("train")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100  # Default train size
    assert 'id' in df.columns
    assert 'code' in df.columns
    assert 'label' in df.columns


def test_get_statistics():
    """Test statistics calculation"""
    loader = TaskDataLoader(task="A")
    df = loader.load_split("train")
    stats = loader.get_statistics(df)
    
    assert 'total_samples' in stats
    assert 'label_distribution' in stats
    assert stats['total_samples'] == len(df)


if __name__ == "__main__":
    print("Running data_loader tests...")
    test_data_loader_initialization()
    print("✓ test_data_loader_initialization")
    
    test_load_split_generates_sample_data()
    print("✓ test_load_split_generates_sample_data")
    
    test_get_statistics()
    print("✓ test_get_statistics")
    
    print("\n✅ All data_loader tests passed!")
