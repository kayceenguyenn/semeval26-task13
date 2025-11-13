"""
Configuration management with Pydantic - BountyBench Style

Type-safe configuration with validation and IDE autocomplete support.

Usage:
    from config import Config, ModelConfig, PathConfig
    
    config = Config.load()
    print(config.model.random_forest.n_estimators)
"""

from pathlib import Path
from typing import Dict, List, Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import yaml


class TaskType(str, Enum):
    """Available task types"""
    A = "A"
    B = "B"
    C = "C"


class ModelType(str, Enum):
    """Available baseline model types"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class LogisticRegressionConfig(BaseModel):
    """Configuration for Logistic Regression model"""
    max_iter: int = Field(default=1000, ge=100, description="Maximum iterations")
    random_state: int = Field(default=42, description="Random seed")
    class_weight: str = Field(default="balanced", description="Class weight strategy")
    

class RandomForestConfig(BaseModel):
    """Configuration for Random Forest model"""
    n_estimators: int = Field(default=100, ge=10, le=1000, description="Number of trees")
    max_depth: Optional[int] = Field(default=20, ge=1, description="Maximum tree depth")
    min_samples_split: int = Field(default=10, ge=2, description="Min samples to split")
    random_state: int = Field(default=42, description="Random seed")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")
    class_weight: str = Field(default="balanced", description="Class weight strategy")


class GradientBoostingConfig(BaseModel):
    """Configuration for Gradient Boosting model"""
    n_estimators: int = Field(default=100, ge=10, le=1000, description="Number of boosting stages")
    max_depth: int = Field(default=5, ge=1, le=20, description="Maximum tree depth")
    learning_rate: float = Field(default=0.1, gt=0, le=1, description="Learning rate")
    random_state: int = Field(default=42, description="Random seed")


class ModelConfig(BaseModel):
    """Model hyperparameters configuration"""
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    gradient_boosting: GradientBoostingConfig = Field(default_factory=GradientBoostingConfig)


class PathConfig(BaseModel):
    """File paths configuration"""
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_data_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    models_dir: Path = Field(default=Path("models"), description="Model checkpoints directory")
    results_dir: Path = Field(default=Path("results"), description="Results directory")
    predictions_dir: Path = Field(default=Path("results/predictions"), description="Predictions directory")
    figures_dir: Path = Field(default=Path("results/figures"), description="Figures directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    
    def create_all(self) -> None:
        """Create all directories"""
        for field_name in self.model_fields:
            path = getattr(self, field_name)
            path.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    """Data loading configuration"""
    dataset_repo: str = Field(
        default="DaniilOr/SemEval-2026-Task13",
        description="HuggingFace dataset repository"
    )
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory for downloads")
    use_cache: bool = Field(default=True, description="Whether to use cached data")


class FeatureConfig(BaseModel):
    """Feature extraction configuration"""
    extract_basic: bool = Field(default=True, description="Extract basic features")
    extract_complexity: bool = Field(default=True, description="Extract complexity features")
    extract_style: bool = Field(default=True, description="Extract style features")
    extract_ast: bool = Field(default=False, description="Extract AST features (slower)")
    batch_size: int = Field(default=1000, ge=1, description="Batch size for feature extraction")


class TrainingConfig(BaseModel):
    """Training configuration"""
    save_model: bool = Field(default=True, description="Save trained model")
    save_predictions: bool = Field(default=True, description="Save predictions")
    eval_on_validation: bool = Field(default=True, description="Evaluate on validation set")
    cross_validate: bool = Field(default=False, description="Perform cross-validation")
    n_folds: int = Field(default=5, ge=2, le=10, description="Number of CV folds")


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_to_file: bool = Field(default=True, description="Log to file")
    log_to_console: bool = Field(default=True, description="Log to console")
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="1 week", description="Log retention period")


class Config(BaseSettings):
    """
    Main configuration class
    
    Load from config.yaml or use defaults.
    
    Example:
        config = Config.load()
        print(config.model.random_forest.n_estimators)
    """
    
    # Sub-configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # General settings
    random_seed: int = Field(default=42, description="Global random seed")
    debug: bool = Field(default=False, description="Debug mode")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file or use defaults
        
        Args:
            config_path: Path to config.yaml (optional)
            
        Returns:
            Config instance
        """
        if config_path and config_path.exists():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        return cls()
    
    def save(self, config_path: Path) -> None:
        """
        Save configuration to YAML file
        
        Args:
            config_path: Path to save config.yaml
        """
        config_dict = self.model_dump(mode='json')
        
        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def setup(self) -> None:
        """Setup configuration (create directories, etc.)"""
        self.paths.create_all()
        
        # Set random seeds
        import random
        import numpy as np
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)


class TaskConfig(BaseModel):
    """Task-specific configuration"""
    task: TaskType = Field(description="Task identifier")
    model_type: ModelType = Field(
        default=ModelType.RANDOM_FOREST,
        description="Model type to use"
    )
    
    @property
    def num_classes(self) -> int:
        """Get number of classes for the task"""
        class_map = {
            TaskType.A: 2,
            TaskType.B: 11,
            TaskType.C: 4,
        }
        return class_map[self.task]
    
    @property
    def expected_baseline_f1(self) -> str:
        """Get expected baseline F1 score"""
        f1_map = {
            TaskType.A: "50-60%",
            TaskType.B: "35-45%",
            TaskType.C: "30-40%",
        }
        return f1_map[self.task]
    
    @property
    def expected_competitive_f1(self) -> str:
        """Get expected competitive F1 score"""
        f1_map = {
            TaskType.A: "90%+",
            TaskType.B: "85%+",
            TaskType.C: "80%+",
        }
        return f1_map[self.task]


# Example usage and testing
if __name__ == "__main__":
    from rich.console import Console
    from rich.tree import Tree
    
    console = Console()
    
    # Load default configuration
    config = Config.load()
    
    # Display configuration
    console.print("\n[bold blue]Configuration Loaded[/bold blue]\n")
    
    tree = Tree("📋 Config")
    
    # Paths
    paths_branch = tree.add("📁 Paths")
    for field_name, field_value in config.paths:
        paths_branch.add(f"{field_name}: {field_value}")
    
    # Model
    model_branch = tree.add("🤖 Model")
    rf_branch = model_branch.add("Random Forest")
    for field_name, field_value in config.model.random_forest:
        rf_branch.add(f"{field_name}: {field_value}")
    
    # Features
    features_branch = tree.add("🔧 Features")
    for field_name, field_value in config.features:
        features_branch.add(f"{field_name}: {field_value}")
    
    console.print(tree)
    
    # Setup (create directories)
    console.print("\n[bold]Setting up directories...[/bold]")
    config.setup()
    console.print("✓ All directories created", style="green")
    
    # Save configuration
    config_path = Path("config.yaml")
    config.save(config_path)
    console.print(f"\n✓ Configuration saved to {config_path}", style="green")
    
    # Test task config
    console.print("\n[bold blue]Task-Specific Configuration[/bold blue]\n")
    
    for task in [TaskType.A, TaskType.B, TaskType.C]:
        task_config = TaskConfig(task=task)
        console.print(f"Task {task.value}:")
        console.print(f"  Classes: {task_config.num_classes}")
        console.print(f"  Expected baseline: {task_config.expected_baseline_f1}")
        console.print(f"  Expected competitive: {task_config.expected_competitive_f1}")
        console.print()
