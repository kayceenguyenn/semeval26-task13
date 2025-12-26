"""
Model implementations for SemEval Task 13

Modular structure to avoid merge conflicts:
- Each student creates their own model file
- No editing the same file simultaneously

Usage:
    from src.models import get_model
    model = get_model('random_forest')
    model = get_model('xgboost', n_estimators=300)
"""

from .baseline import BaselineModel
from .xgboost_model import XGBoostModel
# Add your model imports here:
# from .lightgbm_model import LightGBMModel


def get_model(model_type: str, **kwargs):
    """
    Get a model instance by name

    Args:
        model_type: Model name (e.g., 'random_forest', 'xgboost')
        **kwargs: Model-specific parameters

    Returns:
        Model instance

    Available models:
        - 'logistic_regression': Fast, interpretable baseline
        - 'random_forest': Robust tree ensemble baseline
        - 'gradient_boosting': Best baseline performance
        - 'xgboost': Advanced gradient boosting with GPU support
    """
    if model_type in ['logistic_regression', 'random_forest', 'gradient_boosting']:
        return BaselineModel(model_type=model_type)
    elif model_type == 'xgboost':
        return XGBoostModel(**kwargs)
    # Add your models here:
    # elif model_type == 'lightgbm':
    #     return LightGBMModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: logistic_regression, random_forest, gradient_boosting, xgboost"
        )
