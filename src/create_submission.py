"""Submission file creation for SemEval Task 13"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


def create_submission_from_arrays(predictions: np.ndarray, ids: np.ndarray, output_path: str) -> None:
    """Create a submission CSV file from prediction arrays"""
    logger.info(f"Creating submission file: {output_path}")
    
    submission_df = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    
    submission_df['id'] = submission_df['id'].astype(int)
    submission_df['label'] = submission_df['label'].astype(int)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved: {output_path}")
    logger.info(f"Total samples: {len(submission_df)}")
    logger.info(f"Label distribution: {submission_df['label'].value_counts().to_dict()}")
