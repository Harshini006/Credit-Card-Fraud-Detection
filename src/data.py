import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification
from src.config import DATA_FILE
from src.utils import logger

def load_data(filepath=DATA_FILE):
    """
    Loads the dataset from CSV or generates a synthetic one if the file doesn't exist.
    """
    if os.path.exists(filepath):
        logger.info(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
    else:
        logger.warning(f"File {filepath} not found. Generating synthetic dataset for demonstration...")
        # Generate synthetic data mimicking credit card fraud data. 
        # Using 30 features to mimic V1-V28, Time, Amount.
        X, y = make_classification(n_samples=10000, n_features=30, n_informative=24, n_redundant=2,
                                   n_clusters_per_class=1, weights=[0.99], random_state=42)
        df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'])
        df['Class'] = y
        # Adjust Time and Amount to look more realistic
        df['Time'] = df['Time'].abs() * 1000
        df['Amount'] = df['Amount'].abs() * 100
        
    logger.info(f"Dataset Shape: {df.shape}")
    return df
