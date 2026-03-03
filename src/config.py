import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "creditcard.csv"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "fraud_model.joblib"

# Reporting
REPORTS_DIR = PROJECT_ROOT / "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
