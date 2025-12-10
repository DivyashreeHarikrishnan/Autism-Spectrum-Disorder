import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_CSV = DATA_DIR / "autism_dataset.csv"
MODEL_PATH = MODEL_DIR / "asd_model.joblib"
FEATURES_PATH = MODEL_DIR / "features_list.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

API_HOST = "0.0.0.0"
API_PORT = 8000
RANDOM_STATE = 42
TEST_SIZE = 0.20