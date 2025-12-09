# src/config.py
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_CSV = os.path.join(DATA_DIR, "autism_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
