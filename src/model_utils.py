import joblib
import pandas as pd
from pathlib import Path

def load_model():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    model = joblib.load(model_dir / "asd_model.joblib")
    features = joblib.load(model_dir / "features_list.joblib")
    return model, features

def prepare_features(data_dict):
    df = pd.DataFrame([data_dict])
    df['social_score'] = df[['eye_contact', 'responds_name', 'points_to_objects', 'gestures']].sum(axis=1)
    df['communication_score'] = df[['pretend_play', 'delayed_speech', 'responds_name']].sum(axis=1)
    df['sensory_score'] = df[['sensory_sensitivity', 'repetitive_behaviour', 'restricted_interests']].sum(axis=1)
    df['overall_risk_score'] = df['social_score'] + df['communication_score'] + df['sensory_score'] + df['prefers_alone']
    return df