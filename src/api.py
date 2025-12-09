# api_levelB.py (FastAPI)
from fastapi import FastAPI
import joblib, numpy as np, pandas as pd
from pydantic import BaseModel

app = FastAPI(title="ASD Predictor LevelB")

# Load model and feature list (must exist)
MODEL_PATH = "../models_levelB/calibrated_ensemble_levelB.joblib"
FEATURES_PATH = "../models_levelB/features_list_levelB.joblib"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

class BehaviourInput(BaseModel):
    eye_contact: int
    responds_name: int
    points_to_objects: int
    pretend_play: int
    repetitive_behaviour: int
    sensory_sensitivity: int
    prefers_alone: int = 0
    gestures: int
    delayed_speech: int
    restricted_interests: int

@app.post("/predict")
def predict(data: BehaviourInput):
    x = pd.DataFrame([data.dict()])
    # Feature engineering: same as training
    x['social_score'] = x[['eye_contact','responds_name','points_to_objects','gestures']].sum(axis=1)
    x['communication_score'] = x[['pretend_play','delayed_speech','responds_name']].sum(axis=1)
    x['sensory_score'] = x[['sensory_sensitivity','repetitive_behaviour','restricted_interests']].sum(axis=1)
    if 'prefers_alone' in x.columns:
        x['overall_risk_score'] = x['social_score'] + x['communication_score'] + x['sensory_score'] + x['prefers_alone']
    else:
        x['overall_risk_score'] = x['social_score'] + x['communication_score'] + x['sensory_score']

    X = x[features]  # ensure same order
    prob = model.predict_proba(X)[0][1]
    label = int(prob >= 0.5)
    # Risk level mapping
    if prob < 0.40:
        risk = "Low"
    elif prob < 0.75:
        risk = "Medium"
    else:
        risk = "High"

    # Minimal explanation: top features from feature_importances file
    try:
        import pandas as pd
        imp_df = pd.read_csv("../models_levelB/feature_importances_levelB.csv")
        top_features = imp_df.sort_values("mean_imp", ascending=False).head(5)["feature"].tolist()
    except:
        top_features = []

    return {"probability": float(prob), "label": label, "risk": risk, "top_features": top_features}
