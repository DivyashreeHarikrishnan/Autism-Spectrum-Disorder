from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="ASD Screening API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

try:
    model = joblib.load(MODEL_DIR / "asd_model.joblib")
    features = joblib.load(MODEL_DIR / "features_list.joblib")
except:
    model = None
    features = None

class ScreeningInput(BaseModel):
    eye_contact: int = Field(..., ge=0, le=1)
    responds_name: int = Field(..., ge=0, le=1)
    points_to_objects: int = Field(..., ge=0, le=1)
    pretend_play: int = Field(..., ge=0, le=1)
    repetitive_behaviour: int = Field(..., ge=0, le=1)
    sensory_sensitivity: int = Field(..., ge=0, le=1)
    prefers_alone: int = Field(..., ge=0, le=1)
    gestures: int = Field(..., ge=0, le=1)
    delayed_speech: int = Field(..., ge=0, le=1)
    restricted_interests: int = Field(..., ge=0, le=1)

@app.get("/")
def root():
    return {"message": "ASD Screening API", "status": "running", "model_loaded": model is not None}

@app.get("/questions")
def get_questions():
    return {
        "questions": [
            {"id": 1, "field": "eye_contact", "text": "Does your child make eye contact during interactions?"},
            {"id": 2, "field": "responds_name", "text": "Does your child respond when their name is called?"},
            {"id": 3, "field": "points_to_objects", "text": "Does your child point to objects to show interest?"},
            {"id": 4, "field": "pretend_play", "text": "Does your child engage in pretend or imaginative play?"},
            {"id": 5, "field": "repetitive_behaviour", "text": "Does your child show repetitive movements (hand-flapping, rocking)?"},
            {"id": 6, "field": "sensory_sensitivity", "text": "Is your child overly sensitive to sounds, textures, lights, or touch?"},
            {"id": 7, "field": "prefers_alone", "text": "Does your child prefer to play alone rather than with others?"},
            {"id": 8, "field": "gestures", "text": "Does your child use gestures like waving, nodding, or showing objects?"},
            {"id": 9, "field": "delayed_speech", "text": "Does your child have delayed speech or limited vocabulary for their age?"},
            {"id": 10, "field": "restricted_interests", "text": "Does your child have intense, narrow interests in specific topics or objects?"}
        ]
    }

@app.post("/predict")
def predict(data: ScreeningInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([data.dict()])
        df['social_score'] = df[['eye_contact', 'responds_name', 'points_to_objects', 'gestures']].sum(axis=1)
        df['communication_score'] = df[['pretend_play', 'delayed_speech', 'responds_name']].sum(axis=1)
        df['sensory_score'] = df[['sensory_sensitivity', 'repetitive_behaviour', 'restricted_interests']].sum(axis=1)
        df['overall_risk_score'] = df['social_score'] + df['communication_score'] + df['sensory_score'] + df['prefers_alone']
        
        X = df[features]
        prob = float(model.predict_proba(X)[0][1])
        
        if prob < 0.30:
            risk = "Low"
            msg = "Based on the screening, your child shows minimal indicators. Continue monitoring development."
            rec = "Routine checkup recommended"
        elif prob < 0.70:
            risk = "Medium"
            msg = "Some indicators present. We recommend consulting a pediatrician for comprehensive evaluation."
            rec = "Consultation Recommended - Please schedule an appointment with a pediatrician"
        else:
            risk = "High"
            msg = "Multiple indicators detected. Please consult a healthcare professional for formal assessment."
            rec = "⚠️ Professional Evaluation Urgently Recommended - Consult a specialist immediately"
        
        return {
            "probability": round(prob, 3),
            "prediction": "Positive Indicators" if prob >= 0.5 else "Minimal Indicators",
            "risk_level": risk,
            "confidence": round(max(prob, 1-prob) * 100, 1),
            "message": msg,
            "doctor_recommendation": rec
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))