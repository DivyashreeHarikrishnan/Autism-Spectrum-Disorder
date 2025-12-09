# src/model_utils.py
"""
Utilities for preprocessing user responses and loading/predicting with the trained model.
"""

import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH, PREPROCESSOR_PATH
import json
import os

# The 10-question order (feature names) used for training and inference.
# Adjust these names to match your dataset column names if you have a different CSV.
FEATURES = [
    "eye_contact",        # 1 = reduced eye contact
    "responds_name",      # 1 = does NOT respond to name
    "pointing",           # 1 = does NOT point to show interest
    "pretend_play",       # 1 = lacks pretend play
    "repetitive_behaviour",# 1 = repetitive movements present
    "sensory_sensitivity", # 1 = sensory sensitivity present
    "plays_alone",        # 1 = prefers playing alone
    "gestures",           # 1 = lacks gestures
    "delayed_speech",     # 1 = delayed speech
    "restricted_interests"# 1 = intense, restricted interests
]

# Default mapping: user answers of "yes"/"no" or boolean values:
# Expect user to send JSON with keys matching FEATURES or positional list.
ANSWER_MAP = {
    "yes": 1,
    "no": 0,
    True: 1,
    False: 0,
    "1": 1,
    "0": 0,
    1: 1,
    0: 0
}

def load_model_and_preprocessor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Model or preprocessor not found. Run training first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def answers_to_feature_vector(answers):
    """
    answers: either
      - dict with keys matching FEATURES, values in ("yes"/"no", True/False, 1/0)
      - list/tuple of length len(FEATURES) in same order as FEATURES
    returns: numpy array shape (1, n_features)
    """
    if isinstance(answers, (list, tuple)):
        if len(answers) != len(FEATURES):
            raise ValueError("Answer list length mismatch.")
        mapped = []
        for v in answers:
            mapped.append(ANSWER_MAP.get(v, 0))
        return np.array([mapped], dtype=float)
    elif isinstance(answers, dict):
        mapped = []
        for key in FEATURES:
            if key not in answers:
                # default 0 (no concern)
                v = 0
            else:
                v = answers[key]
            mapped.append(ANSWER_MAP.get(v, 0))
        return np.array([mapped], dtype=float)
    else:
        raise ValueError("Unsupported answers format")

def prepare_dataframe_for_training(df, feature_list=FEATURES):
    """
    Convert an input CSV dataframe to the expected format.
    This function assumes your CSV columns contain features with understandable names.
    If your CSV uses different names, adapt mapping here.
    It also expects a 'label' column (0 = no ASD, 1 = ASD).
    """
    # If columns are present under different names, attempt common variants:
    # This function currently expects the columns to exist. If not, try lower-case keys.
    df2 = df.copy()
    # normalize column names
    df2.columns = [c.strip().lower() for c in df2.columns]
    # Map common possible names to ours:
    mapping_guess = {}
    # Create features with defaults if missing
    for feat in feature_list:
        if feat not in df2.columns:
            # attempt some common synonyms
            guesses = {
                "eye_contact": ["eye_contact", "eyecontact", "eye contact", "gaze"],
                "responds_name": ["responds_name", "response_to_name", "name_response"],
                "pointing": ["pointing", "points", "point"],
                "pretend_play": ["pretend_play", "play_imaginative", "symbolic_play"],
                "repetitive_behaviour": ["repetitive_behaviour", "stereotyped_movements", "repetitive"],
                "sensory_sensitivity": ["sensory_sensitivity", "sensory"],
                "plays_alone": ["plays_alone", "solitary_play", "prefers_alone"],
                "gestures": ["gestures", "use_of_gestures"],
                "delayed_speech": ["delayed_speech", "speech_delay", "language_delay"],
                "restricted_interests": ["restricted_interests", "fixated_interests"]
            }
            # try to find a column that contains one of guesses
            found = False
            for g in guesses.get(feat, []):
                for col in df2.columns:
                    if g in col:
                        mapping_guess[feat] = col
                        found = True
                        break
                if found:
                    break
    # If mapping_guess found columns, rename to our feature names
    if mapping_guess:
        df2 = df2.rename(columns={v: k for k, v in mapping_guess.items()})

    # Ensure feature columns exist; if not, create default 0
    for feat in feature_list:
        if feat not in df2.columns:
            df2[feat] = 0

    # Ensure label exists
    if "label" not in df2.columns:
        raise KeyError("Training CSV must contain a 'label' column with 0/1 values (0=no ASD, 1=ASD).")

    # Convert values to binary 0/1 (using mapping where possible)
    def to_binary(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("yes", "y", "1", "true", "t"):
                return 1
            if v in ("no", "n", "0", "false", "f"):
                return 0
            # fallback numeric parse
            try:
                return int(float(v))
            except:
                return 0
        if isinstance(val, (int, float, bool)):
            return int(bool(val))
        return 0

    for feat in feature_list:
        df2[feat] = df2[feat].apply(to_binary)

    df2["label"] = df2["label"].apply(to_binary)

    # Keep only features + label
    return df2[[*feature_list, "label"]]

def predict_from_answers(answers, return_explanation=False, top_k=5):
    model, preprocessor = load_model_and_preprocessor()
    x = answers_to_feature_vector(answers)
    x_scaled = preprocessor.transform(x)
    proba = model.predict_proba(x_scaled)[0, 1]
    pred = int(model.predict(x_scaled)[0])

    explanation = None
    if return_explanation:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(x_scaled)[1]  # class 1
            # shap_vals is array shape (n_features,)
            feature_contribs = list(zip(FEATURES, shap_vals.tolist()))
            feature_contribs_sorted = sorted(feature_contribs, key=lambda t: -abs(t[1]))
            explanation = [{"feature": f, "shap_value": float(s)} for f, s in feature_contribs_sorted[:top_k]]
        except Exception as e:
            explanation = {"error": f"SHAP explanation failed: {str(e)}"}

    return {
        "prediction": pred,
        "probability": float(proba),
        "explanation": explanation
    }
