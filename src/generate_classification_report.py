# src/generate_classification_report.py
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report

# Paths
DATA_CSV = "data/autism_dataset.csv"
MODEL_PATH = "models/asd_model.joblib"
FEATURES_PATH = "models/features_list.joblib"        # optional but recommended
PREPROCESSOR_PATH = "models/preprocessor.joblib"     # optional: scaler/pipeline
OUT_REPORT_CSV = "models/classification_report.csv"

# 1) load dataset
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

# 2) raw feature names (from your dataset)
raw_features = [
    "eye_contact",
    "responds_name",
    "points_to_objects",
    "pretend_play",
    "repetitive_behaviour",
    "sensory_sensitivity",
    "prefers_alone",
    "gestures",
    "delayed_speech",
    "restricted_interests",
]

# Basic validation
missing = [c for c in raw_features + ["label"] if c not in df.columns]
if missing:
    raise KeyError(f"Required columns missing in dataset: {missing}")

# 3) compute derived features exactly as in training
df = df.copy()
df["social_score"] = df[["eye_contact", "responds_name", "points_to_objects", "gestures"]].sum(axis=1)
df["communication_score"] = df[["pretend_play", "delayed_speech", "responds_name"]].sum(axis=1)
df["sensory_score"] = df[["sensory_sensitivity", "repetitive_behaviour", "restricted_interests"]].sum(axis=1)
df["overall_risk_score"] = df["social_score"] + df["communication_score"] + df["sensory_score"] + df["prefers_alone"]

# 4) decide final feature order
if os.path.exists(FEATURES_PATH):
    try:
        features = joblib.load(FEATURES_PATH)
        # ensure features is a list and all exist in df
        features = list(features)
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            raise KeyError(f"Features listed in {FEATURES_PATH} are missing in the dataframe: {missing_feats}")
    except Exception as e:
        raise RuntimeError(f"Failed to load features list from {FEATURES_PATH}: {e}")
else:
    # fallback: use all raw + derived in a sensible order
    features = raw_features + ["social_score", "communication_score", "sensory_score", "overall_risk_score"]

# 5) build X and y
X = df[features]
y = df["label"]

# 6) load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# 7) optionally apply preprocessor/scaler
X_for_pred = X.copy()
if os.path.exists(PREPROCESSOR_PATH):
    try:
        pre = joblib.load(PREPROCESSOR_PATH)
        # try transform; many preprocessors expect a DataFrame or ndarray:
        try:
            X_for_pred = pre.transform(X_for_pred)
        except Exception:
            # if pipeline expects DataFrame columns by name, try passing df[features]
            X_for_pred = pre.transform(X_for_pred.values)
        print("Applied saved preprocessor successfully.")
    except Exception as e:
        print("⚠️  Could not apply preprocessor:", e)
        print("Proceeding without applying preprocessor.")
else:
    print("No preprocessor found; proceeding without scaling/transform.")

# 8) predict
y_pred = model.predict(X_for_pred)

# 9) create classification report and save
report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(OUT_REPORT_CSV)
print(f"Classification report saved to {OUT_REPORT_CSV}")
