import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_CSV, MODEL_DIR

warnings.filterwarnings("ignore")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_and_engineer(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    df['social_score'] = df[['eye_contact', 'responds_name', 'points_to_objects', 'gestures']].sum(axis=1)
    df['communication_score'] = df[['pretend_play', 'delayed_speech', 'responds_name']].sum(axis=1)
    df['sensory_score'] = df[['sensory_sensitivity', 'repetitive_behaviour', 'restricted_interests']].sum(axis=1)
    df['overall_risk_score'] = df['social_score'] + df['communication_score'] + df['sensory_score'] + df['prefers_alone']
    
    return df

def train():
    print("="*60)
    print("ASD DETECTION MODEL TRAINING")
    print("="*60)
    
    df = load_and_engineer(DATA_CSV)
    features = [c for c in df.columns if c != 'label']
    X = df[features]
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    print("\nBuilding ensemble model...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=2000, random_state=42))])
    svc = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])
    
    voting = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svc', svc)], voting='soft', n_jobs=-1)
    calibrated = CalibratedClassifierCV(base_estimator=voting, cv=3, method='isotonic')
    
    print("Training...")
    calibrated.fit(X_train, y_train)
    print("Training complete!\n")
    
    joblib.dump(calibrated, MODEL_DIR / "asd_model.joblib")
    joblib.dump(features, MODEL_DIR / "features_list.joblib")
    print("Model saved!")
    
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
    print("="*60)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "roc_curve.png", dpi=300)
    plt.close()
    
    print("\nVisualizations saved!")
    print(f"All outputs in: {MODEL_DIR}")

if __name__ == "__main__":
    train()