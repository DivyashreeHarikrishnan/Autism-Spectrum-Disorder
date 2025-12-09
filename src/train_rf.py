"""
train_model_levelB.py
Advanced training pipeline (Level B)
- Feature engineering (composite scores)
- Models: RandomForest, GradientBoosting, LogisticRegression, SVC
- Voting soft ensemble + CalibratedClassifierCV
- Save model, feature list, metrics, and plots
- Optional SHAP summary (will try; if SHAP errors, the script logs that and continues)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib, json, warnings, os
warnings.filterwarnings("ignore")

# SKLearn / plotting
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional SHAP
USE_SHAP = True

# Paths - adjust if necessary
ROOT = Path(__file__).resolve().parents[1]  # one level up from src/
DATA_PATH = ROOT / "Autism_Behaviour_Dataset.csv"   # change if your CSV name differs
OUT_DIR = ROOT / "models_levelB"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_engineer(path):
    df = pd.read_csv(path)
    # Feature engineering: composite scores
    # Ensure the column names match your dataset exactly
    df['social_score'] = df[['eye_contact','responds_name','points_to_objects','gestures']].sum(axis=1)
    df['communication_score'] = df[['pretend_play','delayed_speech','responds_name']].sum(axis=1)
    df['sensory_score'] = df[['sensory_sensitivity','repetitive_behaviour','restricted_interests']].sum(axis=1)
    # overall: add prefers_alone if present (some datasets use 'prefers_alone' name)
    if 'prefers_alone' in df.columns:
        df['overall_risk_score'] = df['social_score'] + df['communication_score'] + df['sensory_score'] + df['prefers_alone']
    else:
        df['overall_risk_score'] = df['social_score'] + df['communication_score'] + df['sensory_score']
    return df

def train():
    df = load_and_engineer(DATA_PATH)
    # Features and label
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column (0/1).")
    features = [c for c in df.columns if c != 'label']
    X = df[features]
    y = df['label'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Base learners
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=2000, random_state=42))])
    svc = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])

    # Soft voting ensemble
    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svc', svc)],
        voting='soft', n_jobs=-1
    )

    # Calibrate to get better probability estimates; use cv=3 for speed
    calibrated = CalibratedClassifierCV(base_estimator=voting, cv=3, method='isotonic')

    print("Training ensemble (may take a few minutes)...")
    calibrated.fit(X_train, y_train)
    print("Training complete.")

    # Save model and feature list
    joblib.dump(calibrated, OUT_DIR / "calibrated_ensemble_levelB.joblib")
    joblib.dump(features, OUT_DIR / "features_list_levelB.joblib")
    print("Saved calibrated model and features.")

    # Predictions & metrics
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    with open(OUT_DIR / "metrics_levelB.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.savefig(OUT_DIR / "confusion_matrix.png", bbox_inches='tight', dpi=200); plt.close(fig)

    # ROC curve
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],'k--',linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(loc='lower right')
    ax.set_title("ROC Curve")
    fig.savefig(OUT_DIR / "roc_curve.png", bbox_inches='tight', dpi=200); plt.close(fig)

    # Calibration curve
    try:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1)
        ax.plot([0,1],[0,1],'k--', linewidth=0.8)
        ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        fig.savefig(OUT_DIR / "calibration_curve.png", bbox_inches='tight', dpi=200); plt.close(fig)
    except Exception as e:
        print("Calibration curve failed:", e)

    # Feature importances: fit RF and GB on full training set to compute importances
    rf_full = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
    gb_full = GradientBoostingClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
    imp_df = pd.DataFrame({
        "feature": features,
        "rf_imp": rf_full.feature_importances_,
        "gb_imp": gb_full.feature_importances_
    })
    imp_df["mean_imp"] = imp_df[["rf_imp","gb_imp"]].mean(axis=1)
    imp_df = imp_df.sort_values("mean_imp", ascending=False).reset_index(drop=True)
    imp_df.to_csv(OUT_DIR / "feature_importances_levelB.csv", index=False)

    # Bar plot (top 10)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="mean_imp", y="feature", data=imp_df.head(10), ax=ax)
    ax.set_title("Top 10 Feature Importances (mean RF+GB)")
    fig.savefig(OUT_DIR / "feature_importance.png", bbox_inches='tight', dpi=200); plt.close(fig)

    # Save a few sample predictions
    sample = X_test.copy()
    sample["y_test"] = y_test; sample["y_pred"] = y_pred; sample["y_prob"] = y_prob
    sample.head(100).to_csv(OUT_DIR / "sample_predictions.csv", index=False)

    # SHAP explanation (optional)
    if USE_SHAP:
        try:
            import shap
            print("Computing SHAP values (may be slow)...")
            explainer = shap.TreeExplainer(rf_full)  # explain tree model
            shap_vals = explainer.shap_values(X_test)
            # summary bar
            plt.figure(figsize=(6,4))
            shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=200, bbox_inches='tight')
            plt.close()
            print("Saved SHAP summary plot.")
        except Exception as e:
            print("SHAP failed (this can happen in some environments):", e)

    print("All outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    train()
