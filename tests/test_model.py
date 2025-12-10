"""Model prediction tests"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_utils import load_model, prepare_features

def test_model_loading():
    try:
        model, features = load_model()
        assert model is not None
        assert features is not None
        print("✓ Model loading test passed")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_feature_preparation():
    test_data = {
        "eye_contact": 1, "responds_name": 1, "points_to_objects": 1,
        "pretend_play": 1, "repetitive_behaviour": 0, "sensory_sensitivity": 0,
        "prefers_alone": 0, "gestures": 1, "delayed_speech": 0, "restricted_interests": 0
    }
    df = prepare_features(test_data)
    assert 'social_score' in df.columns
    assert 'communication_score' in df.columns
    assert 'sensory_score' in df.columns
    print("✓ Feature preparation test passed")

def test_prediction():
    try:
        model, features = load_model()
        test_data = {
            "eye_contact": 1, "responds_name": 1, "points_to_objects": 1,
            "pretend_play": 1, "repetitive_behaviour": 0, "sensory_sensitivity": 0,
            "prefers_alone": 0, "gestures": 1, "delayed_speech": 0, "restricted_interests": 0
        }
        df = prepare_features(test_data)
        X = df[features]
        prob = model.predict_proba(X)[0][1]
        assert 0 <= prob <= 1
        print(f"✓ Prediction test passed - Probability: {prob:.3f}")
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

if __name__ == "__main__":
    print("Running model tests...")
    test_model_loading()
    test_feature_preparation()
    test_prediction()
    print("\n✅ All model tests passed!")