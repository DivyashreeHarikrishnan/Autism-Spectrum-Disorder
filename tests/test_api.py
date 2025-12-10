"""API endpoint tests"""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    print("✓ Root endpoint test passed")

def test_questions():
    response = client.get("/questions")
    assert response.status_code == 200
    data = response.json()
    assert "questions" in data
    assert len(data["questions"]) == 10
    print("✓ Questions endpoint test passed")

def test_predict_low_risk():
    payload = {
        "eye_contact": 1, "responds_name": 1, "points_to_objects": 1,
        "pretend_play": 1, "repetitive_behaviour": 0, "sensory_sensitivity": 0,
        "prefers_alone": 0, "gestures": 1, "delayed_speech": 0, "restricted_interests": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_level" in data
    print(f"✓ Low risk prediction test passed - Risk: {data['risk_level']}")

def test_predict_high_risk():
    payload = {
        "eye_contact": 0, "responds_name": 0, "points_to_objects": 0,
        "pretend_play": 0, "repetitive_behaviour": 1, "sensory_sensitivity": 1,
        "prefers_alone": 1, "gestures": 0, "delayed_speech": 1, "restricted_interests": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ["Medium", "High"]
    print(f"✓ High risk prediction test passed - Risk: {data['risk_level']}")

if __name__ == "__main__":
    print("Running API tests...")
    test_root()
    test_questions()
    test_predict_low_risk()
    test_predict_high_risk()
    print("\n✅ All API tests passed!")