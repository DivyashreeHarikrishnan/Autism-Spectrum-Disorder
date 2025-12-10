"""Integration tests"""
import requests
import time

API_URL = "http://localhost:8000"

def test_full_workflow():
    print("Testing full workflow...")
    
    # Test 1: Backend health
    try:
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200
        print("✓ Backend is running")
    except:
        print("✗ Backend not accessible")
        return False
    
    # Test 2: Get questions
    try:
        response = requests.get(f"{API_URL}/questions")
        questions = response.json()["questions"]
        assert len(questions) == 10
        print("✓ Questions retrieved successfully")
    except:
        print("✗ Failed to get questions")
        return False
    
    # Test 3: Submit prediction
    try:
        test_data = {
            "eye_contact": 1, "responds_name": 1, "points_to_objects": 1,
            "pretend_play": 1, "repetitive_behaviour": 0, "sensory_sensitivity": 0,
            "prefers_alone": 0, "gestures": 1, "delayed_speech": 0, "restricted_interests": 0
        }
        response = requests.post(f"{API_URL}/predict", json=test_data)
        result = response.json()
        assert "risk_level" in result
        print(f"✓ Prediction successful - Risk: {result['risk_level']}")
    except:
        print("✗ Prediction failed")
        return False
    
    return True

if __name__ == "__main__":
    print("Running integration tests...")
    print("Make sure backend is running on port 8000")
    time.sleep(1)
    
    if test_full_workflow():
        print("\n✅ All integration tests passed!")
    else:
        print("\n✗ Integration tests failed")