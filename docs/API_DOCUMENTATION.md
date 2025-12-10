# API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check
**GET** `/`

**Response:**
```json
{
  "message": "ASD Screening API",
  "status": "running",
  "model_loaded": true
}
```

---

### 2. Get Questions
**GET** `/questions`

**Response:**
```json
{
  "questions": [
    {
      "id": 1,
      "field": "eye_contact",
      "text": "Does your child make eye contact during interactions?"
    },
    ...
  ]
}
```

---

### 3. Predict
**POST** `/predict`

**Request Body:**
```json
{
  "eye_contact": 1,
  "responds_name": 1,
  "points_to_objects": 1,
  "pretend_play": 1,
  "repetitive_behaviour": 0,
  "sensory_sensitivity": 0,
  "prefers_alone": 0,
  "gestures": 1,
  "delayed_speech": 0,
  "restricted_interests": 0
}
```

**Response:**
```json
{
  "probability": 0.234,
  "prediction": "Minimal Indicators",
  "risk_level": "Low",
  "confidence": 76.6,
  "message": "Based on the screening...",
  "doctor_recommendation": "Routine checkup recommended"
}
```

**Risk Levels:**
- **Low**: probability < 0.30
- **Medium**: 0.30 ≤ probability < 0.70
- **High**: probability ≥ 0.70

---

## Error Responses

**503 Service Unavailable:**
```json
{
  "detail": "Model not loaded"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Prediction failed: <error message>"
}
```

---

## Testing with cURL
```bash
# Health check
curl http://localhost:8000/

# Get questions
curl http://localhost:8000/questions

# Submit prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "eye_contact": 1,
    "responds_name": 1,
    "points_to_objects": 1,
    "pretend_play": 1,
    "repetitive_behaviour": 0,
    "sensory_sensitivity": 0,
    "prefers_alone": 0,
    "gestures": 1,
    "delayed_speech": 0,
    "restricted_interests": 0
  }'
```