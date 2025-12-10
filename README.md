# ASD Detection System

## Quick Start

1. Install dependencies:
```
   pip install -r requirements.txt
```

2. Train model:
```
   cd src
   python train_model.py
```

3. Start backend:
```
   cd src
   uvicorn api:app --reload --port 8000
```

4. Start frontend (new terminal):
```
   cd frontend
   python -m http.server 3000
```

5. Open browser: http://localhost:3000
```

---

## STEP 2: Data Directory

### File: `data/data_info.txt`
```
Autism Dataset Information
Dataset: autism_dataset.csv
Total Samples: 1000
Features: 10 behavioral indicators
Target: Binary (0=No ASD, 1=ASD)