# Autism Spectrum Disorder Detection System

## Overview
Machine Learning-based early screening tool for ASD in toddlers using behavioral indicators.

## Features
- 10-question behavioral assessment
- 90%+ accuracy ML model
- Real-time risk classification
- Doctor recommendation system

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
cd src
python train_model.py
```

### 3. Start Backend
```bash
cd src
uvicorn api:app --reload --port 8000
```

### 4. Open Frontend
```bash
cd frontend
python -m http.server 3000
```
Visit: http://localhost:3000

## Technology Stack
- **Backend**: FastAPI, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Models**: Random Forest, Gradient Boosting, SVM, Logistic Regression

## Project Structure
See folder structure documentation for details.

## License
MIT License