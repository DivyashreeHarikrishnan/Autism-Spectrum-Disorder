# Model Documentation

## Overview
The ASD Detection System uses an ensemble machine learning approach combining multiple algorithms for robust predictions.

## Model Architecture

### Base Models
1. **Random Forest Classifier**
   - 200 estimators
   - Default parameters
   - Good for feature importance

2. **Gradient Boosting Classifier**
   - 200 estimators
   - Learning rate: 0.05
   - Sequential error correction

3. **Logistic Regression**
   - With StandardScaler
   - Linear decision boundary
   - Interpretable coefficients

4. **Support Vector Machine (SVM)**
   - RBF kernel
   - With StandardScaler
   - Non-linear decision boundary

### Ensemble Method
- **Soft Voting Classifier**
  - Combines probability predictions
  - Weighted averaging
  - More robust than single models

### Calibration
- **CalibratedClassifierCV**
  - Isotonic calibration method
  - 3-fold cross-validation
  - Better probability estimates

## Features

### Original Features (10)
1. eye_contact
2. responds_name
3. points_to_objects
4. pretend_play
5. repetitive_behaviour
6. sensory_sensitivity
7. prefers_alone
8. gestures
9. delayed_speech
10. restricted_interests

### Engineered Features (4)
1. **social_score**: eye_contact + responds_name + points_to_objects + gestures
2. **communication_score**: pretend_play + delayed_speech + responds_name
3. **sensory_score**: sensory_sensitivity + repetitive_behaviour + restricted_interests
4. **overall_risk_score**: sum of all scores

**Total Features**: 14

## Training Process

### Data Split
- Training: 80%
- Testing: 20%
- Stratified split (balanced classes)

### Cross-Validation
- 3-fold stratified CV
- Used for calibration

### Training Time
- Approximately 2-5 minutes
- Depends on hardware

## Performance Metrics

Target metrics (from trained model):
- **Accuracy**: ≥ 90%
- **Precision**: ≥ 90%
- **Recall**: ≥ 90%
- **F1-Score**: ≥ 90%
- **ROC-AUC**: ≥ 0.95

## Risk Classification
```python
if probability < 0.30:
    risk_level = "Low"
elif probability < 0.70:
    risk_level = "Medium"
else:
    risk_level = "High"
```

## Model Files

Generated after training:
- `asd_model.joblib` - Trained model
- `features_list.joblib` - Feature names
- `metrics.json` - Performance metrics
- `confusion_matrix.png` - Visual evaluation
- `roc_curve.png` - ROC curve plot
- `feature_importance.png` - Feature rankings

## Limitations

1. **Age Range**: Designed for 18-36 months
2. **Screening Only**: Not diagnostic
3. **Binary Output**: ASD indicators present/absent
4. **Observer Dependent**: Relies on caregiver observations
5. **Cultural Factors**: May not account for all cultural variations

## Future Improvements

1. Multi-class classification (severity levels)
2. Age-specific models
3. Temporal tracking (multiple assessments)
4. Integration of additional features
5. Explainable AI (SHAP values)
6. Regular retraining with new data