import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'dengue_model.pkl')

model = joblib.load(MODEL_PATH)

# Scenario: Malaria Case
# symptoms: fever=1, headache=0.5, joint=0.5, back=1, appetite=1, diarrhea=1
features = {
    'age': 30,
    'gender': 0, # Female mapping
    'fever': 1.0,
    'headache': 0.5,
    'eye_pain': 0.0,
    'joint_muscle_pain': 0.5,
    'nausea_vomiting': 0.5,
    'rash': 0.0,
    'bleeding': 0.0,
    'ear_bleeding': 0.0,
    'eye_bleeding': 0.0,
    'back_pain': 1.0,
    'urine_burning': 0.0,
    'infected_in_house': 0.0,
    'appetite_loss': 1.0,
    'diarrhea': 1.0,
    'ns1_result': 0,
    'igm_result': 0,
    'pcr_result': 0,
    'temperature': 39.0, # Approximate
    'fever_days': 5.0    # Approximate
}

# The feature order MUST match:
feature_order = ['age', 'gender', 'fever', 'headache', 'eye_pain', 'joint_muscle_pain', 'nausea_vomiting', 'rash', 'bleeding', 'ear_bleeding', 'eye_bleeding', 'back_pain', 'urine_burning', 'infected_in_house', 'appetite_loss', 'diarrhea', 'ns1_result', 'igm_result', 'pcr_result', 'temperature', 'fever_days']

df = pd.DataFrame([features])[feature_order]
prob = model.predict_proba(df)[0]
print(f"Probabilities: {prob}")
print(f"Prediction: {np.argmax(prob)}")
