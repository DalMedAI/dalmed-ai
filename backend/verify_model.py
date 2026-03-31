import urllib.request
import json

def test_predict(name, features):
    url = "http://127.0.0.1:10000/api/predict"
    data = json.dumps(features).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    print(f"\n--- Test: {name} ---")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"Prediction: {result['prediction']} ({result['probability']}% {result['severity_level']})")
            print(f"Breakdown: {result['disease_breakdown']}")
            print(f"Message: {result['message']}")
    except Exception as e:
        print(f"Request failed: {e}")

# Scenario 1: Typical Dengue
test_predict("Dengue Case", {
    "patient_name": "Ali",
    "age": 25,
    "gender": 0,
    "fever": 1,
    "headache": 1,
    "eye_pain": 1,
    "joint_muscle_pain": 1,
    "nausea_vomiting": 0,
    "rash": 1,
    "bleeding": 1,
    "ear_bleeding": 0,
    "eye_bleeding": 0,
    "back_pain": 0.5,
    "urine_burning": 0,
    "infected_in_house": 1,
    "appetite_loss": 1,
    "diarrhea": 0
})

# Scenario 2: Typical Malaria
test_predict("Malaria Case", {
    "patient_name": "Mona",
    "age": 30,
    "gender": 1,
    "fever": 1,
    "headache": 0.5,
    "eye_pain": 0,
    "joint_muscle_pain": 0.5,
    "nausea_vomiting": 0.5,
    "rash": 0,
    "bleeding": 0,
    "ear_bleeding": 0,
    "eye_bleeding": 0,
    "back_pain": 1,
    "urine_burning": 0,
    "infected_in_house": 0,
    "appetite_loss": 1,
    "diarrhea": 1
})

# Scenario 3: Normal/Mild
test_predict("Normal Case", {
    "patient_name": "Omar",
    "age": 20,
    "gender": 0,
    "fever": 0,
    "headache": 0,
    "eye_pain": 0,
    "joint_muscle_pain": 0,
    "nausea_vomiting": 0,
    "rash": 0,
    "bleeding": 0,
    "ear_bleeding": 0,
    "eye_bleeding": 0,
    "back_pain": 0,
    "urine_burning": 0,
    "infected_in_house": 0,
    "appetite_loss": 0,
    "diarrhea": 0
})
