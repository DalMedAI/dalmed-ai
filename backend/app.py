from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# Get absolute path to frontend folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FOLDER = os.path.join(BASE_DIR, 'frontend')

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app) # Enable CORS for all routes

@app.route('/')
def index():
    return app.send_static_file('login.html')

MODEL_PATH = 'dengue_model.pkl'

# Load model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Please train the model first.")

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Contact administrator."}), 500
        
    try:
        data = request.json
        
        # Demographic features
        age = float(data.get('age', 30))
        gender = float(data.get('gender', 0))
        patient_name = data.get('patient_name', 'مريض')
        other_symptoms = data.get('other_symptoms', '')
        
        # Symptoms
        features = [
            age,
            gender,
            float(data.get('fever', 0)),
            float(data.get('headache', 0)),
            float(data.get('eye_pain', 0)),
            float(data.get('joint_muscle_pain', 0)),
            float(data.get('nausea_vomiting', 0)),
            float(data.get('rash', 0)),
            float(data.get('bleeding', 0))
        ]
        
        # Reshape for single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Predict probability for all classes (0: Normal, 1: Dengue, 2: Malaria)
        prob = model.predict_proba(features_array)[0]
        prediction = int(model.predict(features_array)[0])
        
        normal_prob = float(prob[0])
        dengue_prob = float(prob[1]) if len(prob) > 1 else 0.0
        malaria_prob = float(prob[2]) if len(prob) > 2 else 0.0
        
        # Determine highest probability illness (excluding normal if we just want highest disease, but let's take overall highest)
        max_disease_prob = max(dengue_prob, malaria_prob)
        # Severity is based on the highest disease probability
        if max_disease_prob > 0.75:
            severity_level = "High"
        elif max_disease_prob > 0.4:
            severity_level = "Medium"
        else:
            severity_level = "Low"
            
        if prediction == 1:
            message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية مرتفعة لحمى الضنك."
        elif prediction == 2:
            message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية مرتفعة للإصابة بالملاريا."
        else:
            message = f"أهلاً {patient_name}، تشير الأعراض إلى إرهاق أو نزلة برد عادية، الاحتمالية منخفضة للأمراض الخطيرة."
        
        result = {
            "prediction": prediction,
            "probability": round(max_disease_prob * 100, 2), # Main risk probability
            "message": message,
            "severity_level": severity_level,
            "disease_breakdown": {
                "dengue": round(dengue_prob * 100, 2),
                "malaria": round(malaria_prob * 100, 2),
                "normal": round(normal_prob * 100, 2)
            },
            "medical_advice": get_advice(prediction, max_disease_prob),
            "patient_name": patient_name
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def get_advice(prediction, probability):
    """Returns dynamic advice based on the model's prediction."""
    if prediction == 1: # Dengue
        if probability > 0.8:
            return "حالة حرجة (اشتباه حمى الضنك): يُرجى التوجه إلى أقرب مستشفى أو طوارئ فوراً. تجنب تناول المسكنات التي تحتوي على الأسبرين أو الإيبوبروفين واستخدم الباراسيتامول فقط."
        else:
            return "اشتباه باحتمالية متوسطة (حمى الضنك): يُنصح بزيارة الطبيب لإجراء فحص دم. احرص على شرب كميات وفيرة من السوائل والراحة التامة."
    elif prediction == 2: # Malaria
        if probability > 0.8:
            return "حالة حرجة (اشتباه ملاريا): يُرجى التوجه للطوارئ لإجراء فحص ملاريا (Blood Film) وتلقي العلاج المضاد للملاريا فوراً."
        else:
            return "اشتباه باحتمالية متوسطة (ملاريا): يُنصح بزيارة الطبيب لإجراء الفحوصات اللازمة لا سيما إذا كنت في منطقة يكثر فيها البعوض."
    else: # Normal
        if probability > 0.3: # There is some disease risk
            return "الوضع مستقر: الأعراض قد تكون ناتجة عن إرهاق أو زكام. نرجو مراقبة الأعراض وإذا تفاقمت الحمى يُرجى مراجعة الطبيب."
        else:
            return "حفظك الله: الأعراض لا تشير إلى حمى الضنك أو الملاريا باحتمالية عالية. احرص على الراحة وتناول السوائل."

if __name__ == '__main__':
        port = int(os.environ.get("PORT", 10000))
        app.run(host="0.0.0.0", port=port)