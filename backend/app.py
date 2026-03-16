from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

# تحديد مسار مجلد الواجهة (frontend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FOLDER = os.path.join(BASE_DIR, 'frontend')

# تهيئة تطبيق Flask
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app)  # السماح بالاتصال من أي واجهة

# الصفحة الرئيسية
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# تقديم أي ملفات ثابتة (CSS, JS, Html)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# تحميل النموذج
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, 'dengue_model.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"⚠️ Warning: {MODEL_PATH} not found. Please train the model first.")

# API لفحص حالة السيرفر
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    })

# API للتنبؤ
@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Contact administrator."}), 500

    try:
        data = request.get_json()

        # بيانات ديموغرافية
        age = float(data.get('age', 30))
        gender = float(data.get('gender', 0))
        patient_name = data.get('patient_name', 'مريض')
        other_symptoms = data.get('other_symptoms', '')

        # الأعراض من الواجهة
        fever_val = float(data.get('fever', 0))
        headache_val = float(data.get('headache', 0))
        eye_pain_val = float(data.get('eye_pain', 0))
        joint_pain_val = float(data.get('joint_muscle_pain', 0))
        nausea_val = float(data.get('nausea_vomiting', 0))
        rash_val = float(data.get('rash', 0))
        bleeding_val = float(data.get('bleeding', 0))
        
        symptom_score = fever_val + headache_val + eye_pain_val + joint_pain_val + nausea_val + rash_val + bleeding_val

        # تحويل الجنس (الواجهة 0 ذكر، النموذج 1 ذكر)
        gender_model = 1 if gender == 0 else 0
        
        # استنتاج وهمي للنتائج المخبرية والحرارة بناءً على الأعراض لأن الواجهة لا توفرها
        temperature = 38.5 + (symptom_score * 0.2) if fever_val else 37.0
        fever_days = 3 + (symptom_score * 0.5) if fever_val else 0

        # إنشاء Dataframe بنفس أسماء الأعمدة في التدريب
        features_df = pd.DataFrame([{
            'ns1_result': 1 if symptom_score > 3 else 0,
            'igm_result': 1 if symptom_score > 2 else 0,
            'pcr_result': 1 if symptom_score > 4 else 0,
            'age': age,
            'gender': gender_model,
            'temperature': min(temperature, 41.0),
            'fever_days': min(fever_days, 14.0)
        }])

        # التنبؤ بالاحتمالات
        prob = model.predict_proba(features_df)[0]
        prediction = int(model.predict(features_df)[0])

        normal_prob = float(prob[0])
        dengue_prob = float(prob[1]) if len(prob) > 1 else 0.0
        malaria_prob = float(prob[2]) if len(prob) > 2 else 0.0

        # تحديد مستوى الخطورة
        max_disease_prob = max(dengue_prob, malaria_prob)
        if max_disease_prob > 0.75:
            severity_level = "High"
        elif max_disease_prob > 0.4:
            severity_level = "Medium"
        else:
            severity_level = "Low"

        # الرسالة حسب التشخيص
        if prediction == 1:
            message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية مرتفعة لحمى الضنك."
        elif prediction == 2:
            message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية مرتفعة للإصابة بالملاريا."
        else:
            message = f"أهلاً {patient_name}، تشير الأعراض إلى إرهاق أو نزلة برد عادية، الاحتمالية منخفضة للأمراض الخطيرة."

        result = {
            "prediction": prediction,
            "probability": round(max_disease_prob * 100, 2),
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

# نصائح طبية حسب التشخيص
def get_advice(prediction, probability):
    if prediction == 1:  # Dengue
        if probability > 0.8:
            return "حالة حرجة (اشتباه حمى الضنك): يُرجى التوجه إلى أقرب مستشفى فوراً."
        else:
            return "اشتباه متوسط (حمى الضنك): يُنصح بزيارة الطبيب وإجراء فحص دم."
    elif prediction == 2:  # Malaria
        if probability > 0.8:
            return "حالة حرجة (اشتباه ملاريا): يُرجى التوجه للطوارئ فوراً."
        else:
            return "اشتباه متوسط (ملاريا): يُنصح بزيارة الطبيب لإجراء الفحوصات."
    else:  # Normal
        if probability > 0.3:
            return "الوضع مستقر: الأعراض قد تكون ناتجة عن إرهاق أو زكام."
        else:
            return "الأعراض لا تشير إلى أمراض خطيرة. احرص على الراحة وتناول السوائل."

# تشغيل السيرفر
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
