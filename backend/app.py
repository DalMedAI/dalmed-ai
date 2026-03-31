from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import sqlite3
from datetime import datetime

# تحديد مسار مجلد الواجهة (frontend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FOLDER = os.path.join(BASE_DIR, 'frontend')

# تهيئة قاعدة البيانات
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnoses.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            gender TEXT,
            fever INTEGER,
            headache INTEGER,
            eye_pain INTEGER,
            joint_pain INTEGER,
            nausea INTEGER,
            rash INTEGER,
            bleeding INTEGER,
            ear_bleeding INTEGER,
            eye_bleeding INTEGER,
            back_pain INTEGER,
            urine_burning INTEGER,
            infected_in_house INTEGER,
            appetite_loss INTEGER,
            diarrhea INTEGER,
            prediction INTEGER,
            probability REAL,
            severity TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# تهيئة تطبيق Flask
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app)  # السماح بالاتصال من أي واجهة

# الصفحة الرئيسية
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'login.html')

# تقديم أي ملفات ثابتة (CSS, JS, Html)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# تحميل النموذج
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, 'dengue_model.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Please train the model first.")

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

        # 1. جلب القيم من الواجهة (15 عرض + العمر والجنس)
        age_val = float(data.get('age', 30))
        gender_val = float(data.get('gender', 0))
        
        fever_val = float(data.get('fever', 0))
        headache_val = float(data.get('headache', 0))
        fatigue_val = float(data.get('fatigue', 0))
        vomiting_val = float(data.get('vomiting', 0))
        eye_pain_val = float(data.get('eye_pain', 0))
        rash_val = float(data.get('rash', 0))
        joint_pain_val = float(data.get('joint_pain', 0))
        bleeding_val = float(data.get('bleeding', 0))
        chills_val = float(data.get('chills', 0))
        sweating_val = float(data.get('sweating', 0))
        anemia_val = float(data.get('anemia', 0))
        jaundice_val = float(data.get('jaundice', 0))
        abdominal_pain_val = float(data.get('abdominal_pain', 0))
        appetite_loss_val = float(data.get('loss_of_appetite', 0))
        diarrhea_const_val = float(data.get('diarrhea_constipation', 0))

        # 2. إنشاء Dataframe بالترتيب الصحيح للأعمدة (17 feature)
        features_df = pd.DataFrame([{
            'age': age_val,
            'gender': gender_val,
            'fever': fever_val,
            'headache': headache_val,
            'fatigue': fatigue_val,
            'vomiting': vomiting_val,
            'eye_pain': eye_pain_val,
            'rash': rash_val,
            'joint_pain': joint_pain_val,
            'bleeding': bleeding_val,
            'chills': chills_val,
            'sweating': sweating_val,
            'anemia': anemia_val,
            'jaundice': jaundice_val,
            'abdominal_pain': abdominal_pain_val,
            'loss_of_appetite': appetite_loss_val,
            'diarrhea_constipation': diarrhea_const_val
        }])

        # 3. التنبؤ بالاحتمالات
        prob = model.predict_proba(features_df)[0]
        prediction = int(np.argmax(prob))

        # mapping: {0: "Dengue", 1: "Malaria", 2: "Typhoid", 3: "Flu", 4: "COVID"}
        disease_names = ["حمى الضنك", "الملاريا", "التايفويد", "الإنفلونزا", "كوفيد-19"]
        
        # 4. تحديد الرسالة ومستوى الخطورة
        max_prob = float(prob[prediction])
        if max_prob > 0.75:
            severity_level = "High"
        elif max_prob > 0.4:
            severity_level = "Medium"
        else:
            severity_level = "Low"

        detected_disease = disease_names[prediction] if prediction < len(disease_names) else "غير معروف"
        message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية الإصابة بـ {detected_disease} بنسبة ({round(max_prob * 100)}%)."

        result = {
            "prediction": prediction,
            "probability": round(max_prob * 100, 2),
            "message": message,
            "severity_level": severity_level,
            "disease_breakdown": {
                "dengue": round(float(prob[0]) * 100, 2) if len(prob) > 0 else 0,
                "malaria": round(float(prob[1]) * 100, 2) if len(prob) > 1 else 0,
                "typhoid": round(float(prob[2]) * 100, 2) if len(prob) > 2 else 0,
                "flu": round(float(prob[3]) * 100, 2) if len(prob) > 3 else 0,
                "covid": round(float(prob[4]) * 100, 2) if len(prob) > 4 else 0
            },
            "medical_advice": get_advice(prediction, max_prob),
            "patient_name": patient_name
        }

        # حفظ في قاعدة البيانات
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO diagnoses (
                    patient_name, age, gender, fever, headache, eye_pain, 
                    prediction, probability, severity, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_name, age, "Male" if gender == 0 else "Female",
                fever_val, headache_val, eye_pain_val,
                prediction, round(max_prob * 100, 2),
                severity_level, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()
            print(f"Saved result for {patient_name} to database.")
        except Exception as db_e:
            print(f"Warning: Could not save to database: {db_e}")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# نصائح طبية حسب التشخيص
def get_advice(prediction, probability):
    # mapping: {0: "Dengue", 1: "Malaria", 2: "Typhoid", 3: "Flu", 4: "COVID"}
    if prediction == 0:  # Dengue
        if probability > 0.8: return "حالة حرجة (اشتباه حمى الضنك): توجه للمستشفى فوراً."
        return "اشتباه متوسط (حمى الضنك): يرجى مراجعة الطبيب لعمل فحص دم."
    elif prediction == 1:  # Malaria
        if probability > 0.8: return "حالة حرجة (اشتباه ملاريا): توجه للطوارئ فوراً."
        return "اشتباه متوسط (ملاريا): يرجى عمل فحص الملاريا السريع."
    elif prediction == 2:  # Typhoid
        return "اشتباه (تايفويد): ينصح بعمل فحص فيدال (Widal test) والراحة."
    elif prediction == 3:  # Flu
        return "إنفلونزا موسمية: أكثر من السوائل والراحة التامة."
    elif prediction == 4:  # COVID
        return "اشتباه كوفيد-19: يرجى العزل المنزلي وعمل فحص الـ PCR."
    else:
        return "الأعراض غير محددة حالياً، يرجى مراقبة الحالة والراحة."

# تشغيل السيرفر
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
