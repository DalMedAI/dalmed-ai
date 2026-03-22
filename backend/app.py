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

        # 1. تعريف أوزان الأعراض (لكل عرض ثقله الطبي)
        weights = {
            'fever': 1.0,
            'headache': 0.5,
            'eye_pain': 0.6,
            'joint_muscle_pain': 0.8,
            'nausea_vomiting': 0.7,
            'rash': 1.2,
            'bleeding': 1.5
        }

        # 2. جلب القيم من الواجهة
        gender_model = 1 if gender == 0 else 0
        fever_val = float(data.get('fever', 0))
        headache_val = float(data.get('headache', 0))
        eye_pain_val = float(data.get('eye_pain', 0))
        joint_pain_val = float(data.get('joint_muscle_pain', 0))
        nausea_val = float(data.get('nausea_vomiting', 0))
        rash_val = float(data.get('rash', 0))
        bleeding_val = float(data.get('bleeding', 0))

        # 3. حساب النقاط الموزونة (Weighted Score)
        # تم تعديل الأوزان لتكون أكثر توازناً وتقليص فجوة "الحمى"
        symptom_score = (
            fever_val * weights['fever'] +
            headache_val * weights['headache'] +
            eye_pain_val * weights['eye_pain'] +
            joint_pain_val * weights['joint_muscle_pain'] +
            nausea_val * weights['nausea_vomiting'] +
            rash_val * weights['rash'] +
            bleeding_val * weights['bleeding']
        )

        # 4. تحليل النص الذكي (Other Symptoms)
        # البحث عن أعراض في النص حتى لو لم يتم اختيارها في القائمة
        text_weights = {
            'حمى': 0.8, 'حرارة': 0.8, 'fever': 0.8, 'temp': 0.8,
            'نزيف': 1.2, 'دم': 1.0, 'bleeding': 1.2, 'blood': 1.0,
            'طفح': 1.0, 'جلد': 0.5, 'rash': 1.0, 'skin': 0.5,
            'ألم': 0.4, 'وجع': 0.4, 'pain': 0.4, 'ache': 0.4,
            'تعب': 0.3, 'إرهاق': 0.3, 'tired': 0.3, 'fatigue': 0.3,
            'صدمة': 1.5, 'shock': 1.5
        }
        
        other_symptoms_lower = other_symptoms.lower()
        text_boost = 0
        for word, weight in text_weights.items():
            if word in other_symptoms_lower:
                text_boost += weight
        
        # دمج دعم النص بحد أقصى لمنع المبالغة
        symptom_score += min(text_boost, 2.0)

        # 5. استنتاج المؤشرات الحيوية بشكل تدريجي (Continuous Inference)
        # إلغاء القفزة المفاجئة للحرارة بناءً على "حمى" فقط، وجعلها تعتمد على مجموع الأعراض
        # الحرارة الأساسية 37.0 + تأثير مجموع الأعراض
        temperature = 37.2 + (symptom_score * 0.4)
        
        # عدد أيام الحمى (تقديري بناءً على شدة الحالة)
        fever_days = (symptom_score * 0.8) if symptom_score > 0 else 0

        # 6. توليف نتائج المختبر (بناءً على التوزان الجديد)
        # إذا كان المجموع الموزون مرتفعاً، تزيد احتمالية النتائج الإيجابية
        ns1_val = 1 if (symptom_score > 2.2) else 0
        igm_val = 1 if (symptom_score > 1.5) else 0
        pcr_val = 1 if (symptom_score > 3.0) else 0

        # إنشاء Dataframe بنفس أسماء الأعمدة في التدريب
        features_df = pd.DataFrame([{
            'ns1_result': ns1_val,
            'igm_result': igm_val,
            'pcr_result': pcr_val,
            'age': age,
            'gender': gender_model,
            'temperature': min(temperature, 41.5),
            'fever_days': min(fever_days, 14.0)
        }])

        # التنبؤ بالاحتمالات من الموديل الجديد (3 فئات: 0:طبيعي، 1:ضنك، 2:ملاريا)
        prob = model.predict_proba(features_df)[0]
        prediction = int(np.argmax(prob)) # اختيار الفئة الأعلى احتمالية

        normal_prob = float(prob[0])
        dengue_prob = float(prob[1]) if len(prob) > 1 else 0.0
        malaria_prob = float(prob[2]) if len(prob) > 2 else 0.0

        # تحديد مستوى الخطورة بناءً على أعلى عدوى
        active_infection_prob = max(dengue_prob, malaria_prob)
        if active_infection_prob > 0.75:
            severity_level = "High"
        elif active_infection_prob > 0.4:
            severity_level = "Medium"
        else:
            severity_level = "Low"

        # الرسالة حسب التشخيص
        if prediction == 1:
            message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية مرتفعة لحمى الضنك ({round(dengue_prob*100)}%)."
        elif prediction == 2:
            message = f"أهلاً {patient_name}، المؤشرات تدل على احتمالية إصابة بالملاريا ({round(malaria_prob*100)}%)."
        else:
            message = f"أهلاً {patient_name}، المؤشرات طبيعية أو تشير لإرهاق بسيط."

        result = {
            "prediction": prediction,
            "probability": round(max(dengue_prob, malaria_prob, normal_prob) * 100, 2),
            "message": message,
            "severity_level": severity_level,
            "disease_breakdown": {
                "dengue": round(dengue_prob * 100, 2),
                "malaria": round(malaria_prob * 100, 2),
                "normal": round(normal_prob * 100, 2)
            },
            "medical_advice": get_advice(prediction, active_infection_prob),
            "patient_name": patient_name
        }

        # حفظ في قاعدة البيانات
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO diagnoses (
                    patient_name, age, gender, fever, headache, eye_pain, 
                    joint_pain, nausea, rash, bleeding, prediction, 
                    probability, severity, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_name, age, "Male" if gender == 0 else "Female",
                int(fever_val), int(headache_val), int(eye_pain_val),
                int(joint_pain_val), int(nausea_val), int(rash_val),
                int(bleeding_val), prediction, round(active_infection_prob * 100, 2),
                severity_level, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()
            print(f"✅ Saved result for {patient_name} to database.")
        except Exception as db_e:
            print(f"⚠️ Warning: Could not save to database: {db_e}")

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
