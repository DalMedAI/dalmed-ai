from datetime import datetime
import os
import sqlite3

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FOLDER = os.path.join(BASE_DIR, "frontend")
DB_PATH = os.path.join(BACKEND_DIR, "diagnoses.db")
MODEL_PATH = os.path.join(BACKEND_DIR, "dengue_model.pkl")

DEFAULT_LABELS = ["Dengue", "Malaria", "Typhoid"]
DENGUE_FEATURES = ["eye_pain", "rash", "joint_pain", "bleeding"]
MALARIA_FEATURES = ["chills", "sweating", "anemia", "jaundice"]
TYPHOID_FEATURES = ["abdominal_pain", "loss_of_appetite", "diarrhea_constipation"]
SYMPTOM_FEATURES = [
    "fever",
    "headache",
    "fatigue",
    "vomiting",
    "eye_pain",
    "rash",
    "joint_pain",
    "bleeding",
    "chills",
    "sweating",
    "anemia",
    "jaundice",
    "abdominal_pain",
    "loss_of_appetite",
    "diarrhea_constipation",
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
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
        """
    )
    conn.commit()
    conn.close()


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: {MODEL_PATH} not found. Please train the model first.")
        return None, DEFAULT_LABELS, []

    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        print(f"Loaded model artifact from {MODEL_PATH}")
        return (
            artifact["model"],
            artifact.get("labels", DEFAULT_LABELS),
            artifact.get("training_features", []),
        )

    print(f"Loaded legacy model from {MODEL_PATH}")
    return artifact, DEFAULT_LABELS, []


def safe_float(data, key, default=0):
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def clamp(value, low, high):
    return min(max(value, low), high)


def build_feature_frame(data, training_features):
    values = {
        "age": clamp(safe_float(data, "age", 30), 0, 120),
        "gender": 1 if safe_float(data, "gender", 0) >= 0.5 else 0,
        "fever": clamp(safe_float(data, "fever", 0), 0, 1),
        "headache": clamp(safe_float(data, "headache", 0), 0, 1),
        "fatigue": clamp(safe_float(data, "fatigue", 0), 0, 1),
        "vomiting": clamp(safe_float(data, "vomiting", 0), 0, 1),
        "eye_pain": clamp(safe_float(data, "eye_pain", 0), 0, 1),
        "rash": clamp(safe_float(data, "rash", 0), 0, 1),
        "joint_pain": clamp(safe_float(data, "joint_pain", 0), 0, 1),
        "bleeding": clamp(safe_float(data, "bleeding", 0), 0, 1),
        "chills": clamp(safe_float(data, "chills", 0), 0, 1),
        "sweating": clamp(safe_float(data, "sweating", 0), 0, 1),
        "anemia": clamp(safe_float(data, "anemia", 0), 0, 1),
        "jaundice": clamp(safe_float(data, "jaundice", 0), 0, 1),
        "abdominal_pain": clamp(safe_float(data, "abdominal_pain", 0), 0, 1),
        "loss_of_appetite": clamp(safe_float(data, "loss_of_appetite", 0), 0, 1),
        "diarrhea_constipation": clamp(safe_float(data, "diarrhea_constipation", 0), 0, 1),
        "fever_days": clamp(safe_float(data, "fever_days", 5), 0, 21),
    }

    values["symptom_count"] = sum(values[name] for name in SYMPTOM_FEATURES)
    values["dengue_signature_score"] = sum(values[name] for name in DENGUE_FEATURES)
    values["malaria_signature_score"] = sum(values[name] for name in MALARIA_FEATURES)
    values["typhoid_signature_score"] = sum(values[name] for name in TYPHOID_FEATURES)
    signature_scores = [
        values["dengue_signature_score"],
        values["malaria_signature_score"],
        values["typhoid_signature_score"],
    ]
    values["competing_signature_count"] = sum(score >= 2 for score in signature_scores)
    values["overlap_penalty"] = max(values["competing_signature_count"] - 1, 0)

    if not training_features:
        training_features = [
            "age",
            "gender",
            "fever",
            "headache",
            "fatigue",
            "vomiting",
            "eye_pain",
            "rash",
            "joint_pain",
            "bleeding",
            "chills",
            "sweating",
            "anemia",
            "jaundice",
            "abdominal_pain",
            "loss_of_appetite",
            "diarrhea_constipation",
        ]

    return pd.DataFrame([{feature: values.get(feature, 0) for feature in training_features}]), values


def validate_symptom_logic(values):
    warnings = []
    positive_count = sum(1 for name in SYMPTOM_FEATURES if values[name] >= 0.75)
    symptom_load = sum(values[name] for name in SYMPTOM_FEATURES)
    selected_count = sum(1 for name in SYMPTOM_FEATURES if values[name] > 0)

    if positive_count == len(SYMPTOM_FEATURES) or symptom_load >= 13:
        return {
            "can_predict": False,
            "reason": "unrealistic_all_yes",
            "message": "المدخلات غير واقعية طبياً، لا يمكن إنشاء تشخيص دقيق.",
            "warnings": ["المدخلات غير واقعية طبياً، لا يمكن إنشاء تشخيص دقيق."],
        }

    if selected_count == 0 or symptom_load == 0:
        return {
            "can_predict": False,
            "reason": "all_no",
            "message": "الأعراض المدخلة غير كافية لإنشاء تشخيص طبي.",
            "warnings": ["الأعراض المدخلة غير كافية لإنشاء تشخيص طبي."],
        }

    if symptom_load < 2 or selected_count < 2:
        return {
            "can_predict": False,
            "reason": "insufficient_symptoms",
            "message": "الأعراض المدخلة غير كافية لإنشاء تشخيص طبي.",
            "warnings": ["الأعراض المدخلة غير كافية لإنشاء تشخيص طبي."],
        }

    if values["competing_signature_count"] >= 2:
        warnings.append("توجد أعراض متداخلة بين أكثر من مرض، لذلك تم خفض مستوى الثقة.")

    if values["fever"] < 0.5:
        warnings.append("غياب الحمى يقلل موثوقية التشخيص لهذه الأمراض.")

    return {
        "can_predict": True,
        "reason": "valid",
        "message": "",
        "warnings": warnings,
    }


def adjust_probabilities(raw_probabilities, values):
    raw = np.asarray(raw_probabilities[:3], dtype=float)
    raw = raw / raw.sum()
    signature = np.array(
        [
            values["dengue_signature_score"] / len(DENGUE_FEATURES),
            values["malaria_signature_score"] / len(MALARIA_FEATURES),
            values["typhoid_signature_score"] / len(TYPHOID_FEATURES),
        ],
        dtype=float,
    )

    if signature.sum() > 0:
        signature_prior = signature / signature.sum()
        adjusted = (0.72 * raw) + (0.28 * signature_prior)
    else:
        adjusted = raw

    if values["overlap_penalty"] > 0 or values["symptom_count"] >= 9:
        uniform = np.ones(3) / 3
        adjusted = (0.55 * adjusted) + (0.45 * uniform)

    adjusted = np.maximum(adjusted, 0.001)
    adjusted = adjusted / adjusted.sum()

    if values["competing_signature_count"] >= 2:
        top_index = int(np.argmax(adjusted))
        top_value = float(adjusted[top_index])
        if top_value > 0.68:
            excess = top_value - 0.68
            adjusted[top_index] = 0.68
            other_indices = [index for index in range(3) if index != top_index]
            adjusted[other_indices] += excess / 2
            adjusted = adjusted / adjusted.sum()

    return adjusted


def confidence_label(probability, warnings):
    if warnings and probability < 0.85:
        probability = min(probability, 0.74)
    if probability >= 0.75:
        return "High Confidence"
    if probability >= 0.45:
        return "Medium Confidence"
    return "Low Confidence"


def get_advice(prediction, probability):
    if prediction == 0:
        if probability >= 0.75:
            return "Dengue indicators are strong. Seek medical assessment and consider CBC/platelet testing."
        return "Dengue is possible. Monitor warning signs such as bleeding, severe abdominal pain, or persistent vomiting."
    if prediction == 1:
        if probability >= 0.75:
            return "Malaria indicators are strong. A rapid malaria test or blood smear is recommended urgently."
        return "Malaria is possible. Testing is recommended, especially with chills, sweating, anemia, or jaundice."
    if prediction == 2:
        if probability >= 0.75:
            return "Typhoid indicators are strong. Medical review and appropriate laboratory testing are recommended."
        return "Typhoid is possible. Review hydration, gastrointestinal symptoms, and fever duration with a clinician."
    return "Symptoms are not specific enough. Please consult a healthcare professional for proper evaluation."


def generate_medical_recommendations(disease, confidence):
    confidence_value = round(float(confidence), 2)
    disease_key = (disease or "").strip().lower()
    is_uncertain = confidence_value < 50
    uncertainty_note = "النتيجة غير مؤكدة لأن نسبة الثقة أقل من 50%. يُنصح بإجراء فحوصات مخبرية ومراجعة الطبيب."

    recommendations = {
        "diagnosis": disease or "غير محدد",
        "confidence": f"{confidence_value}%",
        "recommended_medications": [],
        "avoid_medications": [],
        "home_care": [],
        "warning": "",
        "disclaimer": "هذه التوصيات إرشادية فقط ولا تؤكد التشخيص ولا تغني عن مراجعة الطبيب أو إجراء الفحوصات الطبية."
    }

    if disease_key == "dengue":
        recommendations.update({
            "diagnosis": "حمى الضنك",
            "recommended_medications": [
                "باراسيتامول لتخفيف الحمى أو الألم حسب الجرعة المناسبة",
                "محاليل الإماهة أو السوائل عند الحاجة"
            ],
            "avoid_medications": [
                "الأسبرين",
                "الإيبوبروفين",
                "الديكلوفيناك"
            ],
            "home_care": [
                "الراحة التامة",
                "الإكثار من السوائل",
                "مراقبة الحرارة وأي علامات نزيف"
            ],
            "warning": "راجع الطبيب فوراً عند ظهور نزيف، ألم شديد بالبطن، قيء مستمر، دوخة شديدة، أو تدهور عام."
        })
    elif disease_key == "malaria":
        recommendations.update({
            "diagnosis": "الملاريا",
            "recommended_medications": [
                "استشارة طبية لتأكيد الحالة",
                "Artemether/Lumefantrine تحت إشراف طبي فقط"
            ],
            "avoid_medications": [
                "تجنب أدوية الملاريا دون وصفة طبية",
                "تجنب إيقاف العلاج مبكراً إذا وصفه الطبيب"
            ],
            "home_care": [
                "الراحة",
                "الإكثار من السوائل",
                "متابعة الحرارة والقشعريرة والتعرق"
            ],
            "warning": "راجع الطوارئ عند حدوث تشنجات، اصفرار شديد، فقر دم واضح، اضطراب وعي، أو حمى شديدة مستمرة."
        })
    elif disease_key == "typhoid":
        recommendations.update({
            "diagnosis": "التايفويد",
            "recommended_medications": [
                "Azithromycin تحت إشراف طبي",
                "Ceftriaxone تحت إشراف طبي عند الحاجة"
            ],
            "avoid_medications": [
                "تجنب المضادات الحيوية دون وصفة",
                "تجنب أدوية إيقاف الإسهال دون استشارة الطبيب"
            ],
            "home_care": [
                "الإكثار من السوائل",
                "تناول غذاء خفيف وجيد",
                "الراحة ومراقبة مدة الحمى وأعراض الجهاز الهضمي"
            ],
            "warning": "راجع الطبيب فوراً عند استمرار الحمى، ألم شديد بالبطن، جفاف، دم في البراز، أو تدهور الحالة."
        })
    else:
        recommendations.update({
            "diagnosis": "غير محدد",
            "recommended_medications": [
                "استشارة طبيب قبل استخدام أي علاج"
            ],
            "avoid_medications": [
                "تجنب العلاج الذاتي أو استخدام مضادات حيوية دون وصفة"
            ],
            "home_care": [
                "الراحة",
                "شرب السوائل",
                "مراقبة الأعراض"
            ],
            "warning": "النتيجة غير كافية لتوجيه توصية مرضية محددة. يُنصح بمراجعة الطبيب وإجراء الفحوصات المناسبة."
        })

    if is_uncertain:
        recommendations["warning"] = f"{uncertainty_note} {recommendations['warning']}"

    return recommendations


def no_prediction_result(patient_name, message, warnings):
    return {
        "prediction": None,
        "predicted_disease": None,
        "probability": 0,
        "message": message,
        "severity_level": "No Prediction",
        "disease_breakdown": {
            "dengue": 0,
            "malaria": 0,
            "typhoid": 0,
        },
        "warnings": warnings,
        "medical_advice": message,
        "medical_recommendations": {
            "diagnosis": "غير محدد",
            "confidence": "0%",
            "recommended_medications": [],
            "avoid_medications": [],
            "home_care": [],
            "warning": message,
            "disclaimer": "هذه التوصيات إرشادية فقط ولا تؤكد التشخيص ولا تغني عن مراجعة الطبيب أو إجراء الفحوصات الطبية."
        },
        "patient_name": patient_name,
        "no_prediction": True,
    }


init_db()
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path="")
CORS(app)
model, MODEL_LABELS, TRAINING_FEATURES = load_model()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "login.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify(
        {
            "status": "online",
            "model_loaded": model is not None,
            "labels": MODEL_LABELS,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Contact administrator."}), 500

    try:
        data = request.get_json() or {}
        patient_name = data.get("patient_name", "Patient")
        features_df, feature_values = build_feature_frame(data, TRAINING_FEATURES)

        validation = validate_symptom_logic(feature_values)
        validation_warnings = validation["warnings"]
        if not validation["can_predict"]:
            return jsonify(
                no_prediction_result(
                    patient_name,
                    validation["message"],
                    validation_warnings,
                )
            )

        raw_probabilities = model.predict_proba(features_df)[0]
        probabilities = adjust_probabilities(raw_probabilities, feature_values)
        prediction = int(np.argmax(probabilities))
        detected_disease = MODEL_LABELS[prediction] if prediction < len(MODEL_LABELS) else DEFAULT_LABELS[prediction]
        probability = float(probabilities[prediction])
        probability_percent = round(probability * 100, 2)
        severity_level = confidence_label(probability, validation_warnings)

        result = {
            "prediction": prediction,
            "predicted_disease": detected_disease,
            "probability": probability_percent,
            "message": (
                f"{patient_name}, the symptom pattern is most consistent with "
                f"{detected_disease} at {probability_percent}%. This is a screening result, "
                "not a final diagnosis."
            ),
            "severity_level": severity_level,
            "disease_breakdown": {
                "dengue": round(float(probabilities[0]) * 100, 2),
                "malaria": round(float(probabilities[1]) * 100, 2),
                "typhoid": round(float(probabilities[2]) * 100, 2),
            },
            "warnings": validation_warnings,
            "medical_advice": get_advice(prediction, probability),
            "medical_recommendations": generate_medical_recommendations(detected_disease, probability_percent),
            "patient_name": patient_name,
        }

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO diagnoses (
                    patient_name, age, gender, fever, headache, eye_pain,
                    prediction, probability, severity, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_name,
                    int(feature_values["age"]),
                    "Male" if feature_values["gender"] == 0 else "Female",
                    feature_values["fever"],
                    feature_values["headache"],
                    feature_values["eye_pain"],
                    prediction,
                    probability_percent,
                    severity_level,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Warning: Could not save diagnosis: {db_error}")

        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
