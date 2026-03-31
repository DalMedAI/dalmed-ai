import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import numpy as np

# تحديد المسارات برمجياً
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.csv") 
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "dengue_model.pkl")

# قراءة البيانات
if not os.path.exists(DATA_PATH):
    print(f"Error: Dataset not found at {DATA_PATH}")
    exit(1)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# تحويل النصوص لرقم
mappings = {
    "diagnosis": {
        "Dengue": 0,
        "Malaria": 1,
        "Typhoid": 2,
        "Flu": 3,
        "COVID": 4
    }
}

# قائمة الأعمدة المطلوبة للتدريب بالترتيب الصحيح (17 feature)
expected_features = [
    'age', 'gender', 'fever', 'headache', 'fatigue', 'vomiting', 
    'eye_pain', 'rash', 'joint_pain', 'bleeding', 'chills', 
    'sweating', 'anemia', 'jaundice', 'abdominal_pain', 
    'loss_of_appetite', 'diarrhea_constipation'
]

if "diagnosis" in df.columns:
    df["diagnosis"] = df["diagnosis"].map(mappings["diagnosis"])

# تنظيف البيانات
df = df.dropna().drop_duplicates()

if df.empty:
    print("Error: No data left after cleaning. Check mapping keys.")
    exit(1)

# تحديد X و y
X = df[expected_features]
y = df["diagnosis"]

print(f"Training on {len(X.columns)} features: {X.columns.tolist()}")
print(f"Classes found: {df['diagnosis'].unique()}")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Creating XGBoost model...")
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=7,
    random_state=42,
    objective='multi:softprob',
    num_class=5,
    eval_metric="mlogloss",
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
labels = ["Dengue", "Malaria", "Typhoid", "Flu", "COVID"]
present_labels = [labels[i] for i in sorted(np.unique(np.concatenate([y_test, y_pred])))]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_labels, yticklabels=present_labels)
plt.title("Multi-Disease Model Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.close()

# Feature Importance
plt.figure(figsize=(12, 12))
importances = model.feature_importances_
indices = importances.argsort()[::-1]
sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices], palette="viridis")
plt.title("Feature Importance - Comprehensive Dataset")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "feature_importance.png"))
plt.close()

# حفظ الموديل
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved to '{MODEL_SAVE_PATH}'")