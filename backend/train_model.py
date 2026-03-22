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
DATA_PATH = os.path.join(BASE_DIR, "dengue_malaria_dataset.xlsx")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "dengue_model.pkl")

# قراءة البيانات (دعم اكسل)
if not os.path.exists(DATA_PATH):
    print(f"Error: Dataset not found at {DATA_PATH}")
    exit(1)

print(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    print(f"Error reading Excel: {e}. Trying CSV...")
    df = pd.read_csv(DATA_PATH.replace(".xlsx", ".csv"))

# تحويل النصوص لرقم
def robust_map(series, mapping):
    return series.astype(str).str.strip().map(mapping)

mappings = {
    "ns1_result": {"Negative": 0, "Positive": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1, "nan": 0},
    "igm_result": {"Negative": 0, "Positive": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1, "nan": 0},
    "pcr_result": {"Negative": 0, "Positive": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1, "nan": 0},
    "final_diagnosis": {"Non-dengue": 0, "Confirmed dengue": 1, "Malaria": 2}
}

if "gender" in df.columns:
    df["gender"] = robust_map(df["gender"], {"Female": 0, "Male": 1})

for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = robust_map(df[col], mapping)

# تنظيف البيانات
df = df.dropna().drop_duplicates()

if df.empty:
    print("Error: No data left after cleaning. Check mapping keys.")
    exit(1)

# تحديد X و y
columns_to_drop = ["final_diagnosis"]
if "case_id" in df.columns:
    columns_to_drop.append("case_id")
X = df.drop(columns_to_drop, axis=1)
y = df["final_diagnosis"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Creating Multi-class model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective='multi:softprob',
    num_class=3,
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
plt.figure(figsize=(8, 6))
labels = ["Normal", "Dengue", "Malaria"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels[:len(np.unique(y))], yticklabels=labels[:len(np.unique(y))])
plt.title("Multi-class Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.close()

# Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = importances.argsort()[::-1]
sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices], hue=[X.columns[i] for i in indices], palette="viridis", legend=False)
plt.title("Feature Importance - Multi-class")
plt.xlabel("Relative Importance")
plt.savefig(os.path.join(BASE_DIR, "feature_importance.png"))
plt.close()

# حفظ الموديل
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved to '{MODEL_SAVE_PATH}'")