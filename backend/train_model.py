import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# قراءة البيانات من CSV
df = pd.read_csv("dengue_dataset.csv")

# تحويل النصوص لرقم
df["ns1_result"] = df["ns1_result"].map({"Negative":0, "Positive":1})
df["igm_result"] = df["igm_result"].map({"Negative":0, "Positive":1})
df["pcr_result"] = df["pcr_result"].map({"Negative":0, "Positive":1})
df["final_diagnosis"] = df["final_diagnosis"].map({"Non-dengue":0, "Confirmed dengue":1})

# تنظيف البيانات
df = df.dropna().drop_duplicates()

# تحديد X و y
X = df.drop(["case_id","final_diagnosis"], axis=1)
y = df["final_diagnosis"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء النموذج
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False
)

# تدريب النموذج
model.fit(X_train, y_train)

# التقييم
y_pred = model.predict(X_test)
print(f"🔥 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-dengue","Dengue"], yticklabels=["Non-dengue","Dengue"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.close()

# Feature Importance
importances = model.feature_importances_
indices = importances.argsort()[::-1]
sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices], palette="viridis")
plt.title("Feature Importances")
plt.xlabel("Relative Importance")
plt.savefig("feature_importance.png")
plt.close()

# حفظ النموذج
joblib.dump(model, "dengue_model.pkl")
print("✅ Model saved to 'dengue_model.pkl'")