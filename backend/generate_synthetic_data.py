import pandas as pd
import numpy as np
import os

# تحديد المسار
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "dengue_malaria_dataset.xlsx")

np.random.seed(42)
n_samples = 200

def generate_normal(n):
    return pd.DataFrame({
        'ns1_result': np.zeros(n),
        'igm_result': np.zeros(n),
        'pcr_result': np.zeros(n),
        'age': np.random.randint(5, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'temperature': np.random.normal(37.0, 0.3, n),
        'fever_days': np.random.randint(0, 3, n),
        'final_diagnosis': 'Non-dengue'
    })

def generate_dengue(n):
    return pd.DataFrame({
        'ns1_result': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'igm_result': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'pcr_result': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'age': np.random.randint(5, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'temperature': np.random.normal(39.0, 0.7, n),
        'fever_days': np.random.randint(3, 10, n),
        'final_diagnosis': 'Confirmed dengue'
    })

def generate_malaria(n):
    return pd.DataFrame({
        'ns1_result': np.zeros(n),  # الملاريا ما عندها NS1
        'igm_result': np.zeros(n),
        'pcr_result': np.zeros(n),
        'age': np.random.randint(5, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'temperature': np.random.normal(40.0, 0.8, n), # حرارة أعلى
        'fever_days': np.random.randint(5, 14, n),
        'final_diagnosis': 'Malaria'
    })

df_normal = generate_normal(n_samples)
df_dengue = generate_dengue(n_samples)
df_malaria = generate_malaria(n_samples)

df = pd.concat([df_normal, df_dengue, df_malaria]).sample(frac=1).reset_index(drop=True)

# تحويل نتائج المعامل لنصوص ليتوافق مع الموديل القديم لو حبيت
lab_map = {0: 'Negative', 1: 'Positive'}
df['ns1_result'] = df['ns1_result'].map(lab_map)
df['igm_result'] = df['igm_result'].map(lab_map)
df['pcr_result'] = df['pcr_result'].map(lab_map)

# حفظ كملف اكسل
try:
    df.to_excel(FILE_PATH, index=False)
    print(f"✅ Created synthetic dataset: {FILE_PATH}")
except Exception as e:
    print(f"❌ Error saving Excel (maybe openpyxl is missing?): {e}")
    df.to_csv(FILE_PATH.replace(".xlsx", ".csv"), index=False)
    print(f"✅ Saved as CSV instead at {FILE_PATH.replace('.xlsx', '.csv')}")
