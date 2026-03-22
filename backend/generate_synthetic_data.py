import pandas as pd
import numpy as np
import os

# تحديد المسار
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "dengue_malaria_dataset.xlsx")

np.random.seed(42)
n_samples = 300 # زيادة العينات قليلاً

def generate_normal(n):
    return pd.DataFrame({
        'ns1_result': np.zeros(n),
        'igm_result': np.zeros(n),
        'pcr_result': np.zeros(n),
        'age': np.random.randint(5, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'temperature': np.random.normal(37.0, 0.4, n),
        'fever_days': np.random.randint(0, 3, n),
        'final_diagnosis': 'Non-dengue'
    })

def generate_dengue(n):
    return pd.DataFrame({
        # تقليل احتمالية الإيجابية لجعل النموذج لا يعتمد عليها كلياً
        'ns1_result': np.random.choice([0, 1], n, p=[0.4, 0.6]), 
        'igm_result': np.random.choice([0, 1], n, p=[0.5, 0.5]),
        'pcr_result': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'age': np.random.randint(5, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'temperature': np.random.normal(38.8, 0.9, n), # تداخل أكبر
        'fever_days': np.random.randint(2, 8, n),
        'final_diagnosis': 'Confirmed dengue'
    })

def generate_malaria(n):
    return pd.DataFrame({
        'ns1_result': np.zeros(n),
        'igm_result': np.zeros(n),
        'pcr_result': np.zeros(n),
        'age': np.random.randint(5, 80, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'temperature': np.random.normal(38.5, 1.2, n), # حرارة متغيرة جداً
        'fever_days': np.random.randint(3, 12, n),
        'final_diagnosis': 'Malaria'
    })

df_normal = generate_normal(n_samples)
df_dengue = generate_dengue(n_samples)
df_malaria = generate_malaria(n_samples)

df = pd.concat([df_normal, df_dengue, df_malaria]).sample(frac=1).reset_index(drop=True)

# تحويل النتائج لنصوص (متوافق مع mapping في train_model)
lab_map = {0: 'Negative', 1: 'Positive'}
df['ns1_result'] = df['ns1_result'].map(lab_map)
df['igm_result'] = df['igm_result'].map(lab_map)
df['pcr_result'] = df['pcr_result'].map(lab_map)

# حفظ كملف اكسل وتجنب خطأ التشفير في الطباعة
try:
    df.to_excel(FILE_PATH, index=False)
    print("Dataset created successfully (Excel).")
except Exception as e:
    df.to_csv(FILE_PATH.replace(".xlsx", ".csv"), index=False)
    print(f"Saved as CSV due to: {e}")
