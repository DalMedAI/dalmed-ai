import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_multi_disease_dataset(num_samples=3000):
    """
    Generates a synthetic dataset for Normal, Dengue, and Malaria.
    Diagnosis: 0 = Normal/Other, 1 = Dengue, 2 = Malaria
    """
    np.random.seed(42)
    
    data = []
    
    for _ in range(num_samples):
        # 0: Normal/Other, 1: Dengue, 2: Malaria
        diagnosis = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
        
        # Demographic features
        age = np.random.randint(5, 80)
        gender = np.random.choice([0, 1]) # 0: Male, 1: Female
        
        if diagnosis == 1: # Dengue
            fever = np.random.choice([0, 1], p=[0.05, 0.95])
            headache = np.random.choice([0, 1], p=[0.1, 0.9])
            eye_pain = np.random.choice([0, 1], p=[0.2, 0.8])
            muscle_pain = np.random.choice([0, 1], p=[0.1, 0.9])
            nausea = np.random.choice([0, 1], p=[0.3, 0.7])
            rash = np.random.choice([0, 1], p=[0.4, 0.6])
            bleeding = np.random.choice([0, 1], p=[0.8, 0.2])
        elif diagnosis == 2: # Malaria
            fever = np.random.choice([0, 1], p=[0.02, 0.98])
            headache = np.random.choice([0, 1], p=[0.1, 0.9])
            eye_pain = np.random.choice([0, 1], p=[0.8, 0.2]) # Less common in Malaria
            muscle_pain = np.random.choice([0, 1], p=[0.3, 0.7])
            nausea = np.random.choice([0, 1], p=[0.2, 0.8])
            rash = np.random.choice([0, 1], p=[0.95, 0.05]) # Rare in Malaria
            bleeding = np.random.choice([0, 1], p=[0.98, 0.02])
        else: # Normal / Cold
            fever = np.random.choice([0, 1], p=[0.5, 0.5])
            headache = np.random.choice([0, 1], p=[0.4, 0.6])
            eye_pain = np.random.choice([0, 1], p=[0.9, 0.1])
            muscle_pain = np.random.choice([0, 1], p=[0.7, 0.3])
            nausea = np.random.choice([0, 1], p=[0.8, 0.2])
            rash = np.random.choice([0, 1], p=[0.99, 0.01])
            bleeding = np.random.choice([0, 1], p=[0.99, 0.01])
            
        data.append([age, gender, fever, headache, eye_pain, muscle_pain, nausea, rash, bleeding, diagnosis])
        
    df = pd.DataFrame(data, columns=[
        'age', 'gender', 'fever', 'headache', 'eye_pain', 'joint_muscle_pain', 
        'nausea_vomiting', 'rash', 'bleeding', 'diagnosis'
    ])
    
    return df

def plot_confusion_matrix(y_test, y_pred, classes):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Dengue vs Malaria vs Normal')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved Confusion Matrix plot as 'confusion_matrix.png'")

def plot_feature_importance(model, feature_names):
    """Generates and saves a feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances for Disease Diagnosis')
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Saved Feature Importance plot as 'feature_importance.png'")

if __name__ == "__main__":
    print("Generating Kaggle-style simulated Multi-Disease dataset...")
    df = generate_multi_disease_dataset(5000)
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    feature_columns = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier model...")
    # XGBoost handles multi-class naturally, objective='multi:softprob' is used behind the scenes
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully! Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    
    target_names = ['Normal/Other', 'Dengue', 'Malaria']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Generate Plots
    print("\nGenerating Visualizations...")
    plot_confusion_matrix(y_test, y_pred, target_names)
    plot_feature_importance(model, feature_columns)
    
    # Save the model
    joblib.dump(model, 'dengue_model.pkl')
    print("Model saved to 'dengue_model.pkl'")
