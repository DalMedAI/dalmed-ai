import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "advanced_full_medical_dataset (2).csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "dengue_model.pkl")

LABELS = ["Dengue", "Malaria", "Typhoid"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}

BASE_FEATURES = [
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
    "fever_days",
]

DENGUE_FEATURES = ["eye_pain", "rash", "joint_pain", "bleeding"]
MALARIA_FEATURES = ["chills", "sweating", "anemia", "jaundice"]
TYPHOID_FEATURES = ["abdominal_pain", "loss_of_appetite", "diarrhea_constipation"]

ENGINEERED_FEATURES = [
    "symptom_count",
    "dengue_signature_score",
    "malaria_signature_score",
    "typhoid_signature_score",
    "competing_signature_count",
    "overlap_penalty",
]

TRAINING_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def add_medical_features(df):
    df = df.copy()
    symptom_columns = [feature for feature in BASE_FEATURES if feature not in ["age", "gender", "fever_days"]]

    df["symptom_count"] = df[symptom_columns].sum(axis=1)
    df["dengue_signature_score"] = df[DENGUE_FEATURES].sum(axis=1)
    df["malaria_signature_score"] = df[MALARIA_FEATURES].sum(axis=1)
    df["typhoid_signature_score"] = df[TYPHOID_FEATURES].sum(axis=1)

    signature_scores = df[
        ["dengue_signature_score", "malaria_signature_score", "typhoid_signature_score"]
    ]
    df["competing_signature_count"] = (signature_scores >= 2).sum(axis=1)
    df["overlap_penalty"] = np.maximum(df["competing_signature_count"] - 1, 0)
    return df


def validate_dataset(df):
    missing = [column for column in BASE_FEATURES + ["diagnosis"] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    df = df.dropna(subset=BASE_FEATURES + ["diagnosis"]).drop_duplicates()
    df = df[df["diagnosis"].isin(LABELS)].copy()

    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=BASE_FEATURES)
    df["age"] = df["age"].clip(0, 120)
    df["gender"] = df["gender"].clip(0, 1)
    df["fever_days"] = df["fever_days"].clip(0, 21)

    binary_columns = [column for column in BASE_FEATURES if column not in ["age", "gender", "fever_days"]]
    for column in binary_columns:
        df[column] = df[column].clip(0, 1)

    if df.empty:
        raise ValueError("No usable rows left after cleaning the dataset.")

    df["diagnosis"] = df["diagnosis"].map(LABEL_TO_ID)
    return add_medical_features(df)


def build_model():
    return build_pipeline(
        CalibratedClassifierCV(
            estimator=build_random_forest(),
            method="sigmoid",
            cv=5,
        )
    )


def build_preprocessor():
    scaler = ColumnTransformer(
        transformers=[
            ("scaled_numeric", StandardScaler(), TRAINING_FEATURES),
        ],
        remainder="drop",
    )
    return scaler


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=700,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_pipeline(classifier):
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_comparison_models():
    return {
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            )
        ),
        "SVM": build_pipeline(
            SVC(
                class_weight="balanced",
                random_state=42,
            )
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} Evaluation")
    print("-" * (len(name) + 11))
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    return {
        "Algorithm Name": name,
        "Accuracy Result": f"{accuracy * 100:.2f}%",
        "accuracy": accuracy,
        "predictions": y_pred,
    }


def print_comparison_table(results):
    comparison_df = pd.DataFrame(
        [
            {
                "Algorithm Name": result["Algorithm Name"],
                "Accuracy Result": result["Accuracy Result"],
            }
            for result in results
        ]
    )

    print("\nMachine Learning Algorithm Comparison")
    print("-------------------------------------")
    print(comparison_df.to_string(index=False))


def save_plot(filename):
    path = os.path.join(BASE_DIR, filename)
    try:
        plt.savefig(path)
    except PermissionError:
        fallback = os.path.join(BASE_DIR, filename.replace(".png", "_latest.png"))
        plt.savefig(fallback)
        print(f"Could not overwrite {path}; saved plot to {fallback} instead.")


def calculate_sample_weights(X, y):
    disease_scores = np.column_stack(
        [
            X["dengue_signature_score"],
            X["malaria_signature_score"],
            X["typhoid_signature_score"],
        ]
    )
    own_signature = disease_scores[np.arange(len(y)), y.to_numpy()]
    strongest_other = np.partition(disease_scores, -2, axis=1)[:, -2]

    clear_case_bonus = np.where(own_signature >= strongest_other + 1, 1.25, 1.0)
    overlap_penalty = np.where(X["overlap_penalty"] > 0, 0.72, 1.0)
    overload_penalty = np.where(X["symptom_count"] >= 10, 0.55, 1.0)
    return clear_case_bonus * overlap_penalty * overload_penalty


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    raw_df = pd.read_csv(DATA_PATH)
    df = validate_dataset(raw_df)

    X = df[TRAINING_FEATURES]
    y = df["diagnosis"].astype(int)

    print(f"Training rows: {len(X)}")
    print(f"Features: {TRAINING_FEATURES}")
    print("Class distribution:")
    print(y.map(dict(enumerate(LABELS))).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    sample_weights = calculate_sample_weights(X_train, y_train)

    comparison_results = []
    for algorithm_name, comparison_model in build_comparison_models().items():
        print(f"\nTraining {algorithm_name} comparison model...")
        comparison_model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        comparison_results.append(evaluate_model(algorithm_name, comparison_model, X_test, y_test))

    print("Training calibrated RandomForest model...")
    model.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    print("\nEvaluating calibrated RandomForest production model...")
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    random_forest_accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {random_forest_accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
    print(f"Mean max probability: {np.max(probabilities, axis=1).mean() * 100:.2f}%")

    comparison_results.append(
        {
            "Algorithm Name": "Random Forest",
            "Accuracy Result": f"{random_forest_accuracy * 100:.2f}%",
            "accuracy": random_forest_accuracy,
            "predictions": y_pred,
        }
    )
    print_comparison_table(comparison_results)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Dengue, Malaria, Typhoid Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    save_plot("confusion_matrix.png")
    plt.close()

    classifier = model.named_steps["classifier"].calibrated_classifiers_[0].estimator
    importances = classifier.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(12, 10))
    sns.barplot(x=importances[indices], y=[TRAINING_FEATURES[i] for i in indices], palette="viridis")
    plt.title("Feature Importance - Calibrated RandomForest")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    save_plot("feature_importance.png")
    plt.close()

    artifact = {
        "model": model,
        "labels": LABELS,
        "base_features": BASE_FEATURES,
        "training_features": TRAINING_FEATURES,
        "disease_signature_features": {
            "dengue": DENGUE_FEATURES,
            "malaria": MALARIA_FEATURES,
            "typhoid": TYPHOID_FEATURES,
        },
        "version": "2026-05-10-realistic-dmt-v1",
    }
    joblib.dump(artifact, MODEL_SAVE_PATH)
    print(f"Model artifact saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
