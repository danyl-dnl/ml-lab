"""
Question 6: Logistic Regression on Pima Indians Diabetes Dataset
Tasks:
  1. Load and preprocess the Pima Indians Diabetes dataset.
  2. Implement logistic regression for binary classification.
  3. Evaluate model performance with and without feature scaling.
  4. Analyze metrics such as accuracy, precision, recall, and F1-score.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4),
    }


def main():
    # 1. Load and preprocess dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"
    ]
    df = pd.read_csv(url, names=columns)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Logistic Regression WITHOUT feature scaling
    model_unscaled = LogisticRegression(max_iter=1000, random_state=42)
    model_unscaled.fit(X_train, y_train)
    y_pred_unscaled = model_unscaled.predict(X_test)
    metrics_unscaled = evaluate(y_test, y_pred_unscaled)

    # 3. Logistic Regression WITH feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_scaled = LogisticRegression(max_iter=1000, random_state=42)
    model_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    metrics_scaled = evaluate(y_test, y_pred_scaled)

    # 4. Compare performance metrics
    results = pd.DataFrame([metrics_unscaled, metrics_scaled], index=["Without Scaling", "With Scaling"])
    print("=" * 50)
    print("QUESTION 6: LOGISTIC REGRESSION (PIMA DIABETES)")
    print("=" * 50)
    print(results)
    print("=" * 50)
    print(f"Iterations to converge without scaling: {model_unscaled.n_iter_[0]}")
    print(f"Iterations to converge with scaling:    {model_scaled.n_iter_[0]}")


if __name__ == "__main__":
    main()
