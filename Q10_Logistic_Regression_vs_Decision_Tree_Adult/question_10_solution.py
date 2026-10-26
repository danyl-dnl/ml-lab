"""
Question 10: Logistic Regression vs Decision Trees on Adult Income Dataset
Tasks:
  1. Load and preprocess the Adult Income dataset.
  2. Implement both Logistic Regression and Decision Trees.
  3. Compare the models based on metrics such as accuracy, precision, recall, and F1-score.
  4. Discuss the interpretability of both models and their suitability for the dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4),
    }


def main():
    # 1. Load dataset
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv("../data/adult.data", names=cols, sep=r",\s*", engine="python", na_values="?")
    df = df.dropna()

    # Target: 1 if >50K else 0
    y = (df["income"] == ">50K").astype(int)

    # Features: one-hot encode categoricals and scale numericals
    X_raw = df[["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "marital_status", "sex"]]
    X = pd.get_dummies(X_raw, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Implement Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    res_lr = evaluate("Logistic Regression", y_test, y_pred_lr)

    # 3. Implement Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    res_dt = evaluate("Decision Tree (depth=6)", y_test, y_pred_dt)

    # 4. Compare models based on metrics
    results = pd.DataFrame([res_lr, res_dt]).set_index("Model")
    print("=" * 60)
    print("QUESTION 10: LOGISTIC REGRESSION VS DECISION TREE (ADULT INCOME)")
    print("=" * 60)
    print(results)
    print("=" * 60)

    # 5. Discuss interpretability and suitability
    print("\n--- Discussion: Interpretability and Suitability ---")
    print("Interpretability:")
    print("  - Logistic Regression: Provides linear odds ratios (exp(w_i)). Easy to see the global impact")
    print("    of each predictor (e.g. each year of education increases odds of earning >50K by a fixed factor).")
    print("  - Decision Tree: Highly intuitive if-then rules (e.g. IF marital_status == Married AND education > 12).")
    print("    Naturally captures non-linear splits and feature interactions without manual feature engineering.")
    print("Suitability for Census Dataset:")
    print("  - Decision Trees handle mixed demographic data and threshold cutoffs naturally.")
    print("  - Logistic Regression provides well-calibrated probabilities for risk assessment and policy analysis.")


if __name__ == "__main__":
    main()
