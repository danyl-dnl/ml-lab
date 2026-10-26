"""
Question 10: Logistic Regression vs. Decision Tree on Adult Income Dataset
Description:
    Implement and compare Logistic Regression and Decision Trees on the Adult
    Income dataset for predicting income levels (>50K vs <=50K). Evaluate both
    models based on performance metrics and interpretability.

Tasks:
    1. Load and preprocess the Adult Income dataset (handle missing values,
       one-hot encode categoricals, standardize numericals).
    2. Implement Logistic Regression and Decision Tree classifiers.
    3. Compare models across Accuracy, Precision, Recall, F1-score, and ROC-AUC.
    4. Generate comparative ROC curves, confusion matrices, and feature importance/odds plots.
    5. Discuss interpretability and architectural suitability for census/tabular demographic data.
"""

import os
import urllib.request
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_adult_dataset(data_dir: str = "../data") -> pd.DataFrame:
    """Load Adult Income dataset from local storage or UCI machine learning repository."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "adult.data")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]

    if not os.path.exists(local_path):
        print(f"Downloading Adult dataset from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print("Download complete.")

    df = pd.read_csv(local_path, names=columns, sep=r",\s*", engine="python", na_values="?")
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess_adult_data(df: pd.DataFrame):
    """Clean data, encode features, and perform stratified train-test splitting."""
    # Target encoding: >50K is positive class (1), <=50K is negative class (0)
    y = (df["income"].str.contains(">50K")).astype(int).values

    # Drop non-predictive weight identifier fnlwgt and target column
    X = df.drop(columns=["income", "fnlwgt"])

    num_cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    cat_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

    # Preprocessing pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Fitting preprocessor (StandardScaler + OneHotEncoder)...")
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # Extract feature names
    cat_features = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_features)

    print(f"Transformed features: {X_train.shape[1]} dimensional representation.")
    return X_train, X_test, y_train, y_test, all_feature_names


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test):
    """Train model, measure latency, and compute metrics."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    infer_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Train Time (s)": train_time,
        "Inference Time (s)": infer_time,
    }
    return metrics, y_pred, y_prob


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 10: LOGISTIC REGRESSION VS DECISION TREES (ADULT DATASET)")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    df = load_adult_dataset(data_dir=data_dir)
    X_train, X_test, y_train, y_test, feature_names = preprocess_adult_data(df)

    # 1. Logistic Regression Model
    print("\n[1] Training Logistic Regression...")
    clf_lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    res_lr, y_pred_lr, y_prob_lr = evaluate_model("Logistic Regression", clf_lr, X_train, y_train, X_test, y_test)

    # 2. Decision Tree Model (tuned depth to balance expressiveness and prevent overfitting)
    print("[2] Training Decision Tree (max_depth=8)...")
    clf_dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=25, random_state=42)
    res_dt, y_pred_dt, y_prob_dt = evaluate_model("Decision Tree (depth=8)", clf_dt, X_train, y_train, X_test, y_test)

    # Tabulate comparison
    df_compare = pd.DataFrame([res_lr, res_dt])
    print("\n" + "=" * 80)
    print(" PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print(df_compare.to_string(index=False))
    print("=" * 80)

    print("\n--- Logistic Regression Classification Report ---")
    print(classification_report(y_test, y_pred_lr, target_names=["<=50K", ">50K"]))

    print("\n--- Decision Tree Classification Report ---")
    print(classification_report(y_test, y_pred_dt, target_names=["<=50K", ">50K"]))

    # Interpretability Analysis
    # Logistic Regression: Top 8 positive and negative coefficients
    coefs = clf_lr.coef_[0]
    top_pos_idx = np.argsort(coefs)[-8:]
    top_neg_idx = np.argsort(coefs)[:8]
    top_indices = np.concatenate([top_neg_idx, top_pos_idx])
    top_features = [feature_names[i] for i in top_indices]
    top_weights = coefs[top_indices]

    # Decision Tree: Top 10 feature importances
    dt_importances = clf_dt.feature_importances_
    dt_top_idx = np.argsort(dt_importances)[-10:]
    dt_top_features = [feature_names[i] for i in dt_top_idx]
    dt_top_vals = dt_importances[dt_top_idx]

    # Visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # (A) Metric Comparison Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    vals_lr = [res_lr[m] for m in metrics]
    vals_dt = [res_dt[m] for m in metrics]

    b1 = ax1.bar(x - width/2, vals_lr, width, label='Logistic Regression', color='#2b5c8f', alpha=0.9, edgecolor='black', linewidth=0.5)
    b2 = ax1.bar(x + width/2, vals_dt, width, label='Decision Tree', color='#e06666', alpha=0.9, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("(A) Metric Comparison", fontsize=12, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', frameon=True)
    for bar in b1 + b2:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # (B) ROC Curves
    ax2 = fig.add_subplot(gs[0, 1])
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

    ax2.plot(fpr_lr, tpr_lr, color='#2b5c8f', lw=2.2, label=f"Logistic Regression (AUC = {res_lr['ROC-AUC']:.3f})")
    ax2.plot(fpr_dt, tpr_dt, color='#e06666', lw=2.2, linestyle='--', label=f"Decision Tree (AUC = {res_dt['ROC-AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], color='#888888', linestyle=':', label='Random Chance (AUC = 0.500)')
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("(B) Receiver Operating Characteristic (ROC)", fontsize=12, weight='bold')
    ax2.legend(loc='lower right', frameon=True)

    # (C) Interpretability: Logistic Regression Coefficients (Log-Odds Impact)
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['#e74c3c' if w < 0 else '#27ae60' for w in top_weights]
    ax3.barh(range(len(top_weights)), top_weights, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_yticks(range(len(top_weights)))
    ax3.set_yticklabels(top_features, fontsize=9)
    ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.set_xlabel("Weight Coefficient (Log-Odds Impact)", fontsize=10)
    ax3.set_title("(C) Logistic Regression: Top Predictors (Interpretability)", fontsize=12, weight='bold')

    # (D) Interpretability: Decision Tree Feature Importance
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(range(len(dt_top_vals)), dt_top_vals, color='#f39c12', edgecolor='black', linewidth=0.5)
    ax4.set_yticks(range(len(dt_top_vals)))
    ax4.set_yticklabels(dt_top_features, fontsize=9)
    ax4.set_xlabel("Gini Importance Score", fontsize=10)
    ax4.set_title("(D) Decision Tree: Feature Importance (Gini Impurity Reduction)", fontsize=12, weight='bold')

    fig.suptitle("Comparative Evaluation: Logistic Regression vs. Decision Trees on Adult Income Dataset",
                 fontsize=14, weight='bold', y=0.98)

    plot_path_local = os.path.join(os.path.dirname(__file__), "q10_logreg_vs_decision_tree.png")
    plot_path_global = os.path.join(output_dir, "q10_logreg_vs_decision_tree.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_compare


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
