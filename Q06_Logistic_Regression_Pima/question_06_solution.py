"""
Question 6: Logistic Regression on Pima Indians Diabetes Dataset
Description:
    Implement a logistic regression model to predict the likelihood of disease
    using the Pima Indians Diabetes dataset. Compare performance with and
    without feature scaling.

Tasks:
    1. Load and preprocess the Pima Indians Diabetes dataset (handle zero/missing values).
    2. Implement logistic regression for binary classification.
    3. Evaluate model performance with and without feature scaling.
    4. Analyze metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
    5. Save visualizations and output comparative findings.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_pima_dataset(data_dir: str = "../data") -> pd.DataFrame:
    """Load Pima Indians Diabetes dataset from local cache or remote source."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "pima-indians-diabetes.csv")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]

    if not os.path.exists(local_path):
        print(f"Downloading Pima Indians Diabetes dataset from {url}...")
        urllib.request.urlretrieve(url, local_path)
        print("Download complete.")

    df = pd.read_csv(local_path, names=columns)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Handle biologically implausible zeros in physiological attributes.
    Zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI
    represent missing measurements and are imputed via median strategy.
    """
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = df.copy()
    for col in zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"].values

    # Train-test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Impute missing values using training medians (prevent data leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    feature_names = X.columns.tolist()
    return X_train_imp, X_test_imp, y_train, y_test, feature_names


def evaluate_model(name: str, y_true, y_pred, y_prob):
    """Compute and return standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
    }


def run_experiment(output_dir: str = "../outputs"):
    """Run full logistic regression comparison with and without scaling."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./", exist_ok=True)

    print("=" * 70)
    print(" QUESTION 6: LOGISTIC REGRESSION - PIMA INDIANS DIABETES DATASET")
    print("=" * 70)

    # 1. Load Data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    df = load_pima_dataset(data_dir=data_dir)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {dict(df['Outcome'].value_counts())}")

    # 2. Preprocess Data
    X_train_raw, X_test_raw, y_train, y_test, feature_names = preprocess_data(df)

    # 3. Apply Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 4. Train Model WITHOUT Feature Scaling
    print("\n--- Training Model 1: Without Feature Scaling ---")
    clf_no_scale = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    clf_no_scale.fit(X_train_raw, y_train)

    y_pred_no_scale = clf_no_scale.predict(X_test_raw)
    y_prob_no_scale = clf_no_scale.predict_proba(X_test_raw)[:, 1]
    res_no_scale = evaluate_model("Without Scaling", y_test, y_pred_no_scale, y_prob_no_scale)
    cm_no_scale = confusion_matrix(y_test, y_pred_no_scale)

    # 5. Train Model WITH Feature Scaling
    print("--- Training Model 2: With Feature Scaling (StandardScaler) ---")
    clf_scaled = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    clf_scaled.fit(X_train_scaled, y_train)

    y_pred_scaled = clf_scaled.predict(X_test_scaled)
    y_prob_scaled = clf_scaled.predict_proba(X_test_scaled)[:, 1]
    res_scaled = evaluate_model("With Scaling", y_test, y_pred_scaled, y_prob_scaled)
    cm_scaled = confusion_matrix(y_test, y_pred_scaled)

    # 6. Tabular Metrics Comparison
    df_metrics = pd.DataFrame([res_no_scale, res_scaled])
    print("\n" + "=" * 55)
    print(" PERFORMANCE COMPARISON SUMMARY")
    print("=" * 55)
    print(df_metrics.to_string(index=False))
    print("=" * 55)

    print("\nClassification Report (With Feature Scaling):")
    print(classification_report(y_test, y_pred_scaled, target_names=["Non-Diabetic (0)", "Diabetic (1)"]))

    # 7. Convergence analysis
    print(f"Iterations to converge (No Scaling): {clf_no_scale.n_iter_[0]}")
    print(f"Iterations to converge (With Scaling): {clf_scaled.n_iter_[0]}")

    # 8. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Logistic Regression Performance: With vs Without Feature Scaling\n(Pima Indians Diabetes)", fontsize=15, weight='bold', y=0.98)

    # (A) Metric Comparison Bar Chart
    ax1 = axes[0, 0]
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    vals_no = [res_no_scale[m] for m in metrics_to_plot]
    vals_sc = [res_scaled[m] for m in metrics_to_plot]

    b1 = ax1.bar(x - width/2, vals_no, width, label='Without Scaling', color='#4A90E2', alpha=0.85, edgecolor='black', linewidth=0.5)
    b2 = ax1.bar(x + width/2, vals_sc, width, label='With Scaling (StandardScaler)', color='#50E3C2', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('(A) Metric Comparison', fontsize=12, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_to_plot, fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right', frameon=True)
    for bar in b1 + b2:
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # (B) ROC Curves
    ax2 = axes[0, 1]
    fpr_no, tpr_no, _ = roc_curve(y_test, y_prob_no_scale)
    fpr_sc, tpr_sc, _ = roc_curve(y_test, y_prob_scaled)

    ax2.plot(fpr_no, tpr_no, color='#4A90E2', lw=2, label=f'Without Scaling (AUC = {res_no_scale["ROC-AUC"]:.3f})')
    ax2.plot(fpr_sc, tpr_sc, color='#E94E77', lw=2, linestyle='--', label=f'With Scaling (AUC = {res_scaled["ROC-AUC"]:.3f})')
    ax2.plot([0, 1], [0, 1], color='#888888', linestyle=':', label='Chance Level (AUC = 0.500)')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax2.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax2.set_title('(B) Receiver Operating Characteristic (ROC)', fontsize=12, weight='bold')
    ax2.legend(loc='lower right', frameon=True)

    # (C) Confusion Matrix: Without Scaling
    ax3 = axes[1, 0]
    sns.heatmap(cm_no_scale, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False,
                xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax3.set_title(f'(C) Confusion Matrix: Without Scaling\n(Acc: {res_no_scale["Accuracy"]:.3f}, F1: {res_no_scale["F1-Score"]:.3f})', fontsize=11, weight='bold')
    ax3.set_xlabel('Predicted Label', fontsize=10)
    ax3.set_ylabel('True Label', fontsize=10)

    # (D) Confusion Matrix: With Scaling
    ax4 = axes[1, 1]
    sns.heatmap(cm_scaled, annot=True, fmt='d', cmap='Greens', ax=ax4, cbar=False,
                xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax4.set_title(f'(D) Confusion Matrix: With Scaling\n(Acc: {res_scaled["Accuracy"]:.3f}, F1: {res_scaled["F1-Score"]:.3f})', fontsize=11, weight='bold')
    ax4.set_xlabel('Predicted Label', fontsize=10)
    ax4.set_ylabel('True Label', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plots
    plot_path_local = os.path.join(os.path.dirname(__file__), "q06_feature_scaling_comparison.png")
    plot_path_global = os.path.join(output_dir, "q06_feature_scaling_comparison.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_metrics


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
