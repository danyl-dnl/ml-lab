r"""
Question 11: Linear Support Vector Machine (SVM) on Iris Dataset
Description:
    Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset.
    Visualize the decision boundary, the margin lines, and the support vectors.
    Discuss how the margin is determined mathematically and how it influences
    classification performance and generalization.

Tasks:
    1. Load and preprocess the Iris dataset (2D feature space for boundary visualization).
    2. Implement Linear SVM for binary classification (Setosa vs Non-Setosa, and
       Versicolor vs Virginica soft-margin).
    3. Visualize decision boundary ($w^T x + b = 0$), positive/negative margins
       ($w^T x + b = \pm 1$), and highlight exact support vectors.
    4. Analyze metrics: Accuracy, Precision, Recall, Margin Width ($2/||w||$).
    5. Discuss the concept of the margin, support vectors, and generalization theory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_svm_decision_boundary(clf, X, y, ax, title, feature_names, class_labels):
    """
    Plot 2D decision boundary, margin boundaries (w^T x + b = +-1),
    and highlight support vectors.
    """
    # Scatter training data points
    colors = ['#3498db', '#e74c3c']
    markers = ['o', 's']
    for idx, c in enumerate(np.unique(y)):
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            c=colors[idx],
            marker=markers[idx],
            s=60,
            label=class_labels[idx],
            edgecolor='black',
            linewidth=0.6,
            alpha=0.9
        )

    # Highlight support vectors
    sv = clf.support_vectors_
    ax.scatter(
        sv[:, 0],
        sv[:, 1],
        s=160,
        facecolors='none',
        edgecolors='#2c3e50',
        linewidth=1.8,
        label=f'Support Vectors ({len(sv)})'
    )

    # Generate grid
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary (Z = 0) and margins (Z = -1, Z = 1)
    ax.contour(
        xx, yy, Z,
        colors=['#27ae60', '#2c3e50', '#27ae60'],
        levels=[-1, 0, 1],
        linestyles=['--', '-', '--'],
        linewidths=[1.8, 2.5, 1.8]
    )

    # Fill margin region
    ax.contourf(xx, yy, Z, levels=[-1, 1], colors=['#27ae60'], alpha=0.12)

    # Calculate weights and margin width
    w = clf.coef_[0]
    b = clf.intercept_[0]
    w_norm = np.linalg.norm(w)
    margin_width = 2.0 / w_norm

    ax.set_title(f"{title}\nMargin Width: {margin_width:.3f} | ||w||: {w_norm:.3f}", fontsize=11, weight='bold')
    ax.set_xlabel(feature_names[0], fontsize=10)
    ax.set_ylabel(feature_names[1], fontsize=10)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    return margin_width, w, b


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 11: LINEAR SUPPORT VECTOR MACHINE (SVM) ON IRIS DATASET")
    print("=" * 70)

    iris = load_iris()
    # Features: Petal Length and Petal Width (most discriminative 2D plane)
    feature_indices = [2, 3]
    feature_names = [iris.feature_names[i] for i in feature_indices]
    X_all = iris.data[:, feature_indices]
    y_all = iris.target

    # Experiment A: Setosa vs Non-Setosa (Linearly Separable, Hard-Margin Demonstration)
    print("\n--- Experiment A: Setosa (Class 0) vs Non-Setosa (Classes 1 & 2) ---")
    y_bin_a = (y_all == 0).astype(int)  # 1 = Setosa, 0 = Non-Setosa

    scaler_a = StandardScaler()
    X_scaled_a = scaler_a.fit_transform(X_all)

    clf_a = SVC(kernel='linear', C=10.0, random_state=42)
    clf_a.fit(X_scaled_a, y_bin_a)
    y_pred_a = clf_a.predict(X_scaled_a)
    acc_a = accuracy_score(y_bin_a, y_pred_a)

    w_a = clf_a.coef_[0]
    b_a = clf_a.intercept_[0]
    margin_a = 2.0 / np.linalg.norm(w_a)
    print(f"Accuracy: {acc_a*100:.2f}% (Perfect Linear Separation)")
    print(f"Weight Vector w: {w_a}, Intercept b: {b_a:.3f}")
    print(f"Number of Support Vectors: {len(clf_a.support_vectors_)}")
    print(f"Geometric Margin Width (2 / ||w||): {margin_a:.4f}")

    # Experiment B: Versicolor vs Virginica (Soft-Margin Demonstration with Overlap)
    print("\n--- Experiment B: Versicolor (Class 1) vs Virginica (Class 2) ---")
    mask_b = (y_all != 0)
    X_b = X_all[mask_b]
    y_bin_b = (y_all[mask_b] == 2).astype(int)  # 1 = Virginica, 0 = Versicolor

    scaler_b = StandardScaler()
    X_scaled_b = scaler_b.fit_transform(X_b)

    clf_b = SVC(kernel='linear', C=1.0, random_state=42)
    clf_b.fit(X_scaled_b, y_bin_b)
    y_pred_b = clf_b.predict(X_scaled_b)
    acc_b = accuracy_score(y_bin_b, y_pred_b)

    w_b = clf_b.coef_[0]
    b_b = clf_b.intercept_[0]
    margin_b = 2.0 / np.linalg.norm(w_b)
    print(f"Accuracy: {acc_b*100:.2f}%")
    print(f"Weight Vector w: {w_b}, Intercept b: {b_b:.3f}")
    print(f"Number of Support Vectors: {len(clf_b.support_vectors_)}")
    print(f"Geometric Margin Width (2 / ||w||): {margin_b:.4f}")

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Linear Support Vector Machine: Decision Boundaries, Margins & Support Vectors\n(Iris Dataset)",
                 fontsize=14, weight='bold', y=0.98)

    # Plot A
    plot_svm_decision_boundary(
        clf_a, X_scaled_a, y_bin_a, axes[0],
        title="(A) Setosa vs. Non-Setosa (Linearly Separable)",
        feature_names=[f"{f} (Standardized)" for f in feature_names],
        class_labels=["Non-Setosa", "Setosa"]
    )

    # Plot B
    plot_svm_decision_boundary(
        clf_b, X_scaled_b, y_bin_b, axes[1],
        title="(B) Versicolor vs. Virginica (Soft Margin with Overlap)",
        feature_names=[f"{f} (Standardized)" for f in feature_names],
        class_labels=["Versicolor", "Virginica"]
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    plot_path_local = os.path.join(os.path.dirname(__file__), "q11_linear_svm_iris_margin.png")
    plot_path_global = os.path.join(output_dir, "q11_linear_svm_iris_margin.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")

    return {
        "ExpA_Margin": margin_a,
        "ExpA_Acc": acc_a,
        "ExpB_Margin": margin_b,
        "ExpB_Acc": acc_b
    }


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
