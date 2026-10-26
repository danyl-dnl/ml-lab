r"""
Question 12: SVM Kernels Comparison (Linear, Polynomial, RBF) on Fashion MNIST
Description:
    Implement and compare the performance of Support Vector Machine (SVM)
    classifiers with Linear, Polynomial, and Radial Basis Function (RBF) kernels
    on the Fashion MNIST dataset. Analyze the advantages, disadvantages,
    and computational profiles of each kernel type.

Tasks:
    1. Load and preprocess Fashion MNIST (normalize pixels to [0, 1]).
    2. Implement SVM with Linear, Polynomial (degree=3), and RBF kernels.
    3. Evaluate classification performance: Accuracy, Precision, Recall, F1-Score,
       training time, inference latency, and support vector counts.
    4. Generate comparative plots (metric bar charts, latency profiles, confusion matrices).
    5. Discuss the mathematical theory, advantages, and limitations of each kernel.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets as dsets

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_fashion_mnist(data_dir: str = "../data", n_train: int = 4000, n_test: int = 1000):
    """Load and normalize Fashion-MNIST data for kernel SVM comparison."""
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = dsets.FashionMNIST(root=data_dir, train=True, download=True)
    test_dataset = dsets.FashionMNIST(root=data_dir, train=False, download=True)

    X_train_raw = train_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train_raw = train_dataset.targets.numpy()

    X_test_raw = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test_raw = test_dataset.targets.numpy()

    # Stratified balanced sampling across all 10 classes
    np.random.seed(42)
    train_idx = []
    samples_per_class_train = n_train // 10
    for c in range(10):
        idx = np.where(y_train_raw == c)[0]
        chosen = np.random.choice(idx, samples_per_class_train, replace=False)
        train_idx.extend(chosen)

    test_idx = []
    samples_per_class_test = n_test // 10
    for c in range(10):
        idx = np.where(y_test_raw == c)[0]
        chosen = np.random.choice(idx, samples_per_class_test, replace=False)
        test_idx.extend(chosen)

    return X_train_raw[train_idx], y_train_raw[train_idx], X_test_raw[test_idx], y_test_raw[test_idx]


def evaluate_svm_kernel(kernel_name: str, clf: SVC, X_train, y_train, X_test, y_test):
    """Fit SVM, measure runtime, count support vectors, and compute metrics."""
    print(f"\n--- Training SVM with {kernel_name} Kernel ---")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    infer_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    n_sv = clf.n_support_.sum()

    print(f"[{kernel_name}] Accuracy: {acc*100:.2f}% | F1 (Macro): {f1_macro:.4f} | SVs: {n_sv} | Train: {train_time:.2f}s | Infer: {infer_time:.2f}s")

    metrics = {
        "Kernel": kernel_name,
        "Accuracy": acc,
        "Precision (W)": prec_weighted,
        "Recall (W)": rec_weighted,
        "F1-Score (Macro)": f1_macro,
        "F1-Score (Weighted)": f1_weighted,
        "Support Vectors": n_sv,
        "SV Ratio (%)": (n_sv / len(X_train)) * 100.0,
        "Train Time (s)": train_time,
        "Inference Time (s)": infer_time,
    }
    return metrics, y_pred, confusion_matrix(y_test, y_pred)


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 12: SVM KERNEL COMPARISON (LINEAR, POLYNOMIAL, RBF)")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    X_train, y_train, X_test, y_test = load_fashion_mnist(data_dir=data_dir, n_train=4000, n_test=1000)
    print(f"Dataset split: {len(X_train)} train, {len(X_test)} test, {X_train.shape[1]} features.")

    # Kernels to compare
    kernel_configs = [
        ("Linear Kernel", SVC(kernel="linear", C=1.0, random_state=42)),
        ("Polynomial Kernel (d=3)", SVC(kernel="poly", degree=3, C=1.0, gamma="scale", random_state=42)),
        ("RBF Kernel", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ]

    results = []
    cm_dict = {}
    preds_dict = {}

    for name, clf in kernel_configs:
        res, y_pred, cm = evaluate_svm_kernel(name, clf, X_train, y_train, X_test, y_test)
        results.append(res)
        cm_dict[name] = cm
        preds_dict[name] = y_pred

    df_summary = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print(" SVM KERNEL COMPARISON SUMMARY ON FASHION MNIST")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    # Detailed report for the best kernel
    best_kernel_name = df_summary.loc[df_summary["Accuracy"].idxmax()]["Kernel"]
    print(f"\nOptimal Kernel: {best_kernel_name}")
    print(classification_report(y_test, preds_dict[best_kernel_name], target_names=FASHION_CLASSES))

    # Visualizations
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

    # 1. Accuracy & F1 Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(df_summary))
    width = 0.35
    b1 = ax1.bar(x - width/2, df_summary["Accuracy"] * 100, width, label='Accuracy (%)', color='#2b5c8f', edgecolor='black', linewidth=0.5)
    b2 = ax1.bar(x + width/2, df_summary["F1-Score (Macro)"] * 100, width, label='Macro F1 (%)', color='#e06666', edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([k.split(' ')[0] for k in df_summary["Kernel"]], fontsize=10)
    ax1.set_ylabel("Score (%)", fontsize=10)
    ax1.set_title("(A) Classification Performance", fontsize=11, weight='bold')
    ax1.set_ylim(70, 90)
    ax1.legend(loc='lower right', frameon=True)
    for bar in b1 + b2:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 2. Computational Timings (Train & Inference)
    ax2 = fig.add_subplot(gs[0, 1])
    b3 = ax2.bar(x - width/2, df_summary["Train Time (s)"], width, label='Train Time (s)', color='#34495e', edgecolor='black', linewidth=0.5)
    b4 = ax2.bar(x + width/2, df_summary["Inference Time (s)"], width, label='Inference Time (s)', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.split(' ')[0] for k in df_summary["Kernel"]], fontsize=10)
    ax2.set_ylabel("Seconds", fontsize=10)
    ax2.set_title("(B) Computational Efficiency", fontsize=11, weight='bold')
    ax2.legend(loc='upper right', frameon=True)
    for bar in b3 + b4:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 3. Support Vector Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    b5 = ax3.bar(x, df_summary["SV Ratio (%)"], width=0.5, color='#16a085', edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([k.split(' ')[0] for k in df_summary["Kernel"]], fontsize=10)
    ax3.set_ylabel("Support Vector Ratio (%)", fontsize=10)
    ax3.set_title("(C) Sparsity & Support Vector Retention", fontsize=11, weight='bold')
    ax3.set_ylim(0, 100)
    for bar in b5:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 4, 5, 6: Confusion Matrices for the 3 kernels
    cm_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    cm_palettes = ['Blues', 'Purples', 'Greens']

    for idx, (k_name, _) in enumerate(kernel_configs):
        ax_cm = cm_axes[idx]
        sns.heatmap(cm_dict[k_name], annot=False, cmap=cm_palettes[idx], ax=ax_cm, cbar=False)
        acc_val = df_summary.loc[df_summary["Kernel"] == k_name]["Accuracy"].values[0]
        ax_cm.set_title(f"CM: {k_name}\nAcc: {acc_val*100:.2f}%", fontsize=10, weight='bold')
        ax_cm.set_xlabel("Predicted Class", fontsize=9)
        if idx == 0:
            ax_cm.set_ylabel("True Class", fontsize=9)
        ax_cm.set_xticks(range(10))
        ax_cm.set_yticks(range(10))
        ax_cm.set_xticklabels(range(10), fontsize=8)
        ax_cm.set_yticklabels(range(10), fontsize=8)

    fig.suptitle("Empirical Comparison of SVM Kernels (Linear, Polynomial, RBF) on Fashion MNIST",
                 fontsize=14, weight='bold', y=0.98)

    plot_path_local = os.path.join(os.path.dirname(__file__), "q12_svm_kernel_comparison.png")
    plot_path_global = os.path.join(output_dir, "q12_svm_kernel_comparison.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_summary


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
