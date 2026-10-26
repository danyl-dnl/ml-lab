"""
Question 8: K-Nearest Neighbors (KNN) on Fashion MNIST Dataset
Description:
    Implement the K-Nearest Neighbors (KNN) algorithm for image classification
    using the Fashion MNIST dataset. Experiment with different values of K and
    analyze their impact on model performance and computational efficiency.

Tasks:
    1. Load and preprocess the Fashion MNIST dataset (flattening, min-max scaling).
    2. Implement KNN for multi-class classification (10 classes).
    3. Experiment with different values of K (K = 1, 3, 5, 7, 9, 11, 15).
    4. Evaluate and compare performance: Accuracy, Precision, Recall, F1-Score,
       Inference latency, and Confusion Matrix.
    5. Discuss the impact of K on model accuracy and computational efficiency.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets as dsets

from sklearn.neighbors import KNeighborsClassifier
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
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]


def load_fashion_mnist(data_dir: str = "../data", n_train: int = 6000, n_test: int = 1000):
    """
    Load Fashion-MNIST, normalize pixels to [0, 1], and select a stratified
    balanced sample for efficient multi-K neighbor evaluation.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = dsets.FashionMNIST(root=data_dir, train=True, download=True)
    test_dataset = dsets.FashionMNIST(root=data_dir, train=False, download=True)

    X_train_raw = train_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train_raw = train_dataset.targets.numpy()

    X_test_raw = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test_raw = test_dataset.targets.numpy()

    # Stratified subsampling for rigorous and swift evaluation
    np.random.seed(42)
    train_indices = []
    samples_per_class_train = n_train // 10
    for c in range(10):
        idx = np.where(y_train_raw == c)[0]
        chosen = np.random.choice(idx, samples_per_class_train, replace=False)
        train_indices.extend(chosen)

    test_indices = []
    samples_per_class_test = n_test // 10
    for c in range(10):
        idx = np.where(y_test_raw == c)[0]
        chosen = np.random.choice(idx, samples_per_class_test, replace=False)
        test_indices.extend(chosen)

    X_train = X_train_raw[train_indices]
    y_train = y_train_raw[train_indices]
    X_test = X_test_raw[test_indices]
    y_test = y_test_raw[test_indices]

    print(f"Loaded Fashion-MNIST subset: {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Features per image: {X_train.shape[1]} (28x28 normalized pixels).")
    return X_train, y_train, X_test, y_test


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 8: K-NEAREST NEIGHBORS (KNN) ON FASHION MNIST")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    X_train, y_train, X_test, y_test = load_fashion_mnist(data_dir=data_dir, n_train=6000, n_test=1000)

    # K values to experiment with
    k_values = [1, 3, 5, 7, 9, 11, 15]
    results = []
    best_k = None
    best_acc = -1.0
    best_pred = None

    print("\n--- Evaluating KNN Across Different Values of K ---")
    for k in k_values:
        # Distance metric: Euclidean distance (p=2)
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)

        # Fit (indexing) time
        t0 = time.time()
        knn.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Predict (inference) time
        t0 = time.time()
        y_pred = knn.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            "K": k,
            "Accuracy": acc,
            "Macro Precision": prec,
            "Macro Recall": rec,
            "Macro F1": f1,
            "Fit Time (s)": fit_time,
            "Inference Time (s)": infer_time,
            "Latency per sample (ms)": (infer_time / len(X_test)) * 1000.0,
        })

        print(f"K = {k:2d} | Accuracy: {acc*100:.2f}% | Macro F1: {f1:.4f} | Inference Time: {infer_time:.2f}s")

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_pred = y_pred

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print(" KNN EXPERIMENTAL RESULTS ACROSS K VALUES")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    print(f"\nOptimal K value identified: K = {best_k} (Accuracy: {best_acc*100:.2f}%)")
    print(f"\nClassification Report for Best K ({best_k}):")
    print(classification_report(y_test, best_pred, target_names=FASHION_CLASSES))

    # Confusion matrix for best K
    cm_best = confusion_matrix(y_test, best_pred)

    # Visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # (1) Accuracy and F1-score vs K
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_results["K"], df_results["Accuracy"] * 100, marker='o', linewidth=2.2, color='#2b5c8f', label='Accuracy (%)')
    ax1.plot(df_results["K"], df_results["Macro F1"] * 100, marker='s', linewidth=2.2, color='#e06666', linestyle='--', label='Macro F1 (%)')
    ax1.axvline(best_k, color='#27ae60', linestyle=':', label=f'Optimal K={best_k}')
    ax1.set_xlabel("Value of K (Number of Neighbors)", fontsize=11)
    ax1.set_ylabel("Score (%)", fontsize=11)
    ax1.set_title("(A) Classification Accuracy & F1 vs K", fontsize=12, weight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # (2) Computational Efficiency: Inference Time vs K
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(df_results["K"].astype(str), df_results["Inference Time (s)"], color='#34495e', alpha=0.85, width=0.55)
    ax2.set_xlabel("Value of K", fontsize=11)
    ax2.set_ylabel("Test Inference Time (seconds)", fontsize=11)
    ax2.set_title("(B) Computational Efficiency (Inference Time vs K)", fontsize=12, weight='bold')
    for idx, row in df_results.iterrows():
        ax2.annotate(f"{row['Inference Time (s)']:.2f}s", xy=(idx, row['Inference Time (s)']),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # (3) Confusion Matrix for Best K
    ax3 = fig.add_subplot(gs[1, :])
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=FASHION_CLASSES, yticklabels=FASHION_CLASSES,
                cbar_kws={'label': 'Count'})
    ax3.set_title(f"(C) Confusion Matrix on Fashion MNIST (Optimal K = {best_k}, Accuracy = {best_acc*100:.2f}%)",
                  fontsize=12, weight='bold')
    ax3.set_xlabel("Predicted Fashion Category", fontsize=11)
    ax3.set_ylabel("True Fashion Category", fontsize=11)

    fig.suptitle("K-Nearest Neighbors (KNN) Performance & Complexity Analysis on Fashion MNIST",
                 fontsize=14, weight='bold', y=0.98)

    plot_path_local = os.path.join(os.path.dirname(__file__), "q08_knn_k_value_analysis.png")
    plot_path_global = os.path.join(output_dir, "q08_knn_k_value_analysis.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_results


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
