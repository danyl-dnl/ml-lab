"""
Question 12: SVM Kernels (Linear, Polynomial, RBF) on Fashion MNIST
Tasks:
  1. Load and preprocess the Fashion MNIST dataset.
  2. Implement SVM with linear, polynomial, and RBF kernels.
  3. Compare the classification performance for each kernel.
  4. Discuss the strengths and weaknesses of each kernel type.
"""

import time
import pandas as pd
import torchvision.datasets as dsets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def load_fashion_data(n_train=3000, n_test=1000):
    train_set = dsets.FashionMNIST(root='../data', train=True, download=True)
    test_set = dsets.FashionMNIST(root='../data', train=False, download=True)

    X_train = train_set.data.numpy().reshape(-1, 28 * 28)[:n_train] / 255.0
    y_train = train_set.targets.numpy()[:n_train]

    X_test = test_set.data.numpy().reshape(-1, 28 * 28)[:n_test] / 255.0
    y_test = test_set.targets.numpy()[:n_test]

    return X_train, y_train, X_test, y_test


def main():
    # 1. Load and preprocess dataset
    X_train, y_train, X_test, y_test = load_fashion_data()

    # 2. Implement SVM with linear, polynomial, and RBF kernels
    kernels = [
        ("Linear Kernel", SVC(kernel="linear", random_state=42)),
        ("Polynomial Kernel (degree=3)", SVC(kernel="poly", degree=3, random_state=42)),
        ("RBF Kernel", SVC(kernel="rbf", random_state=42)),
    ]

    records = []
    print("Training and evaluating SVM classifiers with different kernels...")
    for name, clf in kernels:
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        records.append({
            "Kernel Type": name,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "Train Time (s)": round(train_time, 2),
        })

    # 3. Compare classification performance
    results = pd.DataFrame(records).set_index("Kernel Type")
    print("\n" + "=" * 65)
    print("QUESTION 12: SVM KERNEL COMPARISON (FASHION MNIST)")
    print("=" * 65)
    print(results)
    print("=" * 65)

    # 4. Discuss strengths and weaknesses
    print("\n--- Discussion: Strengths and Weaknesses of Each Kernel ---")
    print("1. Linear Kernel:")
    print("   + Strengths: Fast training and testing, simple, avoids overfitting.")
    print("   - Weaknesses: Cannot model non-linear image manifolds; assumes linear separability.")
    print("2. Polynomial Kernel:")
    print("   + Strengths: Explicitly models high-order feature interactions (combinations of pixels).")
    print("   - Weaknesses: Prone to numerical instability with high degrees; higher training cost.")
    print("3. RBF (Gaussian) Kernel:")
    print("   + Strengths: Maps data to an infinite-dimensional space; universally models complex non-linear shapes.")
    print("   - Weaknesses: Requires tuning gamma; slower inference on large numbers of support vectors.")


if __name__ == "__main__":
    main()
