"""
Question 8: K-Nearest Neighbors (KNN) on Fashion MNIST Dataset
Tasks:
  1. Load and preprocess the Fashion MNIST dataset.
  2. Implement KNN for multi-class classification.
  3. Experiment with different values of K and evaluate performance.
  4. Discuss the impact of different K values on model accuracy and computational efficiency.
"""

import time
import pandas as pd
import torchvision.datasets as dsets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def load_data(n_train=5000, n_test=1000):
    train_set = dsets.FashionMNIST(root='../data', train=True, download=True)
    test_set = dsets.FashionMNIST(root='../data', train=False, download=True)

    # Flatten 28x28 images to 784-dim vectors and scale to [0, 1]
    X_train = train_set.data.numpy().reshape(-1, 28 * 28)[:n_train] / 255.0
    y_train = train_set.targets.numpy()[:n_train]

    X_test = test_set.data.numpy().reshape(-1, 28 * 28)[:n_test] / 255.0
    y_test = test_set.targets.numpy()[:n_test]

    return X_train, y_train, X_test, y_test


def main():
    # 1. Load and preprocess dataset
    X_train, y_train, X_test, y_test = load_data()

    # 2 & 3. Experiment with different values of K
    k_values = [1, 3, 5, 7, 9]
    records = []

    print("Training and evaluating KNN for different values of K...")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit model
        knn.fit(X_train, y_train)

        # Measure inference time and evaluate
        t0 = time.time()
        y_pred = knn.predict(X_test)
        inference_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        records.append({
            "K": k,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "Inference Time (s)": round(inference_time, 3),
        })

    results = pd.DataFrame(records).set_index("K")

    print("\n" + "=" * 55)
    print("QUESTION 8: KNN ON FASHION MNIST")
    print("=" * 55)
    print(results)
    print("=" * 55)

    # 4. Discuss impact of K on accuracy and efficiency
    print("\n--- Discussion: Impact of K Values ---")
    print("Model Accuracy (Bias-Variance Tradeoff):")
    print("  - Very small K (K=1): Low bias but high variance; sensitive to noisy or mislabeled pixels.")
    print("  - Moderate K (K=5 to 9): Smooths decision boundaries, improving generalization.")
    print("  - Very large K: High bias; includes samples from distant classes, causing underfitting.")
    print("Computational Efficiency:")
    print("  - KNN is a lazy learner (no training phase, O(1)).")
    print("  - Test/Inference time is O(N * d) per sample, making prediction computationally expensive for large datasets.")


if __name__ == "__main__":
    main()
