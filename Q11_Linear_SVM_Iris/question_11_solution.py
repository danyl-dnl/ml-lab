"""
Question 11: Linear Support Vector Machine (SVM) on Iris Dataset
Tasks:
  1. Load and preprocess the Iris dataset.
  2. Implement a Linear SVM for binary classification (Setosa vs Non-Setosa).
  3. Visualize the decision boundary and margin.
  4. Discuss the concept of the margin and how it influences classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    # 1. Load dataset (using 2 features for 2D boundary visualization: Petal Length and Petal Width)
    iris = load_iris()
    X = iris.data[:, [2, 3]]  # Petal length, Petal width
    # 2. Binary classification: Setosa (1) vs Non-Setosa (0)
    y = (iris.target == 0).astype(int)

    # Implement Linear SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)

    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin_width = 2.0 / np.linalg.norm(w)

    print("=" * 55)
    print("QUESTION 11: LINEAR SVM (IRIS DATASET)")
    print("=" * 55)
    print(f"Accuracy:                  {acc * 100:.2f}% (Linearly Separable)")
    print(f"Number of Support Vectors: {len(clf.support_vectors_)}")
    print(f"Margin Width (2 / ||w||):  {margin_width:.4f}")
    print("=" * 55)

    # 3. Visualize decision boundary and margin
    plt.figure(figsize=(8, 6))

    # Scatter points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='#2b5c8f', label='Setosa (1)', edgecolor='k', s=50)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='#e06666', label='Non-Setosa (0)', edgecolor='k', s=50)

    # Highlight support vectors
    sv = clf.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=140, facecolors='none', edgecolors='black', linewidth=1.5, label='Support Vectors')

    # Plot decision boundary and margin lines
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors=['green', 'black', 'green'], linewidths=[1.5, 2, 1.5])

    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(f"Linear SVM Decision Boundary & Margin (Margin Width = {margin_width:.3f})", weight='bold')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("q11_linear_svm_iris_margin.png", dpi=200)
    plt.savefig("../outputs/q11_linear_svm_iris_margin.png", dpi=200)
    plt.close()
    print("[✓] Plot saved to q11_linear_svm_iris_margin.png")

    # 4. Discuss concept of the margin
    print("\n--- Discussion: The Concept of the Margin ---")
    print("1. Definition of Margin:")
    print("   - The margin is the separation distance between the decision boundary (w^T x + b = 0)")
    print("     and the closest training points from either class. The geometric width is 2 / ||w||.")
    print("2. Role of Support Vectors:")
    print("   - Only points lying directly on the margin boundaries (w^T x + b = +/-1) determine the hyperplane.")
    print("   - Moving non-support vector points has zero impact on the boundary.")
    print("3. Influence on Generalization:")
    print("   - Maximizing the margin minimizes structural risk and provides maximum safety against unseen test noise.")


if __name__ == "__main__":
    main()
