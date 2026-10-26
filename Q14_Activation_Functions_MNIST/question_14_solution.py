"""
Question 14: Neural Networks with Different Activation Functions (Sigmoid, ReLU, Tanh) on MNIST
Tasks:
  1. Load and preprocess the MNIST dataset.
  2. Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
  3. Train and evaluate each network.
  4. Compare training times, convergence, and classification accuracy.
"""

import time
import warnings
import pandas as pd
import torchvision.datasets as dsets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


def load_mnist_data(n_train=8000, n_test=2000):
    train_set = dsets.MNIST(root='../data', train=True, download=True)
    test_set = dsets.MNIST(root='../data', train=False, download=True)

    X_train = train_set.data.numpy().reshape(-1, 28 * 28)[:n_train] / 255.0
    y_train = train_set.targets.numpy()[:n_train]

    X_test = test_set.data.numpy().reshape(-1, 28 * 28)[:n_test] / 255.0
    y_test = test_set.targets.numpy()[:n_test]

    return X_train, y_train, X_test, y_test


def main():
    # 1. Load and preprocess dataset
    X_train, y_train, X_test, y_test = load_mnist_data()

    # 2. Implement networks with Sigmoid, Tanh, and ReLU activations
    activations = [
        ("Sigmoid (Logistic)", "logistic"),
        ("Tanh", "tanh"),
        ("ReLU", "relu"),
    ]

    records = []
    print("Training neural networks with different activation functions on MNIST...")

    # 3. Train and evaluate each network
    for display_name, act_func in activations:
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation=act_func,
            max_iter=25,
            random_state=42
        )

        t0 = time.time()
        mlp.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        records.append({
            "Activation Function": display_name,
            "Test Accuracy": round(acc, 4),
            "Final Loss": round(mlp.loss_, 4),
            "Training Time (s)": round(train_time, 2),
            "Iterations": mlp.n_iter_,
        })

    # 4. Compare training times, convergence, and accuracy
    results = pd.DataFrame(records).set_index("Activation Function")
    print("\n" + "=" * 70)
    print("QUESTION 14: ACTIVATION FUNCTIONS ON MNIST (SIGMOID, TANH, RELU)")
    print("=" * 70)
    print(results)
    print("=" * 70)

    # Discussion
    print("\n--- Discussion: Impact of Activation Functions ---")
    print("1. Convergence & Accuracy:")
    print("   - ReLU achieves the lowest loss and highest accuracy due to its non-saturating derivative (1 for z > 0).")
    print("   - Tanh outperforms Sigmoid because it is zero-centered (range [-1, 1]), preventing directional gradient bias.")
    print("2. The Vanishing Gradient Problem:")
    print("   - Sigmoid has a maximum derivative of 0.25. As backpropagation proceeds across multiple layers,")
    print("     gradients shrink exponentially towards zero, slowing down training.")
    print("3. Computational Speed:")
    print("   - ReLU is computationally simplest (max(0, z)), eliminating expensive exponential calculations.")


if __name__ == "__main__":
    main()
