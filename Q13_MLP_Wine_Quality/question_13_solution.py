"""
Question 13: Multilayer Feed-Forward Network (MLP) on Wine Quality Dataset
Tasks:
  1. Load and preprocess the Wine Quality dataset.
  2. Design and implement an MLP with varying architectures (different hidden layers and neurons).
  3. Train and evaluate the network.
  4. Discuss the impact of architecture choices on performance.
"""

import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')


def main():
    # 1. Load and preprocess dataset
    df = pd.read_csv("../data/winequality-red.csv", sep=";")

    # Binary target: 1 = Good quality (score >= 6), 0 = Regular/Poor (score < 6)
    X = df.drop("quality", axis=1)
    y = (df["quality"] >= 6).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # 2. Design MLP with varying architectures (different hidden layers & neurons)
    architectures = [
        ("Shallow: 1 Layer (32 neurons)", (32,)),
        ("Medium: 2 Layers (64, 32 neurons)", (64, 32)),
        ("Deep: 3 Layers (128, 64, 32 neurons)", (128, 64, 32)),
    ]

    records = []
    print("Training MLP classifiers with varying architectures...")
    # 3. Train and evaluate each network
    for name, hidden_layers in architectures:
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=600, random_state=42)
        mlp.fit(X_train_scaled, y_train)

        y_pred = mlp.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        records.append({
            "Architecture": name,
            "Hidden Layers": str(hidden_layers),
            "Test Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "Iterations to Converge": mlp.n_iter_,
        })

    results = pd.DataFrame(records).set_index("Architecture")
    print("\n" + "=" * 70)
    print("QUESTION 13: MLP ARCHITECTURAL COMPARISON (WINE QUALITY)")
    print("=" * 70)
    print(results)
    print("=" * 70)

    # 4. Discuss impact of architecture choices
    print("\n--- Discussion: Impact of Architecture Choices on Performance ---")
    print("1. Capacity & Expressiveness (Width & Depth):")
    print("   - Adding hidden layers enables the network to learn hierarchical non-linear feature combinations.")
    print("2. Underfitting vs Overfitting:")
    print("   - Shallow networks (32) may underfit complex decision boundaries.")
    print("   - Deep networks (128, 64, 32) have much higher capacity, which can lead to overfitting on smaller")
    print("     tabular datasets unless properly regularized.")
    print("3. Convergence Dynamics:")
    print("   - Deeper networks require more weight updates and careful learning rate tuning to avoid vanishing gradients.")


if __name__ == "__main__":
    main()
