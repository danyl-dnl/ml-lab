"""Question 13: Compare neural networks with different hidden-layer sizes."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the red wine data. Here, quality 6 or above means "good".
file_path = Path(__file__).resolve().parent.parent / "data" / "winequality-red.csv"
data = pd.read_csv(file_path, sep=";")
X = data.drop("quality", axis=1)
y = (data["quality"] >= 6).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2-3. Train three networks and compare their test results.
architectures = [("1 layer: 32", (32,)), ("2 layers: 64, 32", (64, 32)), ("3 layers: 128, 64, 32", (128, 64, 32))]
results = []
for name, layer_sizes in architectures:
    model = MLPClassifier(hidden_layer_sizes=layer_sizes, max_iter=600, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append([name, accuracy_score(y_test, predictions), f1_score(y_test, predictions), model.n_iter_])
results = pd.DataFrame(results, columns=["Network", "Accuracy", "F1-score", "Training iterations"]).set_index("Network").round(4)
print("QUESTION 13: NEURAL NETWORK ARCHITECTURES")
print(results)

# 4. More layers and neurons let a network learn more patterns, but also add complexity.
print("A small network is simpler but may miss patterns. A large network can learn more, but may overfit.")
print("Compare the test scores above to see which size worked best for this run.")
