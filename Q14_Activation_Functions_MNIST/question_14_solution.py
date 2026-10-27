"""Question 14: Compare sigmoid, tanh, and ReLU activation functions."""
from pathlib import Path
import time
import pandas as pd
import torchvision.datasets as datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load MNIST images and scale pixel values between 0 and 1.
data_folder = Path(__file__).resolve().parent.parent / "data"
train_set = datasets.MNIST(root=str(data_folder), train=True, download=True)
test_set = datasets.MNIST(root=str(data_folder), train=False, download=True)
X_train = train_set.data.numpy().reshape(-1, 784)[:8000] / 255.0
y_train = train_set.targets.numpy()[:8000]
X_test = test_set.data.numpy().reshape(-1, 784)[:2000] / 255.0
y_test = test_set.targets.numpy()[:2000]

# 2-3. Train the same small network with each activation function.
activations = [("Sigmoid", "logistic"), ("Tanh", "tanh"), ("ReLU", "relu")]
results = []
for name, activation in activations:
    model = MLPClassifier(hidden_layer_sizes=(64,), activation=activation, max_iter=25, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start
    predictions = model.predict(X_test)
    results.append([name, accuracy_score(y_test, predictions), model.loss_, seconds, model.n_iter_])

# 4. Compare test accuracy, final training loss, and training time.
results = pd.DataFrame(results, columns=["Activation", "Accuracy", "Final loss", "Training seconds", "Iterations"]).set_index("Activation").round(4)
print("QUESTION 14: ACTIVATION FUNCTIONS")
print(results)
print("ReLU is commonly fast to train. Sigmoid and tanh can learn more slowly when gradients become very small.")
print("The Iterations column shows how long each model trained; if it equals 25, training reached the set limit and may not have fully converged.")
print("Use the table to see which activation performed best for this run.")
