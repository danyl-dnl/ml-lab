"""Question 15: See how learning rate, batch size, and epochs affect a network."""
from pathlib import Path
import time
import pandas as pd
import torchvision.datasets as datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Fashion-MNIST and scale pixels to between 0 and 1.
data_folder = Path(__file__).resolve().parent.parent / "data"
train_set = datasets.FashionMNIST(root=str(data_folder), train=True, download=True)
test_set = datasets.FashionMNIST(root=str(data_folder), train=False, download=True)
X_train = train_set.data.numpy().reshape(-1, 784)[:6000] / 255.0
y_train = train_set.targets.numpy()[:6000]
X_test = test_set.data.numpy().reshape(-1, 784)[:1000] / 255.0
y_test = test_set.targets.numpy()[:1000]

# 2. Each row changes one setting while keeping the other settings mostly fixed.
experiments = [
    ("Default", 0.001, 64, 10), ("Higher learning rate", 0.01, 64, 10),
    ("Very high learning rate", 0.1, 64, 10), ("Small batch", 0.001, 32, 10),
    ("Large batch", 0.001, 128, 10), ("Fewer epochs", 0.001, 64, 5),
    ("More epochs", 0.001, 64, 15),
]

# 3. Train and evaluate one network for each setting.
results = []
for name, learning_rate, batch_size, epochs in experiments:
    model = MLPClassifier(hidden_layer_sizes=(64,), learning_rate_init=learning_rate, batch_size=batch_size, max_iter=epochs, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start
    predictions = model.predict(X_test)
    results.append([name, learning_rate, batch_size, epochs, accuracy_score(y_test, predictions), seconds])

# 4. Show the comparison and explain the settings.
results = pd.DataFrame(results, columns=["Experiment", "Learning rate", "Batch size", "Epochs", "Accuracy", "Training seconds"]).set_index("Experiment").round(4)
print("QUESTION 15: HYPERPARAMETER COMPARISON")
print(results)
print("Learning rate controls how large each learning step is; a very high value can make learning unstable.")
print("Batch size is how many examples are used for one update. Epochs are how many times the model sees the training set.")
print("More epochs can help until the model starts overfitting. Compare the measured accuracy and time above.")
