"""Question 8: Test different K values for K-nearest neighbors."""
from pathlib import Path
import time
import pandas as pd
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Fashion-MNIST and make each 28x28 image into 784 pixel values.
data_folder = Path(__file__).resolve().parent.parent / "data"
train_set = datasets.FashionMNIST(root=str(data_folder), train=True, download=True)
test_set = datasets.FashionMNIST(root=str(data_folder), train=False, download=True)
X_train = train_set.data.numpy().reshape(-1, 784)[:5000] / 255.0
y_train = train_set.targets.numpy()[:5000]
X_test = test_set.data.numpy().reshape(-1, 784)[:1000] / 255.0
y_test = test_set.targets.numpy()[:1000]

# 2-3. Try several values of K. K is the number of nearby training images considered.
results = []
for k in [1, 3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - fit_start
    start = time.time()
    predictions = model.predict(X_test)
    seconds = time.time() - start
    results.append([k, accuracy_score(y_test, predictions), f1_score(y_test, predictions, average="weighted"), fit_seconds, seconds])

# 4. Compare results and explain the tradeoff.
results = pd.DataFrame(results, columns=["K", "Accuracy", "Weighted F1", "Fit seconds", "Prediction seconds"]).set_index("K").round(4)
print("QUESTION 8: K-NEAREST NEIGHBORS")
print(results)
print("A small K is sensitive to individual examples. A larger K smooths predictions, but an extremely large K can underfit.")
print("KNN stores training examples, so fitting is quick but predicting can take longer as the dataset grows.")
