"""Question 12: Compare linear, polynomial, and RBF SVM kernels."""
from pathlib import Path
import time
import pandas as pd
import torchvision.datasets as datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# 1. Load and scale Fashion-MNIST pixel values to between 0 and 1.
data_folder = Path(__file__).resolve().parent.parent / "data"
train_set = datasets.FashionMNIST(root=str(data_folder), train=True, download=True)
test_set = datasets.FashionMNIST(root=str(data_folder), train=False, download=True)
X_train = train_set.data.numpy().reshape(-1, 784)[:3000] / 255.0
y_train = train_set.targets.numpy()[:3000]
X_test = test_set.data.numpy().reshape(-1, 784)[:1000] / 255.0
y_test = test_set.targets.numpy()[:1000]

# 2-3. Train each SVM, time training, and compare accuracy and weighted F1.
results = []
for name, kernel in [("Linear", "linear"), ("Polynomial", "poly"), ("RBF", "rbf")]:
    model = SVC(kernel=kernel, degree=3, random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start
    predictions = model.predict(X_test)
    results.append([name, accuracy_score(y_test, predictions), f1_score(y_test, predictions, average="weighted"), training_seconds])
results = pd.DataFrame(results, columns=["Kernel", "Accuracy", "Weighted F1", "Training seconds"]).set_index("Kernel").round(4)
print("QUESTION 12: SVM KERNEL COMPARISON")
print(results)

# 4. Explain each choice simply.
print("Linear: simplest and often quickest, but only makes straight boundaries.")
print("Polynomial: can model combinations of features, but may take longer and needs tuning.")
print("RBF: can make flexible curved boundaries, but can be slower and needs tuning.")
