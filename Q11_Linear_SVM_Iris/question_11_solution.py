"""Question 11: Use a linear SVM to separate Iris Setosa from other flowers."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load Iris. Use petal length and width so we can draw a 2D picture.
iris = load_iris()
X = iris.data[:, [2, 3]]
y = (iris.target == 0).astype(int)  # 1 = Setosa; 0 = another Iris type

# 2. Train a linear support vector machine.
model = SVC(kernel="linear", C=1.0)
model.fit(X, y)
predictions = model.predict(X)
weights = model.coef_[0]
margin_width = 2 / np.linalg.norm(weights)
print("QUESTION 11: LINEAR SVM")
print("Accuracy on these Iris examples:", round(accuracy_score(y, predictions), 4))
print("Note: this is training accuracy because the model is measured on the same examples it learned from.")
print("Support vectors:", len(model.support_vectors_))
print("Margin width:", round(margin_width, 4))

# 3. Draw the data, decision boundary, and margins.
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Setosa")
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Other Iris")
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=130, facecolors="none", edgecolors="black", label="Support vectors")
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
values = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, values, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors=["green", "black", "green"])
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Linear SVM: boundary and margins")
plt.legend()
plt.tight_layout()
output_path = Path(__file__).resolve().parent / "q11_linear_svm_iris_margin.png"
plt.savefig(output_path, dpi=200)
plt.close()
print("Plot saved to:", output_path)

# 4. The margin is the gap between the decision line and its nearest examples.
print("The nearest examples are called support vectors; they help set the decision boundary.")
print("A wider margin usually makes the classifier less sensitive to small changes in the data.")
