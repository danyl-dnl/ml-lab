"""Question 6: Compare logistic regression with and without feature scaling."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the data. Use the included copy when possible.
repo = Path(__file__).resolve().parent.parent
data_file = repo / "data" / "pima-indians-diabetes.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
if data_file.exists():
    data = pd.read_csv(data_file, names=columns)
else:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    data = pd.read_csv(url, names=columns)

# Separate the input columns (X) from the answer column (y).
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train and evaluate the model using the original feature values.
plain_model = LogisticRegression(max_iter=1000, random_state=42)
plain_model.fit(X_train, y_train)
plain_predictions = plain_model.predict(X_test)

# 3. Scale features. Fit the scaler only on training data to avoid test-data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_model = LogisticRegression(max_iter=1000, random_state=42)
scaled_model.fit(X_train_scaled, y_train)
scaled_predictions = scaled_model.predict(X_test_scaled)

# 4. Compare accuracy, precision, recall, and F1-score.
results = pd.DataFrame([
    [accuracy_score(y_test, plain_predictions), precision_score(y_test, plain_predictions), recall_score(y_test, plain_predictions), f1_score(y_test, plain_predictions)],
    [accuracy_score(y_test, scaled_predictions), precision_score(y_test, scaled_predictions), recall_score(y_test, scaled_predictions), f1_score(y_test, scaled_predictions)],
], index=["Without scaling", "With scaling"], columns=["Accuracy", "Precision", "Recall", "F1-score"]).round(4)
print("QUESTION 6: LOGISTIC REGRESSION")
print(results)
print("Iterations without scaling:", plain_model.n_iter_[0])
print("Iterations with scaling:   ", scaled_model.n_iter_[0])
print("Scaling puts input columns on similar ranges and can help the model learn faster.")
