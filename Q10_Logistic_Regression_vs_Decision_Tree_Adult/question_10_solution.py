"""Question 10: Compare logistic regression and a decision tree for income."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the Adult dataset. A value of 1 means income is above $50K.
columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
file_path = Path(__file__).resolve().parent.parent / "data" / "adult.data"
data = pd.read_csv(file_path, names=columns, sep=r",\s*", engine="python", na_values="?").dropna()
y = (data["income"] == ">50K").astype(int)
input_columns = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "marital_status", "sex"]
X = pd.get_dummies(data[input_columns], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Logistic regression works better when input columns are scaled.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)

# 3. A decision tree uses simple yes/no splits and does not need scaling.
tree_model = DecisionTreeClassifier(max_depth=6, random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# Compare accuracy, precision, recall, and F1-score for both models.
rows = []
for name, predictions in [("Logistic Regression", logistic_predictions), ("Decision Tree", tree_predictions)]:
    rows.append([name, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions), f1_score(y_test, predictions)])
results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]).set_index("Model").round(4)
print("QUESTION 10: LOGISTIC REGRESSION VS DECISION TREE")
print(results)

# 4. Simple interpretation guide.
print("Logistic regression summarizes how each input relates to the odds of higher income, assuming other inputs stay the same.")
print("A decision tree is read as a set of if/then rules and can show non-linear patterns. Neither model proves that a feature causes income.")
