"""Question 9: Build and explain an entropy-based customer decision tree."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score

# 1. Load transactions and remove rows without a customer or with invalid quantities/prices.
file_path = Path(__file__).resolve().parent.parent / "data" / "Online Retail.xlsx"
transactions = pd.read_excel(file_path)
transactions = transactions.dropna(subset=["CustomerID"])
transactions = transactions[(transactions["Quantity"] > 0) & (transactions["UnitPrice"] > 0)].copy()
transactions["TotalSpend"] = transactions["Quantity"] * transactions["UnitPrice"]
transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
snapshot = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

# Make one row per customer: Recency (days since last order), Frequency (number of orders), Monetary (total spend).
rfm = transactions.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda dates: (snapshot - dates.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalSpend", "sum"),
)

# Give each RFM measure a simple low/middle/high score. Ranking handles tied values safely.
rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), 3, labels=[3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
rfm["Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
rfm["Segment"] = pd.cut(rfm["Score"], bins=[0, 5, 7, 10], labels=["At-Risk", "Regular", "High-Value"], include_lowest=True)

# 2. Train an ID3-style decision tree. Entropy is the split rule used by ID3.
features = ["Recency", "Frequency", "Monetary"]
X = rfm[features]
y = rfm["Segment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

# Show model scores and which features helped split the data most.
print("QUESTION 9: CUSTOMER SEGMENTATION")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Weighted F1-score:", round(f1_score(y_test, predictions, average="weighted"), 4))
print("Feature importance (larger means the tree used it more):")
print(pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False).round(4))

# 3. Draw and save the tree picture.
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=features, class_names=[str(name) for name in tree.classes_], filled=True, rounded=True, fontsize=9)
plt.title("Customer Segments: Entropy Decision Tree")
plt.tight_layout()
output_path = Path(__file__).resolve().parent / "q09_decision_tree_customer_segments.png"
plt.savefig(output_path, dpi=200)
plt.close()
print("Tree picture saved to:", output_path)

# 4. Explain what the tree means.
print("The tree shows RFM thresholds used to assign the score-based customer segments.")
print("These are descriptive rules because the segment labels were created from the same RFM measures; they do not predict future purchases by themselves.")
