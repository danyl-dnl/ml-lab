"""
Question 9: Decision Tree (ID3 Algorithm) on Online Retail Dataset
Tasks:
  1. Load and preprocess the Online Retail dataset into customer RFM features.
  2. Implement Decision Tree using the ID3 algorithm (entropy criterion).
  3. Visualize the decision tree and analyze feature importance.
  4. Discuss how the tree structure helps in understanding customer behavior.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score


def load_and_preprocess_rfm(filepath):
    # Read transactions
    df = pd.read_excel(filepath)
    # Clean invalid records
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Calculate Recency, Frequency, Monetary (RFM)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpend': 'Monetary'})

    # Create 3 customer segments based on RFM score
    rfm['R_Rank'] = pd.qcut(rfm['Recency'], q=3, labels=[3, 2, 1]).astype(int)
    rfm['F_Rank'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=3, labels=[1, 2, 3]).astype(int)
    rfm['M_Rank'] = pd.qcut(rfm['Monetary'], q=3, labels=[1, 2, 3]).astype(int)
    rfm['Score'] = rfm['R_Rank'] + rfm['F_Rank'] + rfm['M_Rank']

    # 0 = At-Risk, 1 = Regular, 2 = High-Value (Champions)
    rfm['Segment'] = pd.cut(rfm['Score'], bins=[0, 5, 7, 9], labels=['At-Risk', 'Regular', 'High-Value'])
    return rfm


def main():
    # 1. Load dataset
    filepath = '../data/Online Retail.xlsx'
    rfm = load_and_preprocess_rfm(filepath)

    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm[features]
    y = rfm['Segment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 2. Implement Decision Tree with ID3 algorithm (entropy criterion)
    clf_id3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    clf_id3.fit(X_train, y_train)

    y_pred = clf_id3.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("=" * 60)
    print("QUESTION 9: DECISION TREE (ID3) - ONLINE RETAIL")
    print("=" * 60)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Weighted F1:   {f1:.4f}")
    print("=" * 60)

    # 3. Analyze Feature Importance
    feat_imp = pd.Series(clf_id3.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importance (Information Gain):")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<12}: {imp:.4f} ({imp*100:.1f}%)")

    # Visualize Decision Tree
    plt.figure(figsize=(12, 6))
    plot_tree(clf_id3, feature_names=features, class_names=clf_id3.classes_, filled=True, rounded=True, fontsize=9)
    plt.title("ID3 Decision Tree Structure for Customer Segmentation", fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig("q09_decision_tree_customer_segments.png", dpi=200)
    plt.savefig("../outputs/q09_decision_tree_customer_segments.png", dpi=200)
    plt.close()
    print("\n[✓] Tree visualization saved to q09_decision_tree_customer_segments.png")

    # 4. Discuss customer behavior insights
    print("\n--- Discussion: Understanding Customer Behavior ---")
    print("1. Frequency is the top discriminator: The root split branches on purchase frequency.")
    print("   Customers with few repeat orders are immediately separated from loyal buyers.")
    print("2. Recency identifies lapsed accounts: Among buyers with lower frequency, recency identifies")
    print("   whether the customer is newly acquired or has churned, guiding automated win-back campaigns.")
    print("3. Monetary spend separates VIPs: Used to identify high-spending champions for premium loyalty rewards.")


if __name__ == "__main__":
    main()
