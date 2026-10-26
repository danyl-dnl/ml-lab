"""
Question 9: Decision Tree Classifier using ID3 Algorithm on Online Retail Dataset
Description:
    Implement a Decision Tree classifier using the ID3 algorithm (Information Gain
    based on Shannon Entropy) to segment customers based on their purchasing
    behavior using the Online Retail dataset. Analyze the tree structure and
    discuss feature importance.

Tasks:
    1. Load and preprocess the Online Retail dataset into customer-level RFM features
       (Recency, Frequency, Monetary, AvgBasketValue, TotalUnits).
    2. Segment customers into actionable behavioral tiers (Champions, Regular, At-Risk).
    3. Implement Decision Tree with ID3 algorithm (Information Gain / Entropy criterion).
    4. Visualize the decision tree, analyze feature importance, and plot confusion matrix.
    5. Discuss how the tree structure helps in understanding customer behavior.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def calculate_entropy(y: np.ndarray) -> float:
    """Calculate Shannon Entropy: H(S) = -sum(p_i * log2(p_i))."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(-np.sum([p * np.log2(p) for p in probabilities if p > 0]))


def calculate_information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate Information Gain: IG = H(Parent) - [|Left|/|Parent| * H(Left) + |Right|/|Parent| * H(Right)]."""
    n = len(y_parent)
    if n == 0 or len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    h_parent = calculate_entropy(y_parent)
    h_left = calculate_entropy(y_left)
    h_right = calculate_entropy(y_right)
    weighted_entropy = (len(y_left) / n) * h_left + (len(y_right) / n) * h_right
    return float(h_parent - weighted_entropy)


def load_and_preprocess_rfm(data_path: str) -> pd.DataFrame:
    """
    Load transactions from Online Retail dataset and engineer RFM features:
    - Recency: Days since last purchase
    - Frequency: Count of unique invoices
    - Monetary: Total monetary spend
    - AvgBasketValue: Monetary / Frequency
    - TotalUnits: Total quantity purchased
    """
    print(f"Loading transaction records from {data_path}...")
    if data_path.endswith('.xlsx'):
        # Read transactions (sampling or full read)
        df_raw = pd.read_excel(data_path)
    else:
        df_raw = pd.read_csv(data_path, encoding='ISO-8859-1')

    print(f"Raw transactions loaded: {len(df_raw)} records.")

    # Data hygiene: drop missing CustomerIDs and cancelled orders (InvoiceNo starting with C or negative quantity)
    df = df_raw.dropna(subset=['CustomerID']).copy()
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Snapshot reference date: {snapshot_date.strftime('%Y-%m-%d')}")

    # Aggregate to Customer level
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'TotalUnits']
    rfm['AvgBasketValue'] = rfm['Monetary'] / rfm['Frequency']

    # Filter extreme outliers for robust model training
    rfm = rfm[(rfm['Monetary'] > 0) & (rfm['Monetary'] < rfm['Monetary'].quantile(0.99))]
    rfm = rfm[rfm['Frequency'] < rfm['Frequency'].quantile(0.99)]

    # Assign Customer Segments based on standard RFM Quartile Scoring
    # R score: 1 to 3 (3 = most recent, 1 = least recent)
    # F & M scores: 1 to 3 (3 = highest, 1 = lowest)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=3, labels=[3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=3, labels=[1, 2, 3]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=3, labels=[1, 2, 3]).astype(int)
    rfm['RFM_Sum'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    # Define 3 distinct marketing segments:
    # Segment 0: "At-Risk / Lapsed" (RFM Sum 3-5)
    # Segment 1: "Regular / Potential" (RFM Sum 6-7)
    # Segment 2: "Champions / High-Value" (RFM Sum 8-9)
    def assign_segment(score):
        if score >= 8:
            return "Champions"
        elif score >= 6:
            return "Regular"
        else:
            return "At-Risk"

    rfm['Segment'] = rfm['RFM_Sum'].apply(assign_segment)
    print("\nCustomer Segment Distribution:")
    print(rfm['Segment'].value_counts())
    return rfm


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 9: DECISION TREE (ID3) FOR CUSTOMER SEGMENTATION")
    print("=" * 70)

    # 1. Locate dataset
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    xlsx_path = os.path.join(data_dir, "Online Retail.xlsx")
    csv_path = os.path.join(data_dir, "OnlineRetail.csv")

    target_file = xlsx_path if os.path.exists(xlsx_path) else csv_path
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Neither {xlsx_path} nor {csv_path} was found.")

    rfm = load_and_preprocess_rfm(target_file)

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'AvgBasketValue', 'TotalUnits']
    X = rfm[feature_cols].values
    y_raw = rfm['Segment'].values

    # Mapping segments to numerical labels
    label_mapping = {"At-Risk": 0, "Regular": 1, "Champions": 2}
    inv_label_mapping = {0: "At-Risk", 1: "Regular", 2: "Champions"}
    y = np.array([label_mapping[s] for s in y_raw])
    class_names = ["At-Risk", "Regular", "Champions"]

    # Demonstrate ID3 Entropy & Information Gain calculation on the root node
    root_entropy = calculate_entropy(y)
    print(f"\n[ID3 Principle] Root Dataset Shannon Entropy: {root_entropy:.4f} bits")
    # Example Information Gain test on Recency split at median
    rec_median = np.median(X[:, 0])
    left_mask = X[:, 0] <= rec_median
    ig_recency = calculate_information_gain(y, y[left_mask], y[~left_mask])
    print(f"[ID3 Principle] Sample Information Gain on Recency split <= {rec_median:.1f} days: {ig_recency:.4f} bits")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Train Decision Tree using ID3 (Entropy criterion)
    print("\nTraining Decision Tree with ID3 (Entropy criterion, max_depth=3 for optimal interpretability)...")
    clf_id3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=30, random_state=42)
    clf_id3.fit(X_train, y_train)

    y_pred = clf_id3.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print(" ID3 DECISION TREE EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall:    {rec:.4f}")
    print(f"Weighted F1-Score:  {f1:.4f}")
    print("=" * 60)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Feature Importance Analysis
    importances = clf_id3.feature_importances_
    df_feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance (Entropy Reduction)': importances
    }).sort_values(by='Importance (Entropy Reduction)', ascending=False)

    print("\nFeature Importance (Total Entropy Reduction / Information Gain):")
    print(df_feat_imp.to_string(index=False))

    # Visualizations
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.3, wspace=0.25)

    # (1) Decision Tree Visual Structure
    ax_tree = fig.add_subplot(gs[0, :])
    plot_tree(
        clf_id3,
        feature_names=feature_cols,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax_tree,
        precision=2,
    )
    ax_tree.set_title("ID3 Decision Tree Structure (Entropy & Information Gain Splits for Customer Segmentation)",
                      fontsize=13, weight='bold', pad=12)

    # (2) Feature Importance Bar Chart
    ax_feat = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df_feat_imp, x='Importance (Entropy Reduction)', y='Feature', hue='Feature', ax=ax_feat, palette='viridis', edgecolor='black', legend=False)
    ax_feat.set_title("Feature Importance (Relative Information Gain)", fontsize=12, weight='bold')
    ax_feat.set_xlabel("Normalized Gini / Information Gain Score")
    for i, v in enumerate(df_feat_imp['Importance (Entropy Reduction)']):
        ax_feat.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10, weight='bold')

    # (3) Confusion Matrix
    ax_cm = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 12, 'weight': 'bold'})
    ax_cm.set_title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%, F1: {f1:.3f})", fontsize=12, weight='bold')
    ax_cm.set_xlabel("Predicted Customer Segment")
    ax_cm.set_ylabel("True Customer Segment")

    fig.suptitle("Customer Segmentation using ID3 Decision Tree on Online Retail Dataset",
                 fontsize=15, weight='bold', y=0.98)

    plot_path_local = os.path.join(os.path.dirname(__file__), "q09_decision_tree_customer_segments.png")
    plot_path_global = os.path.join(output_dir, "q09_decision_tree_customer_segments.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_feat_imp


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
