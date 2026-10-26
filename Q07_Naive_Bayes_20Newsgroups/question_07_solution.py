"""
Question 7: Naïve Bayes Classifier on 20 Newsgroups Dataset
Description:
    Implement a Naïve Bayes classifier to categorize text documents into
    topics using the 20 Newsgroups dataset. Compare the performance of
    Multinomial Naïve Bayes with Bernoulli Naïve Bayes.

Tasks:
    1. Load and preprocess the 20 Newsgroups dataset (tokenization, stop words, vectorization).
    2. Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
    3. Evaluate and compare performance using Accuracy, Macro/Weighted F1, and Confusion Matrices.
    4. Analyze and discuss the strengths and weaknesses of each variant for text classification.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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


def load_and_preprocess_newsgroups(categories=None):
    """
    Fetch train and test splits of 20 Newsgroups with headers/footers stripped
    to prevent shortcut learning/overfitting to metadata.
    """
    print("Loading 20 Newsgroups dataset (stripping headers, footers, quotes)...")
    train_data = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42,
    )
    test_data = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42,
    )

    print(f"Loaded {len(train_data.data)} training documents, {len(test_data.data)} test documents.")
    print(f"Categories ({len(train_data.target_names)}): {train_data.target_names}")
    return train_data, test_data


def evaluate_classifier(name, model, X_train, y_train, X_test, y_test):
    """Fit model, record timings, and calculate performance metrics."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        "Classifier": name,
        "Accuracy": acc,
        "Precision (W)": prec,
        "Recall (W)": rec,
        "F1-Score (Macro)": f1_macro,
        "F1-Score (Weighted)": f1_weighted,
        "Train Time (s)": train_time,
        "Inference Time (s)": test_time,
    }, y_pred


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 7: MULTINOMIAL VS BERNOULLI NAÏVE BAYES (20 NEWSGROUPS)")
    print("=" * 70)

    # Use 4 distinct representative categories across diverse domains:
    # Philosophy/Religion, Computer Science, Natural Science, and Politics/Society
    categories = [
        'alt.atheism',
        'comp.graphics',
        'sci.med',
        'soc.religion.christian',
    ]

    train_data, test_data = load_and_preprocess_newsgroups(categories=categories)
    y_train = train_data.target
    y_test = test_data.target
    class_names = [c.split('.')[-1].capitalize() for c in train_data.target_names]

    # 1. Feature Representation for Multinomial Naïve Bayes
    # Multinomial NB leverages term frequency (CountVectorizer) or TF-IDF
    print("\n[1] Vectorizing text for Multinomial NB (term frequencies)...")
    count_vec = CountVectorizer(stop_words='english', min_df=3, max_df=0.85)
    X_train_mnb = count_vec.fit_transform(train_data.data)
    X_test_mnb = count_vec.transform(test_data.data)
    print(f"Vocabulary size: {X_train_mnb.shape[1]} words")

    # 2. Feature Representation for Bernoulli Naïve Bayes
    # Bernoulli NB strictly requires binary presence/absence (binary=True)
    print("[2] Vectorizing text for Bernoulli NB (binary occurrence)...")
    bern_vec = CountVectorizer(stop_words='english', min_df=3, max_df=0.85, binary=True)
    X_train_bnb = bern_vec.fit_transform(train_data.data)
    X_test_bnb = bern_vec.transform(test_data.data)

    # 3. Fit Multinomial Naive Bayes
    mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing
    res_mnb, y_pred_mnb = evaluate_classifier("Multinomial NB", mnb, X_train_mnb, y_train, X_test_mnb, y_test)

    # 4. Fit Bernoulli Naive Bayes
    bnb = BernoulliNB(alpha=1.0)  # Laplace smoothing
    res_bnb, y_pred_bnb = evaluate_classifier("Bernoulli NB", bnb, X_train_bnb, y_train, X_test_bnb, y_test)

    # Also test Multinomial NB with TF-IDF for completeness
    tfidf_vec = TfidfVectorizer(stop_words='english', min_df=3, max_df=0.85)
    X_train_tfidf = tfidf_vec.fit_transform(train_data.data)
    X_test_tfidf = tfidf_vec.transform(test_data.data)
    mnb_tfidf = MultinomialNB(alpha=0.1)
    res_mnb_tfidf, y_pred_tfidf = evaluate_classifier("Multinomial NB (TF-IDF)", mnb_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test)

    # Comparison summary
    df_results = pd.DataFrame([res_mnb, res_bnb, res_mnb_tfidf])
    print("\n" + "=" * 80)
    print(" PERFORMANCE COMPARISON SUMMARY: NAÏVE BAYES CLASSIFIERS")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    print("\n--- Detailed Classification Report: Multinomial NB ---")
    print(classification_report(y_test, y_pred_mnb, target_names=class_names))

    print("\n--- Detailed Classification Report: Bernoulli NB ---")
    print(classification_report(y_test, y_pred_bnb, target_names=class_names))

    # Confusion matrices
    cm_mnb = confusion_matrix(y_test, y_pred_mnb)
    cm_bnb = confusion_matrix(y_test, y_pred_bnb)

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Naïve Bayes Text Classification: Multinomial vs Bernoulli Variants\n(20 Newsgroups Dataset)",
                 fontsize=14, weight='bold', y=1.02)

    # Plot Confusion Matrix: Multinomial NB
    sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f"Multinomial NB (Accuracy: {res_mnb['Accuracy']:.3f})\nMacro F1: {res_mnb['F1-Score (Macro)']:.3f}", fontsize=11, weight='bold')
    axes[0].set_xlabel("Predicted Topic")
    axes[0].set_ylabel("True Topic")

    # Plot Confusion Matrix: Bernoulli NB
    sns.heatmap(cm_bnb, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f"Bernoulli NB (Accuracy: {res_bnb['Accuracy']:.3f})\nMacro F1: {res_bnb['F1-Score (Macro)']:.3f}", fontsize=11, weight='bold')
    axes[1].set_xlabel("Predicted Topic")
    axes[1].set_ylabel("True Topic")

    # Comparative Metrics Bar Chart
    metrics = ["Accuracy", "F1-Score (Macro)", "F1-Score (Weighted)"]
    x = np.arange(len(metrics))
    width = 0.35

    vals_m = [res_mnb[m] for m in metrics]
    vals_b = [res_bnb[m] for m in metrics]

    b1 = axes[2].bar(x - width/2, vals_m, width, label='Multinomial NB', color='#2b5c8f', alpha=0.9)
    b2 = axes[2].bar(x + width/2, vals_b, width, label='Bernoulli NB', color='#e06666', alpha=0.9)

    axes[2].set_ylabel("Score", fontsize=11)
    axes[2].set_title("Variant Comparison Metrics", fontsize=11, weight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics, fontsize=10)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc='lower right', frameon=True)

    for bar in b1 + b2:
        h = bar.get_height()
        axes[2].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    plot_path_local = os.path.join(os.path.dirname(__file__), "q07_naive_bayes_comparison.png")
    plot_path_global = os.path.join(output_dir, "q07_naive_bayes_comparison.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Visualizations saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_results


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
