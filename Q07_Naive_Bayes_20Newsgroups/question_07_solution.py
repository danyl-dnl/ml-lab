"""
Question 7: Naïve Bayes on 20 Newsgroups Dataset
Tasks:
  1. Load and preprocess the 20 Newsgroups dataset.
  2. Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
  3. Evaluate and compare performance using Accuracy and F1-score.
  4. Discuss strengths and weaknesses of each variant for text classification.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def main():
    # 1. Load dataset (4 sample categories for clear, fast text classification)
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    # 2. Text preprocessing and feature extraction
    # Multinomial NB uses word counts (term frequencies)
    count_vec = CountVectorizer(stop_words='english')
    X_train_multi = count_vec.fit_transform(train_data.data)
    X_test_multi = count_vec.transform(test_data.data)

    # Bernoulli NB requires binary indicators (word presence/absence: 1 or 0)
    bern_vec = CountVectorizer(stop_words='english', binary=True)
    X_train_bern = bern_vec.fit_transform(train_data.data)
    X_test_bern = bern_vec.transform(test_data.data)

    y_train = train_data.target
    y_test = test_data.target

    # 3. Implement and train classifiers
    model_multi = MultinomialNB()
    model_multi.fit(X_train_multi, y_train)
    y_pred_multi = model_multi.predict(X_test_multi)

    model_bern = BernoulliNB()
    model_bern.fit(X_train_bern, y_train)
    y_pred_bern = model_bern.predict(X_test_bern)

    # 4. Evaluate Accuracy and F1-Score
    results = pd.DataFrame([
        {
            "Classifier": "Multinomial Naïve Bayes",
            "Accuracy": round(accuracy_score(y_test, y_pred_multi), 4),
            "F1-Score (Macro)": round(f1_score(y_test, y_pred_multi, average='macro'), 4),
            "F1-Score (Weighted)": round(f1_score(y_test, y_pred_multi, average='weighted'), 4),
        },
        {
            "Classifier": "Bernoulli Naïve Bayes",
            "Accuracy": round(accuracy_score(y_test, y_pred_bern), 4),
            "F1-Score (Macro)": round(f1_score(y_test, y_pred_bern, average='macro'), 4),
            "F1-Score (Weighted)": round(f1_score(y_test, y_pred_bern, average='weighted'), 4),
        }
    ]).set_index("Classifier")

    print("=" * 60)
    print("QUESTION 7: NAÏVE BAYES (20 NEWSGROUPS)")
    print("=" * 60)
    print(results)
    print("=" * 60)

    # Discussion
    print("\n--- Strengths and Weaknesses ---")
    print("Multinomial Naïve Bayes:")
    print("  + Strengths: Uses word frequencies (count of keywords), ideal for longer documents.")
    print("  - Weaknesses: Ignores absent words; sensitive to document length imbalances.")
    print("Bernoulli Naïve Bayes:")
    print("  + Strengths: Explicitly models presence and absence of words; robust on short texts.")
    print("  - Weaknesses: Ignores word frequency; penalizes long documents containing many rare words.")


if __name__ == "__main__":
    main()
