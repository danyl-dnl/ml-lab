"""Question 7: Compare two Naive Bayes classifiers for text."""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Load four categories from the 20 Newsgroups dataset.
categories = ["alt.atheism", "comp.graphics", "sci.med", "soc.religion.christian"]
train_data = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
test_data = fetch_20newsgroups(subset="test", categories=categories, remove=("headers", "footers", "quotes"))
y_train, y_test = train_data.target, test_data.target

# 2. Turn words into numbers. Multinomial uses word counts.
count_vectorizer = CountVectorizer(stop_words="english")
X_train_counts = count_vectorizer.fit_transform(train_data.data)
X_test_counts = count_vectorizer.transform(test_data.data)

# Bernoulli uses only whether each word appears (1) or does not appear (0).
binary_vectorizer = CountVectorizer(stop_words="english", binary=True)
X_train_binary = binary_vectorizer.fit_transform(train_data.data)
X_test_binary = binary_vectorizer.transform(test_data.data)

# 3. Train and evaluate both models.
multinomial_model = MultinomialNB()
multinomial_model.fit(X_train_counts, y_train)
multinomial_predictions = multinomial_model.predict(X_test_counts)
bernoulli_model = BernoulliNB()
bernoulli_model.fit(X_train_binary, y_train)
bernoulli_predictions = bernoulli_model.predict(X_test_binary)

# 4. Show accuracy and macro F1-score (averages performance across categories).
results = pd.DataFrame([
    [accuracy_score(y_test, multinomial_predictions), f1_score(y_test, multinomial_predictions, average="macro")],
    [accuracy_score(y_test, bernoulli_predictions), f1_score(y_test, bernoulli_predictions, average="macro")],
], index=["Multinomial (word counts)", "Bernoulli (word presence)"], columns=["Accuracy", "Macro F1-score"]).round(4)
print("QUESTION 7: NAIVE BAYES")
print(results)
print("Multinomial is useful when repeated words matter; it ignores words that are absent.")
print("Bernoulli considers word presence and absence, but ignores how often a present word repeats.")
