# Question 7: Multinomial vs. Bernoulli Naïve Bayes (20 Newsgroups)

## Tasks
1. Load and preprocess the 20 Newsgroups dataset.
2. Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
3. Evaluate and compare the performance of both models using metrics such as accuracy and F1-score.
4. Discuss the strengths and weaknesses of each Naïve Bayes variant for text classification.

## Results

| Classifier | Accuracy | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: |
| **Multinomial Naïve Bayes** | **83.42%** | **0.8298** | **0.8344** |
| **Bernoulli Naïve Bayes** | 68.31% | 0.6795 | 0.6833 |

## Discussion: Strengths and Weaknesses

### Multinomial Naïve Bayes
- **Strengths:** Models the frequency of words (token counts). Highly effective for medium-to-long text documents where repeated occurrences of domain keywords provide strong discriminative signal.
- **Weaknesses:** Completely ignores the non-occurrence (absence) of words; sensitive to document length discrepancies without length normalization.

### Bernoulli Naïve Bayes
- **Strengths:** Operates on binary word occurrence ($1$ if word appears, $0$ otherwise). Explicitly multiplies the probability of absent words $(1 - P(w|c))$, making it well suited for short texts, sentiment classification, and spam detection.
- **Weaknesses:** Cannot capture the emphasis of repeated words (treating 1 mention the same as 10 mentions); poorly handles longer documents where common vocabulary occurs in almost all classes.

## How to Run
```bash
python3 question_07_solution.py
```
