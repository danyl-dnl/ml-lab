# Question 10: Logistic Regression vs Decision Trees on Adult Income

## Tasks
1. Load and preprocess the Adult Income dataset.
2. Implement both Logistic Regression and Decision Trees.
3. Compare the models based on metrics such as accuracy, precision, recall, and F1-score.
4. Discuss the interpretability of both models and their suitability for the dataset.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 84.77% | 73.54% | 58.75% | 0.6532 |
| **Decision Tree (depth=6)** | **85.58%** | **78.69%** | 56.43% | **0.6572** |

## Discussion

### 1. Model Comparison
- **Decision Tree** achieves slightly higher accuracy (85.58%) and precision (78.69%), producing fewer false positive classifications for high earners.
- **Logistic Regression** captures slightly higher recall (58.75%), finding more high-income individuals overall.

### 2. Interpretability
- **Logistic Regression:** Transparent via linear log-odds coefficients ($\beta_j$). Exponentiating coefficients gives exact **Odds Ratios** ($\exp(\beta_j)$), allowing econometric interpretation of how a unit increase in education or capital gain multiplies the odds of earning $> \$50\text{K}$.
- **Decision Tree:** Uses intuitive human-readable **if-else decision rules**. Naturally captures non-linear feature interactions (such as the interaction between marital status and education) without needing interaction terms.

### 3. Suitability for Dataset
- Decision Trees are well suited when stakeholders require actionable demographic criteria and segmentation rules.
- Logistic Regression is preferred when continuous probability estimates and odds ratios are required for policy and economic reporting.

## How to Run
```bash
python3 question_10_solution.py
```
