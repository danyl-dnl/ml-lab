# Question 6: Logistic Regression on Pima Indians Diabetes

## Tasks
1. Load and preprocess the Pima Indians Diabetes dataset.
2. Implement logistic regression for binary classification.
3. Evaluate model performance with and without feature scaling.
4. Analyze metrics such as accuracy, precision, recall, and F1-score.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Iterations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Without Scaling** | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 69 |
| **With Scaling (`StandardScaler`)** | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 13 |

## Analysis
- **Convergence:** Feature scaling reduced the number of solver iterations from 69 to 13 (over 5x faster convergence).
- **Scale Invariance:** For this dataset, L-BFGS reaches the same global minimum, but scaling ensures regularized coefficients are treated fairly and gradients do not oscillate across differently scaled features (e.g. `Insulin` vs `DiabetesPedigree`).

## How to Run
```bash
python3 question_06_solution.py
```
