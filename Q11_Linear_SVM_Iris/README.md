# Question 11: Linear Support Vector Machine on Iris Dataset

## Tasks
1. Load and preprocess the Iris dataset.
2. Implement a Linear SVM for binary classification (Setosa vs. Non-Setosa).
3. Visualize the decision boundary and margin.
4. Discuss the concept of the margin and how it influences classification.

## Results
- **Classification Accuracy:** **100.00%** (Linearly Separable)
- **Support Vectors:** **3 points**
- **Margin Width ($2 / \|\mathbf{w}\|$):** **1.625 cm**

## Decision Boundary & Margin Visualization
The plot showing the separating hyperplane, margins, and highlighted support vectors is saved to:
`q11_linear_svm_iris_margin.png`

## Discussion: Concept of the Margin & Generalization
1. **Mathematical Definition:**
   - The separating hyperplane is given by $\mathbf{w}^T \mathbf{x} + b = 0$.
   - The canonical margin lines are $\mathbf{w}^T \mathbf{x} + b = \pm 1$.
   - The geometric distance between the margins is:
     $$\text{Margin Width} = \frac{2}{\|\mathbf{w}\|_2}$$
2. **Support Vectors:**
   - The decision boundary is determined solely by the **support vectors**—the data points that lie directly on the margin boundaries ($\alpha_i > 0$).
   - All other points could be moved or removed without altering the optimal hyperplane.
3. **Influence on Classification:**
   - Maximizing the margin corresponds to structural risk minimization, reducing the classifier's Vapnik-Chervonenkis (VC) dimension and minimizing overfitting on unseen test data.

## How to Run
```bash
python3 question_11_solution.py
```
