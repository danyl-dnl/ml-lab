# Question 8: K-Nearest Neighbors (KNN) on Fashion MNIST

## Tasks
1. Load and preprocess the Fashion MNIST dataset.
2. Implement KNN for multi-class classification.
3. Experiment with different values of K and evaluate performance.
4. Discuss the impact of different K values on model accuracy and computational efficiency.

## Results

| K | Accuracy | F1-Score | Inference Time (s) |
| :-: | :---: | :---: | :---: |
| **1** | 81.30% | 0.8140 | 0.046 s |
| **3** | 82.20% | 0.8231 | 0.029 s |
| **5** | 82.30% | 0.8235 | 0.028 s |
| **7** | 82.30% | 0.8227 | 0.027 s |
| **9** | **82.50%** | **0.8249** | **0.027 s** |

## Discussion

### 1. Impact on Model Accuracy (Bias-Variance Tradeoff)
- **Small K ($K=1$):** Low bias, high variance. The decision boundary conforms tightly to individual noisy training samples.
- **Moderate K ($K=5$ to $9$):** Achieves the best balance (**82.50% accuracy**), aggregating nearby neighbors to filter out pixel variance.
- **Large K:** High bias. When $K$ becomes excessively large, the neighborhood encompasses points from other classes, causing underfitting.

### 2. Computational Efficiency
- **Training Time:** $O(1)$ (lazy learner). No parametric model is learned during training.
- **Inference Time:** $O(N \cdot d)$ per test sample (where $N = 5,000$ and $d = 784$). Distance calculations must be computed against every stored training sample at prediction time, making inference computationally heavy compared to parametric models.

## How to Run
```bash
python3 question_08_solution.py
```
