# Question 13: Multilayer Perceptron (MLP) on Wine Quality

## Tasks
1. Load and preprocess the Wine Quality dataset.
2. Design and implement an MLP with varying architectures (different hidden layers and neurons).
3. Train and evaluate the network.
4. Discuss the impact of architecture choices on performance.

## Results

| Architecture | Hidden Layers | Test Accuracy | F1-Score | Iterations to Converge |
| :--- | :---: | :---: | :---: | :---: |
| **Shallow (1 Layer)** | `(32,)` | 74.06% | 0.7644 | 157 |
| **Medium (2 Layers)** | `(64, 32)` | **75.62%** | **0.7818** | 165 |
| **Deep (3 Layers)** | `(128, 64, 32)` | 74.69% | 0.7705 | 162 |

## Discussion: Impact of Architecture Choices

1. **Depth and Width (Representation Capacity):**
   - The two-hidden-layer network (`(64, 32)`) achieved the best generalization (**75.62% accuracy**). It provides enough depth to extract non-linear combinations of acidity, sulphates, and alcohol without excessive complexity.
2. **Underfitting vs. Overfitting:**
   - **Shallow Network (`(32,)`):** Has constrained representational capacity, resulting in slightly lower accuracy.
   - **Deep Network (`(128, 64, 32)`):** High parameter count increases the risk of memorizing noisy observations on smaller tabular datasets (~1,600 rows).
3. **Training & Convergence:**
   - Adding layers increases the number of backpropagation matrix multiplications. Appropriate normalization (such as `StandardScaler`) is critical to ensure stable gradient flow across all hidden layers.

## How to Run
```bash
python3 question_13_solution.py
```
