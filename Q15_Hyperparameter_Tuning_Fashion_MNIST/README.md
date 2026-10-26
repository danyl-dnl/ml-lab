# Question 15: Hyperparameter Tuning for Neural Network on Fashion MNIST

## Tasks
1. Load and preprocess the Fashion MNIST dataset.
2. Experiment with different hyperparameters (learning rate, batch size, epochs).
3. Train and evaluate the network.
4. Discuss how hyperparameter choices affect model performance.

## Results

| Experiment | Learning Rate | Batch Size | Epochs | Test Accuracy | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default** | 0.001 | 64 | 10 | **84.80%** | 0.42 s |
| **Higher LR** | 0.010 | 64 | 10 | 83.20% | 0.41 s |
| **Very High LR** | 0.100 | 64 | 10 | 74.50% | 0.41 s |
| **Small Batch** | 0.001 | 32 | 10 | **84.90%** | 0.65 s |
| **Large Batch** | 0.001 | 128 | 10 | 83.90% | **0.32 s** |
| **Fewer Epochs** | 0.001 | 64 | 5 | 83.40% | **0.22 s** |
| **More Epochs** | 0.001 | 64 | 15 | **85.30%** | 0.62 s |

## Discussion: Impact of Hyperparameters

1. **Learning Rate ($\eta$):**
   - **Moderate LR ($0.001$):** Achieves the best stability and highest accuracy ($84.80\%$).
   - **Excessive LR ($0.1$):** Steps are too large, causing the optimizer to oscillate or overshoot the loss basin, dropping accuracy significantly to $74.50\%$.
2. **Batch Size ($B$):**
   - **Small Batch ($32$):** Provides beneficial stochastic gradient noise that prevents trapping in sharp local minima, achieving strong generalization ($84.90\%$).
   - **Large Batch ($128$):** Trains ~2x faster due to matrix vectorization, but provides less stochastic regularization.
3. **Training Epochs ($E$):**
   - **$E = 5$:** Results in underfitting ($83.40\%$).
   - **$E = 15$:** Allows sufficient optimization iterations for gradient descent to settle into lower-loss minima, boosting accuracy to **$85.30\%$**.

## How to Run
```bash
python3 question_15_solution.py
```
