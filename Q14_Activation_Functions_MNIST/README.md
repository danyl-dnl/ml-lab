# Question 14: Neural Network Activation Functions on MNIST

## Tasks
1. Load and preprocess the MNIST dataset.
2. Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
3. Train and evaluate each network.
4. Compare training times, convergence, and classification accuracy.

## Results

| Activation Function | Test Accuracy | Final Loss | Training Time (s) | Iterations |
| :--- | :---: | :---: | :---: | :---: |
| **Sigmoid (`logistic`)** | 93.90% | 0.2854 | 2.52 s | 25 |
| **Tanh** | 95.10% | 0.1124 | 2.48 s | 25 |
| **ReLU (Optimal)** | **95.85%** | **0.0632** | **2.21 s** | 25 |

## Discussion: Training Times, Convergence, and Accuracy

1. **Convergence and Loss Minimization:**
   - **ReLU** attained the lowest cross-entropy loss (**0.0632**, over 4x lower than Sigmoid) and highest test accuracy (**95.85%**).
   - Its derivative is constant ($\frac{d}{dz}\text{ReLU}(z) = 1$ for $z > 0$), preventing vanishing gradients and allowing steep descent.
2. **Zero-Centering (Tanh vs. Sigmoid):**
   - **Tanh** outperforms Sigmoid ($95.10\%$ vs $93.90\%$) because its output is symmetric around zero ($[-1, 1]$), eliminating directional gradient oscillations.
3. **The Vanishing Gradient Problem (Sigmoid):**
   - **Sigmoid** outputs are squashed into $(0, 1)$, with a maximum derivative of $\sigma'(0) = 0.25$. Across hidden layers, error gradients are repeatedly multiplied by factors $< 0.25$, causing gradients to vanish and severely hindering weight updates in earlier layers.
4. **Computational Efficiency:**
   - ReLU requires only a conditional threshold comparison ($\max(0, z)$), making it faster than Sigmoid and Tanh which require evaluating exponential functions.

## How to Run
```bash
python3 question_14_solution.py
```
