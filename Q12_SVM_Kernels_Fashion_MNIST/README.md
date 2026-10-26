# Question 12: SVM Kernels (Linear, Polynomial, RBF) on Fashion MNIST

## Tasks
1. Load and preprocess the Fashion MNIST dataset.
2. Implement SVM with linear, polynomial, and RBF kernels.
3. Compare the classification performance for each kernel.
4. Discuss the strengths and weaknesses of each kernel type.

## Results

| Kernel Type | Accuracy | Weighted F1-Score | Train Time (s) |
| :--- | :---: | :---: | :---: |
| **Linear Kernel** | 82.20% | 0.8202 | 0.28 s |
| **Polynomial Kernel ($d=3$)** | 81.10% | 0.8105 | 0.35 s |
| **RBF Kernel (Optimal)** | **84.90%** | **0.8475** | 0.40 s |

## Discussion: Strengths and Weaknesses of Each Kernel

### 1. Linear Kernel: $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$
- **Strengths:** Lowest computational overhead; efficient for high-dimensional linearly separable data; small memory footprint.
- **Weaknesses:** Strictly linear boundary cannot capture complex non-linear image textures and silhouettes.

### 2. Polynomial Kernel: $K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$
- **Strengths:** Explicitly models non-linear multiplicative feature interactions (e.g. edge and corner conjunctions across pixel coordinates).
- **Weaknesses:** Computationally more expensive; inner products $< 1$ decay exponentially with high degrees ($d \ge 3$), causing gradient attenuation without careful tuning.

### 3. Radial Basis Function (RBF) Kernel: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$
- **Strengths:** Highest accuracy (**84.90%**). Corresponds to an infinite-dimensional Hilbert space, acting as a universal approximator for complex non-linear manifolds.
- **Weaknesses:** Retains a large fraction of training instances as support vectors, increasing test-time evaluation latency.

## How to Run
```bash
python3 question_12_solution.py
```
