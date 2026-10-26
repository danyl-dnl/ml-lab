# Machine Learning Repository: Questions 6 – 15

Comprehensive, modular, and production-ready solutions for Machine Learning Questions 6 through 15. Each solution includes complete data preprocessing, algorithm implementation, evaluation metrics, comparative analysis, high-resolution visualizations, and theoretical documentation.

---

## Repository Structure

```text
ml-repo/
├── README.md                                           # Master repository documentation (this file)
├── requirements.txt                                    # Complete project dependencies
├── run_all.py                                          # Master CLI runner script for all questions
├── .gitignore                                          # Standard Python/Git ignore rules
├── data/                                               # Auto-cached local dataset directory
├── outputs/                                            # Centralized store for all generated figures
│   ├── q06_feature_scaling_comparison.png
│   ├── q07_naive_bayes_comparison.png
│   ├── q08_knn_k_value_analysis.png
│   ├── q09_decision_tree_customer_segments.png
│   ├── q10_logreg_vs_decision_tree.png
│   ├── q11_linear_svm_iris_margin.png
│   ├── q12_svm_kernel_comparison.png
│   ├── q13_mlp_wine_architectures.png
│   ├── q14_activation_functions_mnist.png
│   └── q15_hyperparameter_tuning_fashion_mnist.png
├── Q06_Logistic_Regression_Pima/
│   ├── question_06_solution.py                         # Standalone executable solution
│   ├── q06_feature_scaling_comparison.png              # Generated metric & ROC comparison
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q07_Naive_Bayes_20Newsgroups/
│   ├── question_07_solution.py                         # Standalone executable solution
│   ├── q07_naive_bayes_comparison.png                  # Multinomial vs Bernoulli comparison
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q08_KNN_Fashion_MNIST/
│   ├── question_08_solution.py                         # Standalone executable solution
│   ├── q08_knn_k_value_analysis.png                    # Accuracy, latency & confusion matrix
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q09_Decision_Tree_ID3_Online_Retail/
│   ├── question_09_solution.py                         # Standalone executable solution
│   ├── q09_decision_tree_customer_segments.png         # ID3 Tree diagram & RFM importances
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q10_Logistic_Regression_vs_Decision_Tree_Adult/
│   ├── question_10_solution.py                         # Standalone executable solution
│   ├── q10_logreg_vs_decision_tree.png                 # Odds ratios & Gini tree importances
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q11_Linear_SVM_Iris/
│   ├── question_11_solution.py                         # Standalone executable solution
│   ├── q11_linear_svm_iris_margin.png                  # Maximum margin & support vectors
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q12_SVM_Kernels_Fashion_MNIST/
│   ├── question_12_solution.py                         # Standalone executable solution
│   ├── q12_svm_kernel_comparison.png                   # Linear vs Polynomial vs RBF benchmark
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q13_MLP_Wine_Quality/
│   ├── question_13_solution.py                         # Standalone executable solution
│   ├── q13_mlp_wine_architectures.png                  # Loss curves & capacity tradeoffs
│   └── README.md                                       # Detailed theoretical & empirical report
├── Q14_Activation_Functions_MNIST/
│   ├── question_14_solution.py                         # Standalone executable solution
│   ├── q14_activation_functions_mnist.png              # Sigmoid vs Tanh vs ReLU dynamics
│   └── README.md                                       # Detailed theoretical & empirical report
└── Q15_Hyperparameter_Tuning_Fashion_MNIST/
    ├── question_15_solution.py                         # Standalone executable solution
    ├── q15_hyperparameter_tuning_fashion_mnist.png     # LR, Batch Size & Epoch heatmaps
    └── README.md                                       # Detailed theoretical & empirical report
```

---

## Solutions Executive Summary

| Question | Algorithm & Domain | Dataset | Key Experimental Results | Documentation & Code |
| :---: | :--- | :--- | :--- | :--- |
| **Q06** | **Logistic Regression** (Feature Scaling Study) | Pima Indians Diabetes | Feature scaling (`StandardScaler`) improved convergence speed by **22x** (from 175 to 8 iterations) and increased Precision from 58.7% to 60.0%. | [Q06 Folder](file:///Users/danyl/ml-repo/Q06_Logistic_Regression_Pima/README.md) \| [Code](file:///Users/danyl/ml-repo/Q06_Logistic_Regression_Pima/question_06_solution.py) |
| **Q07** | **Naïve Bayes** (Multinomial vs. Bernoulli) | 20 Newsgroups | Multinomial NB substantially outperformed Bernoulli NB (**82.89% vs 67.44% accuracy**, **0.8240 vs 0.6707 Macro F1**) due to capturing repeated domain keyword counts. | [Q07 Folder](file:///Users/danyl/ml-repo/Q07_Naive_Bayes_20Newsgroups/README.md) \| [Code](file:///Users/danyl/ml-repo/Q07_Naive_Bayes_20Newsgroups/question_07_solution.py) |
| **Q08** | **K-Nearest Neighbors** ($K$ Experimentation) | Fashion MNIST | Optimal performance achieved at **$K = 9$ (79.50% accuracy)**. Demonstrates bias-variance tradeoff and test-time latency of lazy learning ($O(N \cdot d)$ per query). | [Q08 Folder](file:///Users/danyl/ml-repo/Q08_KNN_Fashion_MNIST/README.md) \| [Code](file:///Users/danyl/ml-repo/Q08_KNN_Fashion_MNIST/question_08_solution.py) |
| **Q09** | **Decision Tree (ID3)** (Entropy & Info Gain) | Online Retail (UCI) | Engineered RFM features; ID3 tree identified **Purchase Frequency (71.08% importance)** and **Recency (22.51%)** as primary customer segmentation drivers (**88.33% accuracy**). | [Q09 Folder](file:///Users/danyl/ml-repo/Q09_Decision_Tree_ID3_Online_Retail/README.md) \| [Code](file:///Users/danyl/ml-repo/Q09_Decision_Tree_ID3_Online_Retail/question_09_solution.py) |
| **Q10** | **Logistic Regression vs Decision Trees** | Adult Income (Census) | Both achieved high AUC (**0.9077** for LogReg vs **0.9060** for Tree). Decision Tree offers higher precision (80.17%), while LogReg provides higher recall (61.29%) and linear odds ratios. | [Q10 Folder](file:///Users/danyl/ml-repo/Q10_Logistic_Regression_vs_Decision_Tree_Adult/README.md) \| [Code](file:///Users/danyl/ml-repo/Q10_Logistic_Regression_vs_Decision_Tree_Adult/question_10_solution.py) |
| **Q11** | **Linear SVM** (Margin & Boundary Visuals) | Iris Dataset | Visualized maximum margin hyperplanes and support vectors. Margin width $\gamma = \frac{2}{\|\mathbf{w}\|}$ was 1.0326 (hard margin, 2 SVs) and 0.9333 (soft margin, 19 SVs). | [Q11 Folder](file:///Users/danyl/ml-repo/Q11_Linear_SVM_Iris/README.md) \| [Code](file:///Users/danyl/ml-repo/Q11_Linear_SVM_Iris/question_11_solution.py) |
| **Q12** | **SVM Kernels** (Linear, Polynomial, RBF) | Fashion MNIST | **RBF Kernel achieved the highest accuracy (84.70%)**, outperforming Linear (81.70%) and Polynomial (80.70%), proving the necessity of infinite-dimensional RKHS on image manifolds. | [Q12 Folder](file:///Users/danyl/ml-repo/Q12_SVM_Kernels_Fashion_MNIST/README.md) \| [Code](file:///Users/danyl/ml-repo/Q12_SVM_Kernels_Fashion_MNIST/question_12_solution.py) |
| **Q13** | **Multilayer Perceptron (MLP)** (Architectures) | Wine Quality (UCI) | Evaluated 4 PyTorch architectures; **Medium `[64, 32]` achieved the best generalization (62.36% test accuracy, 0.6096 F1)**, outperforming shallow (underfitting) and deep (overfitting). | [Q13 Folder](file:///Users/danyl/ml-repo/Q13_MLP_Wine_Quality/README.md) \| [Code](file:///Users/danyl/ml-repo/Q13_MLP_Wine_Quality/question_13_solution.py) |
| **Q14** | **Neural Activations** (Sigmoid, Tanh, ReLU) | MNIST Handwritten Digits | **ReLU achieved 97.02% test accuracy** and converged to $0.0003$ loss (140x lower than Sigmoid). Tanh outperformed Sigmoid (96.48% vs 95.85%) due to zero-centered gradient flow. | [Q14 Folder](file:///Users/danyl/ml-repo/Q14_Activation_Functions_MNIST/README.md) \| [Code](file:///Users/danyl/ml-repo/Q14_Activation_Functions_MNIST/question_14_solution.py) |
| **Q15** | **Hyperparameter Tuning** (LR, Batch, Epochs) | Fashion MNIST | Grid search identified optimal configuration: **$\text{LR} = 0.002$, $\text{Batch Size} = 64$, $\text{Epochs} = 15$ (86.48% accuracy)**. Large batch sizes boosted throughput by +56.7%. | [Q15 Folder](file:///Users/danyl/ml-repo/Q15_Hyperparameter_Tuning_Fashion_MNIST/README.md) \| [Code](file:///Users/danyl/ml-repo/Q15_Hyperparameter_Tuning_Fashion_MNIST/question_15_solution.py) |

---

## Installation & Setup

### 1. Clone Repository & Install Dependencies
```bash
# Clone the repository
git clone <your-repo-url>
cd ml-repo

# Install all required packages
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.9
- `torch` & `torchvision` (Deep learning backends)
- `scikit-learn` & `scipy` (Machine learning & statistical models)
- `pandas` & `numpy` (Tabular data manipulation & linear algebra)
- `matplotlib` & `seaborn` (Publication-grade scientific visualizations)
- `openpyxl` & `ucimlrepo` (Dataset parsers)

---

## Running the Solutions

### Option A: Master Runner (`run_all.py`)
You can execute all solutions end-to-end or run specific questions using the CLI runner:

```bash
# Execute all solutions sequentially (Questions 6 through 15)
python3 run_all.py --all

# Execute specific questions (e.g. Questions 6, 11, and 14)
python3 run_all.py --questions 6 11 14
```

### Option B: Running Individual Solutions
Each folder contains an autonomous, self-contained Python script:

```bash
# Question 6: Logistic Regression (Pima Diabetes)
cd Q06_Logistic_Regression_Pima
python3 question_06_solution.py

# Question 7: Naïve Bayes (20 Newsgroups)
cd ../Q07_Naive_Bayes_20Newsgroups
python3 question_07_solution.py

# Question 8: KNN (Fashion MNIST)
cd ../Q08_KNN_Fashion_MNIST
python3 question_08_solution.py

# Question 9: Decision Tree ID3 (Online Retail)
cd ../Q09_Decision_Tree_ID3_Online_Retail
python3 question_09_solution.py

# Question 10: Logistic Regression vs Decision Trees (Adult Income)
cd ../Q10_Logistic_Regression_vs_Decision_Tree_Adult
python3 question_10_solution.py

# Question 11: Linear SVM (Iris Dataset)
cd ../Q11_Linear_SVM_Iris
python3 question_11_solution.py

# Question 12: SVM Kernels (Fashion MNIST)
cd ../Q12_SVM_Kernels_Fashion_MNIST
python3 question_12_solution.py

# Question 13: Multilayer Perceptron Architectures (Wine Quality)
cd ../Q13_MLP_Wine_Quality
python3 question_13_solution.py

# Question 14: Activation Functions (MNIST)
cd ../Q14_Activation_Functions_MNIST
python3 question_14_solution.py

# Question 15: Hyperparameter Tuning (Fashion MNIST)
cd ../Q15_Hyperparameter_Tuning_Fashion_MNIST
python3 question_15_solution.py
```

---

## Key Highlights & Scientific Best Practices
1. **No Data Leakage:** All data preprocessing (imputations, standard scalers, one-hot encoders) is strictly fitted on training splits only and transformed onto test splits.
2. **Reproducibility:** Seed values (`random_state=42`, `torch.manual_seed(42)`) are pinned across all experiments to guarantee identical, deterministic results.
3. **Automated Data Fetching & Caching:** All datasets are automatically downloaded from authenticated repositories (UCI, PyTorch torchvision, Scikit-learn, GitHub) and cached in `data/`.
4. **Rich Visualizations:** Every script generates 300 DPI publication-grade figures saved simultaneously to the question's local folder and the centralized `outputs/` folder.
