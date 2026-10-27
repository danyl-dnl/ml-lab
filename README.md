# Machine Learning Repository: Questions 6 – 15

Clean, concise, and focused implementations for Machine Learning Questions 6 through 15. Every solution directly answers only the requested tasks in a straightforward manner.

---

## Repository Structure

```text
ml-repo/
├── README.md                                           # Master repository documentation (this file)
├── requirements.txt                                    # Project dependencies
├── run_all.py                                          # Master script to run all or selected questions
├── .gitignore                                          # Git ignore configuration
├── data/                                               # Cached dataset storage (auto-downloaded)
├── outputs/                                            # Visualizations (Decision Tree & Margin plots)
│   ├── q09_decision_tree_customer_segments.png
│   └── q11_linear_svm_iris_margin.png
├── Q06_Logistic_Regression_Pima/
│   ├── question_06_solution.py                         # Logistic Regression with/without scaling
│   └── README.md                                       # Results & scaling convergence analysis
├── Q07_Naive_Bayes_20Newsgroups/
│   ├── question_07_solution.py                         # Multinomial vs Bernoulli Naïve Bayes
│   └── README.md                                       # Comparison & variant strengths/weaknesses
├── Q08_KNN_Fashion_MNIST/
│   ├── question_08_solution.py                         # KNN image classification with varying K
│   └── README.md                                       # Impact of K & computational complexity
├── Q09_Decision_Tree_ID3_Online_Retail/
│   ├── question_09_solution.py                         # ID3 Decision Tree customer segmentation
│   ├── q09_decision_tree_customer_segments.png         # Tree structure visualization
│   └── README.md                                       # Feature importance & customer behavior
├── Q10_Logistic_Regression_vs_Decision_Tree_Adult/
│   ├── question_10_solution.py                         # Logistic Regression vs Decision Trees
│   └── README.md                                       # Metric comparison & interpretability
├── Q11_Linear_SVM_Iris/
│   ├── question_11_solution.py                         # Linear SVM binary classification
│   ├── q11_linear_svm_iris_margin.png                  # Decision boundary & margin visualization
│   └── README.md                                       # Margin concept & support vectors
├── Q12_SVM_Kernels_Fashion_MNIST/
│   ├── question_12_solution.py                         # Linear, Polynomial, and RBF kernels
│   └── README.md                                       # Performance & kernel pros/cons
├── Q13_MLP_Wine_Quality/
│   ├── question_13_solution.py                         # MLP with varying hidden layer architectures
│   └── README.md                                       # Architecture depth & width analysis
├── Q14_Activation_Functions_MNIST/
│   ├── question_14_solution.py                         # Sigmoid, Tanh, and ReLU comparison
│   └── README.md                                       # Vanishing gradients & training dynamics
└── Q15_Hyperparameter_Tuning_Fashion_MNIST/
    ├── question_15_solution.py                         # Grid tuning: LR, Batch Size, Epochs
    └── README.md                                       # Hyperparameter impact analysis
```

---

## Summary of Solutions & Questions

| Question | Topic | Dataset | Summary of Tasks Solved |
| :---: | :--- | :--- | :--- |
| **Q06** | **Logistic Regression** | Pima Indians Diabetes | Evaluates binary logistic regression with and without `StandardScaler`. Compares accuracy, precision, recall, and F1-score, showing faster convergence with scaling. |
| **Q07** | **Naïve Bayes** | 20 Newsgroups | Compares Multinomial NB (word frequencies) with Bernoulli NB (word presence/absence) using Accuracy and F1-score, highlighting strengths and weaknesses. |
| **Q08** | **K-Nearest Neighbors (KNN)** | Fashion MNIST | Tests neighbor values $K \in [1, 3, 5, 7, 9]$. Analyzes accuracy trends and computational complexity (lazy learner inference latency). |
| **Q09** | **Decision Tree (ID3)** | Online Retail | Computes Recency, Frequency, Monetary (RFM) customer features. Trains an ID3 tree (entropy), visualizes the tree structure, and interprets feature importances. |
| **Q10** | **Logistic Regression vs. Decision Tree** | Adult Income | Trains and evaluates both models for predicting $>50\text{K}$ income. Discusses interpretability (linear odds ratios vs if-else tree rules) and suitability. |
| **Q11** | **Linear SVM** | Iris | Trains a Linear SVM to classify Setosa vs Non-Setosa in 2D feature space. Visualizes the decision boundary, margin lines ($w^Tx+b=\pm 1$), and support vectors. |
| **Q12** | **SVM Kernels** | Fashion MNIST | Compares Linear, Polynomial ($d=3$), and RBF kernels on classification accuracy, F1-score, and training latency. Discusses the advantages/disadvantages of each kernel. |
| **Q13** | **Multilayer Perceptron (MLP)** | Wine Quality | Implements 1-layer, 2-layer, and 3-layer MLP architectures. Compares test accuracy and discusses the impact of depth and width on underfitting vs overfitting. |
| **Q14** | **Neural Activation Functions** | MNIST | Compares neural networks using Sigmoid, Tanh, and ReLU activations. Compares training time, convergence, and accuracy, explaining vanishing gradients. |
| **Q15** | **Hyperparameter Tuning** | Fashion MNIST | Tunes Learning Rate ($0.001, 0.01, 0.1$), Batch Size ($32, 64, 128$), and Epochs ($5, 10, 15$). Analyzes how each hyperparameter impacts stability and accuracy. |

---

## How to Run

### 1. Install Python

Use Python 3.10 or 3.11. Check that Python and pip are available:

```bash
python3 --version
python3 -m pip --version
```

### 2. Open a terminal in the repository folder

The folder should contain `run_all.py` and `requirements.txt`. For example:

```bash
cd ml-repo
```

### 3. Create and activate a virtual environment

A virtual environment keeps this repository's packages separate from other Python projects.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the packages used by the solutions, including pandas, scikit-learn, matplotlib, PyTorch, and torchvision.

### 5. Run a question

Run one question from the repository folder, for example Question 6:

```bash
python Q06_Logistic_Regression_Pima/question_06_solution.py
```

Run a few selected questions:

```bash
python run_all.py --questions 6 11 14
```

Run all questions:

```bash
python run_all.py --all
```

The first run of Questions 8, 12, 14, and 15 may download Fashion-MNIST or MNIST. Question 7 may download the 20 Newsgroups text dataset. Those questions need an internet connection the first time; downloaded copies are cached under `data/`.

The Pima, Adult, Wine Quality, and Online Retail data files are already included in `data/`. Running every question takes longer than running one because several models are trained for comparison.

When finished, leave the virtual environment by running `deactivate`.
