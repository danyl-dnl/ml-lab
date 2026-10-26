"""
Master Runner Script: Machine Learning Problem Set (Questions 6 - 15)
Author: Machine Learning Solutions Team
Description:
    Orchestrates and executes solutions for Questions 6 through 15.
    Can run all questions end-to-end or specific selected questions.

Usage:
    python3 run_all.py --all
    python3 run_all.py --questions 6 7 11
"""

import os
import sys
import argparse
import subprocess
import time

QUESTIONS = {
    6: ("Q06_Logistic_Regression_Pima", "question_06_solution.py", "Logistic Regression with Feature Scaling (Pima Diabetes)"),
    7: ("Q07_Naive_Bayes_20Newsgroups", "question_07_solution.py", "Naïve Bayes Text Classification (20 Newsgroups)"),
    8: ("Q08_KNN_Fashion_MNIST", "question_08_solution.py", "K-Nearest Neighbors Image Classification (Fashion MNIST)"),
    9: ("Q09_Decision_Tree_ID3_Online_Retail", "question_09_solution.py", "ID3 Decision Tree Customer Segmentation (Online Retail)"),
    10: ("Q10_Logistic_Regression_vs_Decision_Tree_Adult", "question_10_solution.py", "Logistic Regression vs Decision Trees (Adult Income)"),
    11: ("Q11_Linear_SVM_Iris", "question_11_solution.py", "Linear SVM & Margin Visualization (Iris Dataset)"),
    12: ("Q12_SVM_Kernels_Fashion_MNIST", "question_12_solution.py", "SVM Kernel Comparison: Linear, Poly, RBF (Fashion MNIST)"),
    13: ("Q13_MLP_Wine_Quality", "question_13_solution.py", "Multilayer Perceptron Architectures (Wine Quality)"),
    14: ("Q14_Activation_Functions_MNIST", "question_14_solution.py", "Neural Network Activation Functions: Sigmoid, Tanh, ReLU (MNIST)"),
    15: ("Q15_Hyperparameter_Tuning_Fashion_MNIST", "question_15_solution.py", "Hyperparameter Tuning for Neural Networks (Fashion MNIST)"),
}


def run_question(qn_num: int, repo_root: str) -> bool:
    if qn_num not in QUESTIONS:
        print(f"[!] Invalid question number: {qn_num}. Choose from 6 to 15.")
        return False

    folder, script, description = QUESTIONS[qn_num]
    folder_path = os.path.join(repo_root, folder)
    script_path = os.path.join(folder_path, script)

    if not os.path.exists(script_path):
        print(f"[x] Script not found at: {script_path}")
        return False

    print("\n" + "#" * 80)
    print(f" EXECUTING QUESTION {qn_num}: {description}")
    print(f" Directory: {folder}/")
    print(f" Script:    {script}")
    print("#" * 80)

    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, script], cwd=folder_path, check=True)
        elapsed = time.time() - t0
        print(f"\n[✓] Question {qn_num} completed successfully in {elapsed:.2f} seconds.\n")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"\n[x] Question {qn_num} failed after {elapsed:.2f} seconds with exit code {e.returncode}.\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for Questions 6 to 15.")
    parser.add_argument("--all", action="store_true", help="Execute all questions sequentially (6 through 15).")
    parser.add_argument("--questions", "-q", nargs="+", type=int, help="Specify question numbers to run (e.g. --questions 6 11 14).")

    args = parser.parse_args()
    repo_root = os.path.abspath(os.path.dirname(__file__))

    if not args.all and not args.questions:
        print("No arguments provided. Running all questions by default...")
        target_questions = sorted(QUESTIONS.keys())
    elif args.all:
        target_questions = sorted(QUESTIONS.keys())
    else:
        target_questions = sorted(args.questions)

    print("=" * 80)
    print(" MACHINE LEARNING SOLUTIONS MASTER PIPELINE")
    print(f" Target questions to execute: {target_questions}")
    print("=" * 80)

    start_time = time.time()
    results = {}

    for q in target_questions:
        success = run_question(q, repo_root)
        results[q] = success

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(" PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for q, success in results.items():
        status = "PASSED [✓]" if success else "FAILED [x]"
        desc = QUESTIONS[q][2]
        print(f" Question {q:2d}: {status:<12} | {desc}")
    print("=" * 80)
    print(f" Total Execution Time: {total_time:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
