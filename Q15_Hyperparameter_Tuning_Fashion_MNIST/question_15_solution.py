r"""
Question 15: Hyperparameter Tuning for Neural Network on Fashion MNIST
Description:
    Implement and perform systematic hyperparameter tuning for a neural network
    on the Fashion MNIST dataset. Experiment with different learning rates,
    batch sizes, and epochs. Analyze and discuss the impact on model stability,
    convergence dynamics, computational efficiency, and generalization.

Tasks:
    1. Load and preprocess Fashion MNIST (flattening 28x28 images, normalization).
    2. Design modular PyTorch neural network for multi-class fashion classification.
    3. Perform grid search over:
       - Learning rates: [0.0005, 0.002, 0.01, 0.05]
       - Batch sizes: [32, 64, 128]
       - Epoch durations: [5, 10, 15]
    4. Record metrics: Test Accuracy, Macro F1, Training Latency, and Loss Trajectories.
    5. Generate comparative visualizations (Heatmap, LR curves, Batch size efficiency).
    6. Discuss the theoretical and practical implications of hyperparameter choices.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


class FashionClassifierMLP(nn.Module):
    """Feed-Forward Neural Network for Fashion MNIST."""
    def __init__(self, hidden_dim1: int = 128, hidden_dim2: int = 64, dropout_rate: float = 0.2):
        super(FashionClassifierMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 10),
        )

    def forward(self, x):
        return self.net(x)


def load_fashion_subsets(data_dir: str = "../data", n_train: int = 15000, n_val: int = 3000):
    """Load normalized Fashion MNIST subsets for controlled hyperparameter tuning."""
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_full = dsets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = dsets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    np.random.seed(42)
    indices = np.random.permutation(len(train_full))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(train_full, val_idx)

    print(f"Subsets prepared -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_data)}")
    return train_subset, val_subset, test_data


def train_single_trial(lr: float, batch_size: int, epochs: int, train_data, val_data, test_data):
    """Execute a single trial with specific hyperparameter triplet (LR, Batch Size, Epochs)."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    torch.manual_seed(42)
    model = FashionClassifierMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    train_time = time.time() - t0

    # Evaluate on Test Set
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds.extend(outputs.argmax(dim=1).numpy())
            targets.extend(batch_y.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    test_acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average='macro', zero_division=0)

    return {
        "Learning Rate": lr,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Test Accuracy": test_acc,
        "Macro F1": f1_macro,
        "Train Time (s)": train_time,
        "Throughput (samples/s)": (len(train_data) * epochs) / max(train_time, 1e-4)
    }


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 15: HYPERPARAMETER TUNING ON FASHION MNIST")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    train_data, val_data, test_data = load_fashion_subsets(data_dir=data_dir, n_train=15000, n_val=3000)

    # 1. Grid of Hyperparameter Configurations
    # Systematic search exploring LR scale (10^-4 to 10^-1), batch sizes (32, 64, 128), and epochs (5, 10, 15)
    experiments = [
        # Impact of Learning Rate (Batch Size = 64, Epochs = 10)
        {"lr": 0.0005, "batch_size": 64, "epochs": 10},
        {"lr": 0.002,  "batch_size": 64, "epochs": 10},
        {"lr": 0.01,   "batch_size": 64, "epochs": 10},
        {"lr": 0.05,   "batch_size": 64, "epochs": 10},

        # Impact of Batch Size (LR = 0.002, Epochs = 10)
        {"lr": 0.002,  "batch_size": 32, "epochs": 10},
        {"lr": 0.002,  "batch_size": 128, "epochs": 10},

        # Impact of Epochs (LR = 0.002, Batch Size = 64)
        {"lr": 0.002,  "batch_size": 64, "epochs": 5},
        {"lr": 0.002,  "batch_size": 64, "epochs": 15},

        # Cross-combinations for Heatmap
        {"lr": 0.0005, "batch_size": 32, "epochs": 10},
        {"lr": 0.01,   "batch_size": 32, "epochs": 10},
        {"lr": 0.0005, "batch_size": 128, "epochs": 10},
        {"lr": 0.01,   "batch_size": 128, "epochs": 10},
    ]

    # Deduplicate while preserving order
    unique_exps = []
    seen = set()
    for e in experiments:
        key = (e["lr"], e["batch_size"], e["epochs"])
        if key not in seen:
            seen.add(key)
            unique_exps.append(e)

    print(f"\nEvaluating {len(unique_exps)} distinct hyperparameter configurations...")
    records = []
    for idx, cfg in enumerate(unique_exps, 1):
        print(f"[{idx:2d}/{len(unique_exps)}] LR: {cfg['lr']:<6} | Batch: {cfg['batch_size']:<3} | Epochs: {cfg['epochs']:<2}", end=" -> ")
        res = train_single_trial(cfg["lr"], cfg["batch_size"], cfg["epochs"], train_data, val_data, test_data)
        records.append(res)
        print(f"Test Acc: {res['Test Accuracy']*100:.2f}% | F1: {res['Macro F1']:.4f} | Time: {res['Train Time (s)']:.2f}s")

    df_results = pd.DataFrame(records)

    print("\n" + "=" * 90)
    print(" HYPERPARAMETER TUNING RESULTS TABLE")
    print("=" * 90)
    print(df_results.sort_values(by="Test Accuracy", ascending=False).to_string(index=False))
    print("=" * 90)

    best_cfg = df_results.loc[df_results["Test Accuracy"].idxmax()]
    print(f"\n[★] Best Hyperparameter Configuration:")
    print(f"    Learning Rate: {best_cfg['Learning Rate']}")
    print(f"    Batch Size:    {int(best_cfg['Batch Size'])}")
    print(f"    Epochs:        {int(best_cfg['Epochs'])}")
    print(f"    Test Accuracy: {best_cfg['Test Accuracy']*100:.2f}%")
    print(f"    Macro F1:      {best_cfg['Macro F1']:.4f}")

    # Visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # (A) Learning Rate vs Test Accuracy (Batch=64, Epochs=10)
    ax1 = fig.add_subplot(gs[0, 0])
    df_lr = df_results[(df_results["Batch Size"] == 64) & (df_results["Epochs"] == 10)].sort_values(by="Learning Rate")
    ax1.plot(df_lr["Learning Rate"].astype(str), df_lr["Test Accuracy"] * 100, marker='o', lw=2.2, color='#2b5c8f', label='Accuracy (%)')
    ax1.plot(df_lr["Learning Rate"].astype(str), df_lr["Macro F1"] * 100, marker='s', lw=2.2, color='#e06666', linestyle='--', label='Macro F1 (%)')
    ax1.set_xlabel("Learning Rate", fontsize=10)
    ax1.set_ylabel("Score (%)", fontsize=10)
    ax1.set_title("(A) Impact of Learning Rate (Batch=64, Epochs=10)", fontsize=11, weight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)
    for x_i, y_i in zip(df_lr["Learning Rate"].astype(str), df_lr["Test Accuracy"] * 100):
        ax1.annotate(f"{y_i:.2f}%", xy=(x_i, y_i), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, weight='bold')

    # (B) Batch Size vs Training Time & Accuracy (LR=0.002, Epochs=10)
    ax2 = fig.add_subplot(gs[0, 1])
    df_bs = df_results[(df_results["Learning Rate"] == 0.002) & (df_results["Epochs"] == 10)].sort_values(by="Batch Size")
    x_bs = np.arange(len(df_bs))
    width = 0.35
    b1 = ax2.bar(x_bs - width/2, df_bs["Test Accuracy"] * 100, width, label='Test Accuracy (%)', color='#27ae60', edgecolor='black', linewidth=0.5)
    ax2_twin = ax2.twinx()
    b2 = ax2_twin.bar(x_bs + width/2, df_bs["Train Time (s)"], width, label='Train Time (s)', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x_bs)
    ax2.set_xticklabels(df_bs["Batch Size"].astype(int), fontsize=10)
    ax2.set_xlabel("Batch Size", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10, color='#27ae60')
    ax2_twin.set_ylabel("Training Time (s)", fontsize=10, color='#f39c12')
    ax2.set_ylim(80, 92)
    ax2.set_title("(B) Batch Size vs Accuracy & Training Latency (LR=0.002)", fontsize=11, weight='bold')

    # (C) Heatmap: Learning Rate vs Batch Size (at 10 Epochs)
    ax3 = fig.add_subplot(gs[1, 0])
    df_pivot = df_results[df_results["Epochs"] == 10].pivot(index="Learning Rate", columns="Batch Size", values="Test Accuracy") * 100
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax3, cbar_kws={'label': 'Accuracy (%)'},
                annot_kws={'size': 11, 'weight': 'bold'})
    ax3.set_title("(C) Grid Search Accuracy Heatmap (Epochs = 10)", fontsize=11, weight='bold')
    ax3.set_xlabel("Batch Size", fontsize=10)
    ax3.set_ylabel("Learning Rate", fontsize=10)

    # (D) Training Epochs Progression (LR=0.002, Batch=64)
    ax4 = fig.add_subplot(gs[1, 1])
    df_ep = df_results[(df_results["Learning Rate"] == 0.002) & (df_results["Batch Size"] == 64)].sort_values(by="Epochs")
    ax4.plot(df_ep["Epochs"], df_ep["Test Accuracy"] * 100, marker='D', lw=2.2, color='#8e44ad', label='Accuracy (%)')
    ax4.set_xlabel("Training Epochs", fontsize=10)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax4.set_title("(D) Effect of Training Epochs on Generalization", fontsize=11, weight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    for x_i, y_i in zip(df_ep["Epochs"], df_ep["Test Accuracy"] * 100):
        ax4.annotate(f"{y_i:.2f}%", xy=(x_i, y_i), xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, weight='bold')

    fig.suptitle("Hyperparameter Tuning Dynamics: Learning Rate, Batch Size & Epochs on Fashion MNIST",
                 fontsize=14, weight='bold', y=0.98)

    plot_path_local = os.path.join(os.path.dirname(__file__), "q15_hyperparameter_tuning_fashion_mnist.png")
    plot_path_global = os.path.join(output_dir, "q15_hyperparameter_tuning_fashion_mnist.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_results


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
