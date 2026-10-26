r"""
Question 13: Multilayer Feed-Forward Neural Network (MLP) on Wine Quality Dataset
Description:
    Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine
    Quality dataset. Experiment with different numbers of hidden layers and
    neurons, and analyze how architectural choices affect learning dynamics,
    generalization, and classification performance.

Tasks:
    1. Load and preprocess the Wine Quality dataset (StandardScaler, train/val/test splits).
    2. Design and implement 4 MLP architectures with varying depth and width:
       - Shallow: [32]
       - Medium: [64, 32]
       - Deep: [128, 64, 32]
       - Bottleneck: [64, 16, 64]
    3. Train each architecture using PyTorch with Adam optimizer and Cross-Entropy loss.
    4. Track training loss and validation accuracy across epochs.
    5. Evaluate on held-out test data: Accuracy, Precision, Recall, F1-Score.
    6. Discuss the impact of architectural depth and width on performance.
"""

import os
import urllib.request
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_wine_dataset(data_dir: str = "../data") -> pd.DataFrame:
    """Load Wine Quality dataset (both Red and White wines for robust modeling)."""
    os.makedirs(data_dir, exist_ok=True)
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    red_path = os.path.join(data_dir, "winequality-red.csv")
    white_path = os.path.join(data_dir, "winequality-white.csv")

    if not os.path.exists(red_path):
        urllib.request.urlretrieve(red_url, red_path)
    if not os.path.exists(white_path):
        urllib.request.urlretrieve(white_url, white_path)

    df_red = pd.read_csv(red_path, sep=';')
    df_red['type'] = 0  # Red wine indicator

    df_white = pd.read_csv(white_path, sep=';')
    df_white['type'] = 1  # White wine indicator

    df = pd.concat([df_red, df_white], ignore_index=True)
    print(f"Total wine samples loaded: {len(df)} ({len(df_red)} red, {len(df_white)} white).")
    return df


def preprocess_wine_data(df: pd.DataFrame):
    """
    Standardize features and define 3 quality tiers:
    - Low: quality <= 5
    - Medium: quality == 6
    - High: quality >= 7
    """
    X = df.drop(columns=['quality']).values

    def assign_quality_tier(q):
        if q <= 5:
            return 0  # Low
        elif q == 6:
            return 1  # Medium
        else:
            return 2  # High

    y = np.array([assign_quality_tier(q) for q in df['quality'].values])

    # 70% train, 15% validation, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution in test: {dict(pd.Series(y_test).value_counts())}")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, X.shape[1]


# Dynamic PyTorch MLP Architecture
class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, num_classes: int = 3, dropout_rate: float = 0.2):
        super(ConfigurableMLP, self).__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_and_evaluate_architecture(name: str, hidden_layers: list, input_dim: int,
                                    train_loader, val_loader, X_test_tensor, y_test,
                                    epochs: int = 40, lr: float = 0.003):
    """Train MLP model, record epoch histories, and evaluate on test set."""
    model = ConfigurableMLP(input_dim=input_dim, hidden_layers=hidden_layers, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_losses = []
    val_accuracies = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_y)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()

        val_acc = correct / len(val_loader.dataset)
        val_accuracies.append(val_acc)

    train_time = time.time() - t0

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        y_pred = test_logits.argmax(dim=1).numpy()

    test_acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"[{name}] Params: {total_params:,} | Test Acc: {test_acc*100:.2f}% | Macro F1: {f1_macro:.4f} | Final Val Acc: {val_accuracies[-1]*100:.2f}%")

    metrics = {
        "Architecture": name,
        "Hidden Layers": str(hidden_layers),
        "Parameters": total_params,
        "Test Accuracy": test_acc,
        "Precision (W)": prec_weighted,
        "Recall (W)": rec_weighted,
        "F1-Score (Macro)": f1_macro,
        "Train Time (s)": train_time,
    }

    history = {
        "train_loss": train_losses,
        "val_acc": val_accuracies,
    }

    return metrics, history, y_pred


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 13: MLP ARCHITECTURAL COMPARISON (WINE QUALITY)")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    df = load_wine_dataset(data_dir=data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, in_dim = preprocess_wine_data(df)

    # PyTorch DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Architectural configurations
    architectures = [
        ("Shallow [32]", [32]),
        ("Medium [64, 32]", [64, 32]),
        ("Deep [128, 64, 32]", [128, 64, 32]),
        ("Bottleneck [64, 16, 64]", [64, 16, 64]),
    ]

    all_metrics = []
    histories = {}
    test_preds = {}

    epochs = 40
    print("\n--- Training Architectures ---")
    for name, hidden in architectures:
        m, h, p = train_and_evaluate_architecture(
            name, hidden, in_dim, train_loader, val_loader, X_test_tensor, y_test, epochs=epochs
        )
        all_metrics.append(m)
        histories[name] = h
        test_preds[name] = p

    df_metrics = pd.DataFrame(all_metrics)

    print("\n" + "=" * 90)
    print(" MULTILAYER PERCEPTRON ARCHITECTURAL BENCHMARK")
    print("=" * 90)
    print(df_metrics.to_string(index=False))
    print("=" * 90)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Multilayer Perceptron (MLP) Architectural Dynamics on Wine Quality",
                 fontsize=14, weight='bold', y=0.98)

    palette = ['#2b5c8f', '#27ae60', '#e74c3c', '#8e44ad']

    # (A) Training Loss Curves
    ax1 = axes[0, 0]
    for idx, (name, _) in enumerate(architectures):
        ax1.plot(range(1, epochs + 1), histories[name]["train_loss"], label=name, color=palette[idx], lw=2)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.set_title("(A) Training Loss Convergence", fontsize=11, weight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # (B) Validation Accuracy Curves
    ax2 = axes[0, 1]
    for idx, (name, _) in enumerate(architectures):
        ax2.plot(range(1, epochs + 1), [acc * 100 for acc in histories[name]["val_acc"]], label=name, color=palette[idx], lw=2)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax2.set_title("(B) Validation Accuracy Evolution", fontsize=11, weight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # (C) Test Accuracy & F1-Score Bar Chart
    ax3 = axes[1, 0]
    x = np.arange(len(df_metrics))
    width = 0.35
    b1 = ax3.bar(x - width/2, df_metrics["Test Accuracy"] * 100, width, label='Test Accuracy (%)', color='#2b5c8f', edgecolor='black', linewidth=0.5)
    b2 = ax3.bar(x + width/2, df_metrics["F1-Score (Macro)"] * 100, width, label='Macro F1 (%)', color='#e06666', edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_metrics["Architecture"], fontsize=9, rotation=15)
    ax3.set_ylabel("Score (%)", fontsize=10)
    ax3.set_title("(C) Test Set Generalization", fontsize=11, weight='bold')
    ax3.set_ylim(40, 75)
    ax3.legend(loc='lower right', frameon=True)
    for bar in b1 + b2:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # (D) Model Complexity: Parameters vs Test Accuracy
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_metrics["Parameters"], df_metrics["Test Accuracy"] * 100, c=palette, s=180, edgecolor='black', linewidth=1.2, zorder=3)
    for idx, row in df_metrics.iterrows():
        ax4.annotate(row["Architecture"], xy=(row["Parameters"], row["Test Accuracy"] * 100),
                     xytext=(10, 5), textcoords="offset points", fontsize=9, weight='bold')
    ax4.set_xlabel("Trainable Parameters", fontsize=10)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax4.set_title("(D) Capacity Tradeoff: Parameters vs Accuracy", fontsize=11, weight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path_local = os.path.join(os.path.dirname(__file__), "q13_mlp_wine_architectures.png")
    plot_path_global = os.path.join(output_dir, "q13_mlp_wine_architectures.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_metrics


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
