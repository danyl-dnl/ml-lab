r"""
Question 14: Neural Network with Different Activation Functions (Sigmoid, ReLU, Tanh) on MNIST
Description:
    Implement and compare the performance of a neural network using different
    activation functions (Sigmoid, ReLU, Tanh) on the MNIST dataset. Analyze
    how each activation function affects the training dynamics, convergence
    speed, vanishing gradient susceptibility, and test accuracy.

Tasks:
    1. Load and preprocess the MNIST dataset (flattening 28x28 images, scaling to [0, 1]).
    2. Implement identical neural network architectures parameterized by activation function.
    3. Train networks with Sigmoid, Tanh, and ReLU under identical optimizer and seed settings.
    4. Track epoch-by-epoch loss convergence, validation accuracy, and training runtimes.
    5. Evaluate on held-out test data (Accuracy, Precision, Recall, F1-Score).
    6. Discuss vanishing gradients, zero-centering, and computational efficiency.
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

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)


class ModularActivationMLP(nn.Module):
    """
    Identical 2-hidden layer feed-forward network parameterized by activation function.
    Architecture: 784 -> 128 -> Activation -> 64 -> Activation -> 10
    """
    def __init__(self, activation_name: str):
        super(ModularActivationMLP, self).__init__()
        self.activation_name = activation_name

        if activation_name == 'Sigmoid':
            act = nn.Sigmoid()
        elif activation_name == 'Tanh':
            act = nn.Tanh()
        elif activation_name == 'ReLU':
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            act,
            nn.Linear(128, 64),
            act,
            nn.Linear(64, 10),
        )

        # Xavier / Glorot initialization for consistent starting scale
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def load_mnist_data(data_dir: str = "../data", batch_size: int = 64):
    """Load MNIST dataset with standard PyTorch dataloaders."""
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = dsets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = dsets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Use 15,000 train samples and 3,000 validation samples for rapid, controlled benchmarking
    np.random.seed(42)
    indices = np.random.permutation(len(train_full))
    train_idx = indices[:15000]
    val_idx = indices[15000:18000]

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(train_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"MNIST subsets -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_data)}")
    return train_loader, val_loader, test_loader


def train_and_evaluate(act_name: str, train_loader, val_loader, test_loader, epochs: int = 15, lr: float = 0.05):
    """Train network with specified activation function and record epoch dynamics."""
    print(f"\n--- Training Network with [{act_name}] Activation Function ---")
    model = ModularActivationMLP(act_name)
    criterion = nn.CrossEntropyLoss()
    # SGD optimizer with momentum highlights gradient propagation differences
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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

        # Validation accuracy
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()

        val_acc = correct / len(val_loader.dataset)
        val_accuracies.append(val_acc)

        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    train_time = time.time() - t0

    # Test evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            all_preds.extend(outputs.argmax(dim=1).numpy())
            all_targets.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    prec_weighted = precision_score(all_targets, all_preds, average='weighted')
    rec_weighted = recall_score(all_targets, all_preds, average='weighted')

    print(f"[{act_name} Final] Test Acc: {test_acc*100:.2f}% | Macro F1: {f1_macro:.4f} | Time: {train_time:.2f}s")

    metrics = {
        "Activation": act_name,
        "Test Accuracy": test_acc,
        "Precision (W)": prec_weighted,
        "Recall (W)": rec_weighted,
        "F1-Score (Macro)": f1_macro,
        "Final Loss": train_losses[-1],
        "Train Time (s)": train_time,
    }

    history = {
        "train_loss": train_losses,
        "val_acc": val_accuracies,
    }

    return metrics, history, all_preds, all_targets


def run_experiment(output_dir: str = "../outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" QUESTION 14: ACTIVATION FUNCTION BENCHMARK ON MNIST (SIGMOID, RELU, TANH)")
    print("=" * 70)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    train_loader, val_loader, test_loader = load_mnist_data(data_dir=data_dir, batch_size=64)

    activations = ["Sigmoid", "Tanh", "ReLU"]
    epochs = 15
    results = []
    histories = {}
    preds_dict = {}

    for act in activations:
        m, h, p, targets = train_and_evaluate(act, train_loader, val_loader, test_loader, epochs=epochs, lr=0.03)
        results.append(m)
        histories[act] = h
        preds_dict[act] = p

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print(" ACTIVATION FUNCTION PERFORMANCE COMPARISON SUMMARY")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Impact of Activation Functions (Sigmoid, Tanh, ReLU) on MNIST Neural Network Training",
                 fontsize=14, weight='bold', y=0.98)

    colors = {'Sigmoid': '#e74c3c', 'Tanh': '#f39c12', 'ReLU': '#27ae60'}

    # (A) Training Loss Progression
    ax1 = axes[0, 0]
    for act in activations:
        ax1.plot(range(1, epochs + 1), histories[act]["train_loss"], label=act, color=colors[act], lw=2.2)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.set_title("(A) Training Loss Convergence", fontsize=11, weight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # (B) Validation Accuracy Progression
    ax2 = axes[0, 1]
    for act in activations:
        ax2.plot(range(1, epochs + 1), [acc * 100 for acc in histories[act]["val_acc"]], label=act, color=colors[act], lw=2.2)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax2.set_title("(B) Validation Accuracy Across Epochs", fontsize=11, weight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # (C) Test Accuracy Comparison Bar Chart
    ax3 = axes[1, 0]
    bars = ax3.bar(df_results["Activation"], df_results["Test Accuracy"] * 100, color=[colors[a] for a in df_results["Activation"]],
                   edgecolor='black', linewidth=0.6, width=0.55)
    ax3.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax3.set_title("(C) Final Test Accuracy", fontsize=11, weight='bold')
    ax3.set_ylim(80, 100)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

    # (D) Training Time vs Accuracy Tradeoff
    ax4 = axes[1, 1]
    for idx, row in df_results.iterrows():
        act = row["Activation"]
        ax4.scatter(row["Train Time (s)"], row["Test Accuracy"] * 100, color=colors[act], s=180, edgecolor='black', linewidth=1.2, zorder=3)
        ax4.annotate(f"{act}\n({row['Test Accuracy']*100:.2f}%)", xy=(row["Train Time (s)"], row["Test Accuracy"] * 100),
                     xytext=(10, 0), textcoords="offset points", fontsize=9, weight='bold')
    ax4.set_xlabel("Total Training Time (seconds)", fontsize=10)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax4.set_title("(D) Efficiency vs Accuracy Tradeoff", fontsize=11, weight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path_local = os.path.join(os.path.dirname(__file__), "q14_activation_functions_mnist.png")
    plot_path_global = os.path.join(output_dir, "q14_activation_functions_mnist.png")
    plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_global, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[✓] Plots saved to:\n  - {plot_path_local}\n  - {plot_path_global}")
    return df_results


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    run_experiment(output_dir=out_dir)
