
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

from data_loader import ECGDataset
from models import ECG_CNN

# Configuration
BATCH_SIZE = 32
device = torch.device("cpu")

classes = ["NORM", "MI", "STTC", "CD", "HYP"]

# Load test dataset
test_dataset = ECGDataset("test")
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = ECG_CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("results/best_cnn_model.pth", map_location=device)
)
model.eval()

all_labels = []
all_probabilities = []

# Generate predictions
with torch.no_grad():
    for signals, labels in test_loader:
        signals = signals.to(device)

        outputs = model(signals)
        probabilities = torch.sigmoid(outputs)

        all_labels.append(labels.numpy())
        all_probabilities.append(probabilities.numpy())

y_true = np.vstack(all_labels)
y_prob = np.vstack(all_probabilities)

# Convert probabilities into binary predictions
y_pred = (y_prob >= 0.5).astype(int)

# Calculate metrics
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

macro_auroc = roc_auc_score(
    y_true, y_prob, average="macro"
)

print("\nBASELINE CNN TEST RESULTS")
print("=" * 40)

print(f"Macro-F1: {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro AUROC: {macro_auroc:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=classes,
    zero_division=0
))

# Save results
np.savez(
    "results/baseline_test_predictions.npz",
    y_true=y_true,
    y_prob=y_prob,
    y_pred=y_pred
)

print("Predictions saved successfully.")
