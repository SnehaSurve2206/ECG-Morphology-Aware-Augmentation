import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).resolve().parent

ROOT = Path(
    os.environ.get(
        "PTBXL_ROOT",
        PROJECT_ROOT / "ptb-xl-dataset" /
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
).expanduser().resolve()

labels = pd.read_csv("results/ptbxl_labels.csv")

classes = ["NORM", "MI", "STTC", "CD", "HYP"]

data = labels[
    (labels[classes].sum(axis=1) > 0) &
    (labels["strat_fold"] <= 8)
]

total = 0
sum_signal = np.zeros(12, dtype=np.float64)
sum_squared = np.zeros(12, dtype=np.float64)
minimum = np.full(12, np.inf)
maximum = np.full(12, -np.inf)

print("Training ECG records:", len(data))
print("Calculating per-lead statistics...")

for i, row in enumerate(data.itertuples(), 1):
    path = ROOT / row.filename_lr

    signal, _ = wfdb.rdsamp(str(path))
    signal = signal.astype(np.float64)

    sum_signal += signal.sum(axis=0)
    sum_squared += np.square(signal).sum(axis=0)

    minimum = np.minimum(minimum, signal.min(axis=0))
    maximum = np.maximum(maximum, signal.max(axis=0))

    total += signal.shape[0]

    if i % 2000 == 0:
        print(f"Processed {i}/{len(data)}")

mean = sum_signal / total
variance = sum_squared / total - mean ** 2
std = np.sqrt(np.maximum(variance, 0))

print("\n--- TRAINING SIGNAL STATISTICS ---")

for i, lead in enumerate(
    ["I", "II", "III", "AVR", "AVL", "AVF",
     "V1", "V2", "V3", "V4", "V5", "V6"]
):
    print(
        f"{lead:>4} | Mean: {mean[i]:.6f} | "
        f"Std: {std[i]:.6f} | "
        f"Min: {minimum[i]:.4f} | "
        f"Max: {maximum[i]:.4f}"
    )

Path("results").mkdir(exist_ok=True)

np.savez(
    "results/training_statistics.npz",
    mean=mean,
    std=std,
    minimum=minimum,
    maximum=maximum
)

print("\nStatistics saved successfully.")

