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

# Only records with at least one target label
classes = ["NORM", "MI", "STTC", "CD", "HYP"]
data = labels[labels[classes].sum(axis=1) > 0]

print("Records to inspect:", len(data))

valid = 0
invalid = 0
nonfinite = 0
wrong_shape = 0

for _, row in data.iterrows():
    record_path = ROOT / row.filename_lr

    try:
        signal, metadata = wfdb.rdsamp(str(record_path))

        if not np.isfinite(signal).all():
            nonfinite += 1
            continue

        if signal.shape != (1000, 12):
            wrong_shape += 1
            continue

        valid += 1

    except Exception as error:
        invalid += 1
        print("Error:", row.ecg_id, error)

print("\n--- SIGNAL QUALITY REPORT ---")
print("Valid signals:", valid)
print("Invalid signals:", invalid)
print("Non-finite signals:", nonfinite)
print("Wrong shape:", wrong_shape)
print("Total checked:", valid + invalid + nonfinite + wrong_shape)

