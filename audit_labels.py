import os
import ast
import pandas as pd
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
metadata = pd.read_csv(ROOT / "ptbxl_database.csv")

classes = ["NORM", "MI", "STTC", "CD", "HYP"]

print("\n--- 1. UNLABELED RECORDS ---")

unlabeled = labels[labels[classes].sum(axis=1) == 0]

print("Count:", len(unlabeled))

print("\nExample unlabeled ECGs and their SCP codes:")

examples = metadata[
    metadata.ecg_id.isin(unlabeled.ecg_id)
].head(10)

for _, row in examples.iterrows():
    print(row.ecg_id, ast.literal_eval(row.scp_codes))


print("\n--- 2. NORM CO-OCCURRENCE ---")

abnormal = ["MI", "STTC", "CD", "HYP"]

norm_overlap = labels[
    (labels.NORM == 1) &
    (labels[abnormal].sum(axis=1) > 0)
]

print("NORM with abnormal labels:", len(norm_overlap))


print("\n--- 3. LABEL COMBINATIONS ---")

combinations = (
    labels[classes]
    .astype(str)
    .agg("".join, axis=1)
    .value_counts()
)

print(combinations.head(20))


print("\n--- 4. FOLD-WISE CLASS DISTRIBUTION ---")

for fold in range(1, 11):
    subset = labels[labels.strat_fold == fold]

    print(f"\nFold {fold}: {len(subset)} records")

    for name in classes:
        print(f"{name}: {int(subset[name].sum())}")


print("\n--- 5. LABEL CARDINALITY ---")

print(
    "Average labels:",
    labels[classes].sum(axis=1).mean()
)

print(
    "Records with exactly one label:",
    int((labels[classes].sum(axis=1) == 1).sum())
)

print(
    "Records with multiple labels:",
    int((labels[classes].sum(axis=1) > 1).sum())
)

print("\nAudit completed.")

