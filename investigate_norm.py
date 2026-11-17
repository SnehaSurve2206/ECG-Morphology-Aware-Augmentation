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

meta = pd.read_csv(ROOT / "ptbxl_database.csv")
labels = pd.read_csv("results/ptbxl_labels.csv")
scp = pd.read_csv(ROOT / "scp_statements.csv", index_col=0)

classes = ["NORM", "MI", "STTC", "CD", "HYP"]

abnormal = ["MI", "STTC", "CD", "HYP"]

overlap = labels[
    (labels.NORM == 1) &
    (labels[abnormal].sum(axis=1) > 0)
]

print("Total NORM + abnormal:", len(overlap))

print("\n--- CO-OCCURRENCE COUNTS ---")

for c in abnormal:
    print(c, int((overlap[c] == 1).sum()))

print("\n--- FIRST 20 EXAMPLES ---")

for _, row in overlap.head(20).iterrows():
    original = meta.loc[
        meta.ecg_id == row.ecg_id, "scp_codes"
    ].iloc[0]

    codes = ast.literal_eval(original)

    print("\nECG ID:", row.ecg_id)
    print("Labels:", {
        c: int(row[c]) for c in classes
    })

    for code, confidence in codes.items():
        if code in scp.index:
            details = scp.loc[code]

            print(
                code,
                "| confidence:", confidence,
                "| diagnostic:", details["diagnostic"],
                "| class:", details["diagnostic_class"]
            )
        else:
            print(code, "| Not found in SCP mapping")

