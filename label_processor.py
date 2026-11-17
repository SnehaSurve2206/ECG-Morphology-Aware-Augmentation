import os
import ast
from pathlib import Path

import pandas as pd

# Dataset paths
import os

PROJECT_ROOT = Path(__file__).resolve().parent

ROOT = Path(
    os.environ.get(
        "PTBXL_ROOT",
        PROJECT_ROOT / "ptb-xl-dataset" /
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
).expanduser().resolve()

METADATA_PATH = ROOT / "ptbxl_database.csv"
SCP_PATH = ROOT / "scp_statements.csv"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def main():
    print("Loading PTB-XL metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    print("Loading SCP statements...")

    # First column contains SCP codes
    scp = pd.read_csv(SCP_PATH, index_col=0)

    # Retain only diagnostic SCP codes
    diagnostic_scp = scp[
        scp["diagnostic"] == 1
    ]

    # Map SCP codes to diagnostic superclasses
    code_to_class = diagnostic_scp[
        "diagnostic_class"
    ].to_dict()

    print("Diagnostic SCP codes:", len(code_to_class))

    def extract_labels(scp_string):
        codes = ast.literal_eval(scp_string)

        labels = {name: 0 for name in CLASSES}

        for code in codes:
            superclass = code_to_class.get(code)

            if superclass in labels:
                labels[superclass] = 1

        return pd.Series(labels)

    print("Processing diagnostic labels...")

    label_matrix = metadata["scp_codes"].apply(
        extract_labels
    )

    # Combine metadata and labels
    output = pd.concat(
        [
            metadata[
                [
                    "ecg_id",
                    "patient_id",
                    "strat_fold",
                    "filename_lr"
                ]
            ],
            label_matrix
        ],
        axis=1
    )

    output_path = OUTPUT_DIR / "ptbxl_labels.csv"

    output.to_csv(output_path, index=False)

    print("\n--- LABEL PROCESSING COMPLETE ---")
    print("Total records:", len(output))

    print("\nClass distribution:")

    for name in CLASSES:
        positive = int(output[name].sum())
        negative = len(output) - positive

        print(
            f"{name}: {positive} positive, "
            f"{negative} negative"
        )

    print("\nLabel cardinality:")
    print(
        "Average labels per ECG:",
        round(output[CLASSES].sum(axis=1).mean(), 3)
    )

    print("\nRecords with no target labels:")
    print(int((output[CLASSES].sum(axis=1) == 0).sum()))

    print("\nSaved:", output_path.resolve())


if __name__ == "__main__":
    main()

