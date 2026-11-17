
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from torch.utils.data import Dataset, DataLoader


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset directory (can be overridden using PTBXL_ROOT)
ROOT = Path(
    os.environ.get(
        "PTBXL_ROOT",
        PROJECT_ROOT
        / "ptb-xl-dataset"
        / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    )
).expanduser().resolve()

RESULTS_DIR = PROJECT_ROOT / "results"

LABELS_FILE = RESULTS_DIR / "ptbxl_labels.csv"
STATS_FILE = RESULTS_DIR / "training_statistics.npz"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ECGDataset(Dataset):

    def __init__(self, fold_type):

        if not LABELS_FILE.exists():
            raise FileNotFoundError(
                f"Labels file not found: {LABELS_FILE}"
            )

        if not STATS_FILE.exists():
            raise FileNotFoundError(
                f"Normalization statistics not found: {STATS_FILE}"
            )

        if not ROOT.exists():
            raise FileNotFoundError(
                f"PTB-XL dataset not found: {ROOT}\n"
                "Set the PTBXL_ROOT environment variable."
            )

        self.labels = pd.read_csv(LABELS_FILE)

        self.labels = self.labels[
            self.labels[CLASSES].sum(axis=1) > 0
        ].reset_index(drop=True)

        if fold_type == "train":
            self.labels = self.labels[
                self.labels.strat_fold <= 8
            ]

        elif fold_type == "validation":
            self.labels = self.labels[
                self.labels.strat_fold == 9
            ]

        elif fold_type == "test":
            self.labels = self.labels[
                self.labels.strat_fold == 10
            ]

        else:
            raise ValueError(
                "fold_type must be train, validation, or test"
            )

        self.labels = self.labels.reset_index(drop=True)

        stats = np.load(STATS_FILE)

        self.mean = stats["mean"]
        self.std = np.maximum(stats["std"], 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        row = self.labels.iloc[index]

        path = ROOT / row.filename_lr

        signal, _ = wfdb.rdsamp(str(path))

        signal = (signal - self.mean) / self.std

        signal = torch.tensor(
            signal.T,
            dtype=torch.float32
        )

        target = torch.tensor(
            row[CLASSES].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        return signal, target


if __name__ == "__main__":

    train_dataset = ECGDataset("train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    signals, labels = next(iter(train_loader))

    print("\n--- DATA LOADER TEST ---")
    print("Dataset path:", ROOT)
    print("Training records:", len(train_dataset))
    print("Signal batch shape:", signals.shape)
    print("Label batch shape:", labels.shape)
    print("Signal data type:", signals.dtype)
    print("Label data type:", labels.dtype)
    print("All values finite:", torch.isfinite(signals).all().item())

    print("\nData Loader working successfully!")
