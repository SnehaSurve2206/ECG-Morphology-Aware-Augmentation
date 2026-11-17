import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import ECGDataset
from models import ECG_CNN
from augmentations import apply_augmentation


# -------------------- CONFIGURATION --------------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")

TECHNIQUES = ["noise", "scaling", "shifting", "warping"]
STRENGTHS = ["mild", "moderate", "strong"]

RESULTS_DIR = Path("results/classification_experiments")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AugmentedTrainingDataset(Dataset):
    """
    For each original training ECG, provide:
    - one unchanged copy
    - one dynamically augmented copy

    Augmentation is applied only to the training dataset.
    """

    def __init__(self, base_dataset, technique=None, strength=None):
        self.base_dataset = base_dataset
        self.technique = technique
        self.strength = strength
        self.use_augmentation = technique is not None

    def __len__(self):
        if self.use_augmentation:
            return 2 * len(self.base_dataset)
        return len(self.base_dataset)

    def __getitem__(self, index):
        if not self.use_augmentation:
            return self.base_dataset[index]

        original_index = index % len(self.base_dataset)
        signal, label = self.base_dataset[original_index]

        # First half: original ECG. Second half: augmented ECG.
        if index >= len(self.base_dataset):
            signal = apply_augmentation(
                signal,
                self.technique,
                self.strength
            )

        return signal, label


def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int32)

    metrics = {
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

    try:
        metrics["macro_auroc"] = roc_auc_score(
            y_true, y_prob, average="macro"
        )
    except ValueError:
        metrics["macro_auroc"] = float("nan")

    return metrics


def run_experiment(experiment_name, technique=None, strength=None):
    print("\n" + "=" * 72)
    print("EXPERIMENT:", experiment_name)
    print("=" * 72)

    # Same initialization seed for every experiment.
    set_seed(SEED)

    base_train = ECGDataset("train")
    train_dataset = AugmentedTrainingDataset(
        base_train,
        technique=technique,
        strength=strength
    )

    # Validation and test data are never augmented.
    val_dataset = ECGDataset("validation")
    test_dataset = ECGDataset("test")

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = ECG_CNN(num_classes=5).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    checkpoint_path = CHECKPOINT_DIR / f"{experiment_name}.pth"
    best_val_loss = float("inf")

    print("Original training records:", len(base_train))
    print("Training examples per epoch:", len(train_dataset))
    print("Validation records:", len(val_dataset))
    print("Test records:", len(test_dataset))

    # -------------------- TRAINING --------------------

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for signals, labels in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * signals.size(0)

        train_loss = train_loss_total / len(train_dataset)

        # Validation uses original, unaugmented ECGs.
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * signals.size(0)

        val_loss = val_loss_total / len(val_dataset)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    # -------------------- TEST EVALUATION --------------------

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for signals, labels in test_loader:
            outputs = model(signals.to(DEVICE))
            probabilities = torch.sigmoid(outputs)

            all_labels.append(labels.numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probabilities)

    metrics = calculate_metrics(y_true, y_prob)

    result = {
        "experiment": experiment_name,
        "technique": technique if technique else "none",
        "strength": strength if strength else "none",
        "best_validation_loss": best_val_loss,
        "epochs": EPOCHS,
        "seed": SEED,
        **metrics,
    }

    print("\nTEST RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save predictions for reproducibility and later analysis.
    np.savez_compressed(
        RESULTS_DIR / f"{experiment_name}_predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
        y_pred=(y_prob >= 0.5).astype(np.int32)
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PTB-XL ECG augmentation classification experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help=(
            "baseline, all, or one of: "
            "noise_mild, noise_moderate, noise_strong, "
            "scaling_mild, scaling_moderate, scaling_strong, "
            "shifting_mild, shifting_moderate, shifting_strong, "
            "warping_mild, warping_moderate, warping_strong"
        )
    )
    args = parser.parse_args()

    experiments = {
        "baseline": (None, None)
    }

    for technique in TECHNIQUES:
        for strength in STRENGTHS:
            experiments[f"{technique}_{strength}"] = (
                technique, strength
            )

    if args.experiment == "all":
        selected = list(experiments.items())
    elif args.experiment in experiments:
        selected = [(args.experiment, experiments[args.experiment])]
    else:
        raise ValueError(
            f"Unknown experiment '{args.experiment}'. "
            f"Choose from: {', '.join(experiments)} or 'all'."
        )

    results = []

    for name, (technique, strength) in selected:
        result = run_experiment(name, technique, strength)
        results.append(result)

        # Save progress after every completed experiment.
        output_file = RESULTS_DIR / "classification_results.csv"
        current = pd.DataFrame(results)

        if output_file.exists() and args.experiment != "all":
            previous = pd.read_csv(output_file)
            previous = previous[previous["experiment"] != name]
            current = pd.concat([previous, current], ignore_index=True)

        current.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    print("\nEXPERIMENTS COMPLETED")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
