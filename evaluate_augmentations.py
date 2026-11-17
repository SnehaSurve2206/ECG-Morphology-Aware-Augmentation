
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path

from data_loader import ECGDataset
from augmentations import apply_augmentation

TECHNIQUES = ["noise", "scaling", "shifting", "warping"]
STRENGTHS = ["mild", "moderate", "strong"]

NUM_RECORDS = 100


def calculate_metrics(original, augmented):

    original = original.numpy()
    augmented = augmented.numpy()

    mae = np.mean(np.abs(original - augmented))
    rmse = np.sqrt(np.mean((original - augmented) ** 2))

    correlation = np.corrcoef(
        original.flatten(),
        augmented.flatten()
    )[0, 1]

    return mae, rmse, correlation


def calculate_rpeak_difference(original, augmented):

    # Lead II
    original = original[1].numpy()
    augmented = augmented[1].numpy()

    peaks_original, _ = find_peaks(
        original, distance=25, prominence=0.8
    )

    peaks_augmented, _ = find_peaks(
        augmented, distance=25, prominence=0.8
    )

    if len(peaks_original) == 0 or len(peaks_augmented) == 0:
        return np.nan

    differences = []

    for peak in peaks_original:
        nearest = np.argmin(np.abs(peaks_augmented - peak))
        difference = abs(peaks_augmented[nearest] - peak)

        if difference <= 15:
            differences.append(difference)

    if not differences:
        return np.nan

    # Convert samples to milliseconds (100 Hz)
    return np.mean(differences) * 10


def main():

    dataset = ECGDataset("train")

    count = min(NUM_RECORDS, len(dataset))

    results = []

    print(f"Evaluating {count} ECG records...")
    
    for technique in TECHNIQUES:
        for strength in STRENGTHS:

            mae_values = []
            rmse_values = []
            correlation_values = []
            rpeak_values = []

            for index in range(count):

                signal, _ = dataset[index]

                augmented = apply_augmentation(
                    signal.clone(), technique, strength
                )

                mae, rmse, correlation = calculate_metrics(
                    signal, augmented
                )

                rpeak_difference = calculate_rpeak_difference(
                    signal, augmented
                )

                mae_values.append(mae)
                rmse_values.append(rmse)
                correlation_values.append(correlation)
                rpeak_values.append(rpeak_difference)

            results.append({
                "Technique": technique,
                "Strength": strength,
                "MAE": np.nanmean(mae_values),
                "RMSE": np.nanmean(rmse_values),
                "Correlation": np.nanmean(correlation_values),
                "R_Peak_Difference_ms": np.nanmean(rpeak_values)
            })

            print(f"Completed: {technique} - {strength}")

    df = pd.DataFrame(results)

    Path("results").mkdir(exist_ok=True)

    df.to_csv(
        "results/augmentation_metrics.csv",
        index=False
    )

    print("\nAUGMENTATION METRICS")
    print("=" * 70)
    print(df.round(4).to_string(index=False))

    print("\nSaved: results/augmentation_metrics.csv")


if __name__ == "__main__":
    main()
