
import random
import numpy as np
import pandas as pd
import wfdb
import torch

from pathlib import Path
from scipy.signal import correlate, correlation_lags

from morphology_500hz import (
    ROOT,
    CONFIG,
    augment,
    TECHNIQUES,
    STRENGTHS
)


# ==========================================
# CONFIGURATION
# ==========================================

NUM_RECORDS = 100
SAMPLING_RATE = 500

# Maximum alignment search: +/- 200 ms
MAX_LAG_MS = 200
MAX_LAG_SAMPLES = int(
    MAX_LAG_MS * SAMPLING_RATE / 1000
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# LOAD ORIGINAL ECG
# ==========================================

def load_record(filename):

    record = wfdb.rdrecord(str(ROOT / filename))

    signal = record.p_signal.T.astype(np.float32)

    if record.fs != SAMPLING_RATE:
        raise ValueError("Incorrect sampling rate")

    return signal


# ==========================================
# ESTIMATE GLOBAL TIME SHIFT
# ==========================================

def estimate_lag(original, augmented):

    # Use Lead II to estimate a common displacement
    x = augmented[1] - np.mean(augmented[1])
    y = original[1] - np.mean(original[1])

    correlation = correlate(
        x,
        y,
        mode="full",
        method="fft"
    )

    lags = correlation_lags(
        len(x),
        len(y),
        mode="full"
    )

    valid = np.abs(lags) <= MAX_LAG_SAMPLES

    best_index = np.argmax(correlation[valid])

    lag = lags[valid][best_index]

    return int(lag)


# ==========================================
# ALIGN SIGNALS
# ==========================================

def align_signals(original, augmented, lag):

    if lag > 0:

        original_aligned = original[:, :-lag]
        augmented_aligned = augmented[:, lag:]

    elif lag < 0:

        original_aligned = original[:, -lag:]
        augmented_aligned = augmented[:, :lag]

    else:

        original_aligned = original
        augmented_aligned = augmented

    return original_aligned, augmented_aligned


# ==========================================
# CALCULATE SIGNAL METRICS
# ==========================================

def calculate_metrics(original, augmented):

    x = original.flatten()
    y = augmented.flatten()

    mae = np.mean(np.abs(x - y))

    rmse = np.sqrt(np.mean((x - y) ** 2))

    correlation = np.corrcoef(x, y)[0, 1]

    return mae, rmse, correlation


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    metadata_path = ROOT / "ptbxl_database.csv"
    labels_path = Path("results/ptbxl_labels.csv")

    metadata = pd.read_csv(metadata_path)
    labels = pd.read_csv(labels_path)

    data = labels.merge(
        metadata[["ecg_id", "filename_hr"]],
        on="ecg_id",
        how="left",
        validate="one_to_one"
    )

    data = data[
        (data["strat_fold"] <= 8) &
        (data[["NORM", "MI", "STTC", "CD", "HYP"]].sum(axis=1) > 0)
    ]

    data = data.head(NUM_RECORDS).reset_index(drop=True)

    results = []

    print("ALIGNMENT-AWARE ECG EVALUATION")
    print("=" * 55)
    print("Records:", len(data))
    print("Sampling rate:", SAMPLING_RATE, "Hz")
    print("Maximum lag:", MAX_LAG_MS, "ms")

    for index, row in data.iterrows():

        try:

            original = load_record(row["filename_hr"])

            for technique in TECHNIQUES:

                for strength in STRENGTHS:

                    augmented = augment(
                        original.copy(),
                        technique,
                        strength
                    )

                    # Raw comparison
                    raw_mae, raw_rmse, raw_corr = (
                        calculate_metrics(original, augmented)
                    )

                    # Estimate common temporal displacement
                    lag = estimate_lag(
                        original,
                        augmented
                    )

                    # Align using the estimated lag
                    original_aligned, augmented_aligned = (
                        align_signals(original, augmented, lag)
                    )

                    # Aligned comparison
                    aligned_mae, aligned_rmse, aligned_corr = (
                        calculate_metrics(
                            original_aligned,
                            augmented_aligned
                        )
                    )

                    results.append({

                        "ECG_ID": row["ecg_id"],
                        "Technique": technique,
                        "Strength": strength,

                        "Estimated_Lag_Samples": lag,

                        "Estimated_Shift_ms":
                            lag * 1000 / SAMPLING_RATE,

                        "Raw_MAE": raw_mae,
                        "Raw_RMSE": raw_rmse,
                        "Raw_Correlation": raw_corr,

                        "Aligned_MAE": aligned_mae,
                        "Aligned_RMSE": aligned_rmse,
                        "Aligned_Correlation": aligned_corr,

                        "Overlap_Samples":
                            original_aligned.shape[1]
                    })

        except Exception as error:

            print(
                f"ECG {row['ecg_id']} failed: {error}"
            )

        print(f"Completed {index + 1}/{len(data)}")

    # ======================================
    # SAVE DETAILED RESULTS
    # ======================================

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        print("No results generated.")
        return

    df.to_csv(
        output_dir / "alignment_500hz_detailed.csv",
        index=False
    )

    # ======================================
    # SUMMARY
    # ======================================

    summary = df.groupby(
        ["Technique", "Strength"]
    ).agg({

        "Estimated_Shift_ms": ["mean", "std"],

        "Raw_MAE": "mean",
        "Raw_RMSE": "mean",
        "Raw_Correlation": "mean",

        "Aligned_MAE": "mean",
        "Aligned_RMSE": "mean",
        "Aligned_Correlation": "mean",

        "Overlap_Samples": "mean"

    }).reset_index()

    summary.columns = [
        "_".join(
            str(part) for part in column if part
        )
        for column in summary.columns
    ]

    summary.to_csv(
        output_dir / "alignment_500hz_summary.csv",
        index=False
    )

    print("\nALIGNMENT EVALUATION SUMMARY")
    print("=" * 90)

    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print("results/alignment_500hz_detailed.csv")
    print("results/alignment_500hz_summary.csv")


if __name__ == "__main__":
    main()
