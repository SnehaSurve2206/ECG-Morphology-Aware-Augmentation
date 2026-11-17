
import random
import numpy as np
import pandas as pd
import neurokit2 as nk
import torch

from pathlib import Path
from data_loader import ECGDataset
from augmentations import apply_augmentation


# =====================================
# CONFIGURATION
# =====================================

SAMPLING_RATE = 100
NUM_RECORDS = 100
MATCH_TOLERANCE_MS = 100

TECHNIQUES = ["noise", "scaling", "shifting", "warping"]
STRENGTHS = ["mild", "moderate", "strong"]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# =====================================
# ECG CLEANING AND QRS DELINEATION
# =====================================

def analyze_qrs(signal):

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=SAMPLING_RATE,
        method="neurokit"
    )

    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=SAMPLING_RATE
    )

    peaks = np.asarray(info["ECG_R_Peaks"])

    _, waves = nk.ecg_delineate(
        cleaned,
        rpeaks=peaks,
        sampling_rate=SAMPLING_RATE,
        method="dwt",
        show=False
    )

    onsets = np.asarray(
        waves["ECG_R_Onsets"], dtype=float
    )

    offsets = np.asarray(
        waves["ECG_R_Offsets"], dtype=float
    )

    durations = np.full(len(peaks), np.nan)

    valid_count = 0

    for i in range(min(len(peaks), len(onsets), len(offsets))):

        onset = onsets[i]
        offset = offsets[i]

        if (
            np.isfinite(onset)
            and np.isfinite(offset)
            and 0 <= onset < offset < len(signal)
        ):

            durations[i] = (
                offset - onset
            ) * 1000 / SAMPLING_RATE

            valid_count += 1

    return {
        "cleaned": cleaned,
        "peaks": peaks,
        "onsets": onsets,
        "offsets": offsets,
        "durations": durations,
        "valid_qrs": valid_count
    }


# =====================================
# R-PEAK MATCHING
# =====================================

def match_peaks(original, augmented):

    tolerance = int(
        MATCH_TOLERANCE_MS * SAMPLING_RATE / 1000
    )

    used = set()
    matches = []

    for oi, original_peak in enumerate(original):

        candidates = [
            (
                abs(original_peak - augmented[ai]),
                ai
            )
            for ai in range(len(augmented))
            if ai not in used
        ]

        if not candidates:
            continue

        distance, ai = min(candidates)

        if distance <= tolerance:
            used.add(ai)
            matches.append((oi, ai))

    return matches


# =====================================
# MAIN EXPERIMENT
# =====================================

def main():

    dataset = ECGDataset("train")
    count = min(NUM_RECORDS, len(dataset))

    results = []

    print("QRS DELINEATION VALIDATION")
    print("=" * 55)
    print(f"Records: {count}")

    for index in range(count):

        signal, _ = dataset[index]

        original = signal[1].numpy()

        try:

            original_data = analyze_qrs(original)

            for technique in TECHNIQUES:

                for strength in STRENGTHS:

                    augmented_signal = apply_augmentation(
                        signal.clone(),
                        technique,
                        strength
                    )

                    augmented = augmented_signal[1].numpy()

                    augmented_data = analyze_qrs(augmented)

                    matches = match_peaks(
                        original_data["peaks"],
                        augmented_data["peaks"]
                    )

                    qrs_differences = []

                    valid_matched_pairs = 0

                    for oi, ai in matches:

                        original_duration = (
                            original_data["durations"][oi]
                            if oi < len(original_data["durations"])
                            else np.nan
                        )

                        augmented_duration = (
                            augmented_data["durations"][ai]
                            if ai < len(augmented_data["durations"])
                            else np.nan
                        )

                        if (
                            np.isfinite(original_duration)
                            and np.isfinite(augmented_duration)
                        ):

                            difference = abs(
                                original_duration
                                - augmented_duration
                            )

                            qrs_differences.append(difference)
                            valid_matched_pairs += 1

                    valid_differences = np.asarray(
                        qrs_differences
                    )

                    results.append({

                        "Record": index,
                        "Technique": technique,
                        "Strength": strength,

                        "Original_R_Peaks":
                            len(original_data["peaks"]),

                        "Augmented_R_Peaks":
                            len(augmented_data["peaks"]),

                        "Original_Valid_QRS":
                            original_data["valid_qrs"],

                        "Augmented_Valid_QRS":
                            augmented_data["valid_qrs"],

                        "Matched_R_Peaks":
                            len(matches),

                        "Valid_Matched_QRS":
                            valid_matched_pairs,

                        "QRS_Mean_Difference_ms":
                            np.mean(valid_differences)
                            if len(valid_differences) else np.nan,

                        "QRS_Median_Difference_ms":
                            np.median(valid_differences)
                            if len(valid_differences) else np.nan,

                        "QRS_Std_Difference_ms":
                            np.std(valid_differences)
                            if len(valid_differences) else np.nan
                    })

        except Exception as error:

            print(f"Record {index} failed: {error}")

        print(f"Completed {index + 1}/{count}")

    # =================================
    # SAVE DETAILED RESULTS
    # =================================

    Path("results").mkdir(exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        print("No results generated.")
        return

    df.to_csv(
        "results/qrs_validation_detailed.csv",
        index=False
    )

    # =================================
    # SUMMARY
    # =================================

    summary = df.groupby(
        ["Technique", "Strength"]
    ).agg({

        "Original_Valid_QRS": "mean",
        "Augmented_Valid_QRS": "mean",
        "Matched_R_Peaks": "mean",
        "Valid_Matched_QRS": "mean",
        "QRS_Mean_Difference_ms": "mean",
        "QRS_Median_Difference_ms": "median",
        "QRS_Std_Difference_ms": "mean"

    }).reset_index()

    summary.to_csv(
        "results/qrs_validation_summary.csv",
        index=False
    )

    print("\nQRS VALIDATION SUMMARY")
    print("=" * 100)

    print(summary.round(3).to_string(index=False))

    print("\nSaved files:")
    print("results/qrs_validation_detailed.csv")
    print("results/qrs_validation_summary.csv")


if __name__ == "__main__":
    main()
