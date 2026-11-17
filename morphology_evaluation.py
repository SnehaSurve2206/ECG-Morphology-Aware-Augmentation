
import random
import numpy as np
import pandas as pd
import neurokit2 as nk
import torch

from pathlib import Path
from data_loader import ECGDataset
from augmentations import apply_augmentation


# ==========================================
# CONFIGURATION
# ==========================================

SAMPLING_RATE = 100
NUM_RECORDS = 100
MATCH_TOLERANCE_MS = 100

TECHNIQUES = ["noise", "scaling", "shifting", "warping"]
STRENGTHS = ["mild", "moderate", "strong"]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# R-PEAK DETECTION
# ==========================================

def detect_rpeaks(signal):

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=SAMPLING_RATE,
        method="neurokit"
    )

    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=SAMPLING_RATE
    )

    return cleaned, np.asarray(info["ECG_R_Peaks"])


# ==========================================
# QRS DELINEATION
# ==========================================

def get_qrs_durations(cleaned, peaks):

    try:
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

        durations = []

        for onset, offset in zip(onsets, offsets):

            if (
                np.isfinite(onset)
                and np.isfinite(offset)
                and offset > onset
            ):
                durations.append(
                    (offset - onset) * 1000 / SAMPLING_RATE
                )
            else:
                durations.append(np.nan)

        return np.asarray(durations)

    except Exception:
        return np.full(len(peaks), np.nan)


# ==========================================
# ONE-TO-ONE R-PEAK MATCHING
# ==========================================

def match_rpeaks(original_peaks, augmented_peaks):

    tolerance = int(
        MATCH_TOLERANCE_MS * SAMPLING_RATE / 1000
    )

    used_augmented = set()
    matched = []

    for original_index, original_peak in enumerate(original_peaks):

        candidates = [
            (
                abs(original_peak - augmented_peak),
                augmented_index
            )
            for augmented_index, augmented_peak
            in enumerate(augmented_peaks)
            if augmented_index not in used_augmented
        ]

        if not candidates:
            continue

        distance, best_index = min(candidates)

        if distance <= tolerance:

            used_augmented.add(best_index)

            matched.append(
                (original_index, best_index, distance)
            )

    return matched, used_augmented


# ==========================================
# SIGNAL METRICS
# ==========================================

def calculate_signal_metrics(original, augmented):

    mae = np.mean(
        np.abs(original - augmented)
    )

    rmse = np.sqrt(
        np.mean((original - augmented) ** 2)
    )

    correlation = np.corrcoef(
        original, augmented
    )[0, 1]

    return mae, rmse, correlation


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    dataset = ECGDataset("train")

    count = min(NUM_RECORDS, len(dataset))

    results = []

    print("MORPHOLOGY EVALUATION")
    print("=" * 55)
    print(f"Records: {count}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")

    for index in range(count):

        signal, _ = dataset[index]

        # Lead II
        original = signal[1].numpy()

        try:

            original_cleaned, original_peaks = detect_rpeaks(
                original
            )

            original_qrs = get_qrs_durations(
                original_cleaned,
                original_peaks
            )

            for technique in TECHNIQUES:

                for strength in STRENGTHS:

                    augmented_signal = apply_augmentation(
                        signal.clone(),
                        technique,
                        strength
                    )

                    augmented = augmented_signal[1].numpy()

                    augmented_cleaned, augmented_peaks = detect_rpeaks(
                        augmented
                    )

                    augmented_qrs = get_qrs_durations(
                        augmented_cleaned,
                        augmented_peaks
                    )

                    matched, used_augmented = match_rpeaks(
                        original_peaks,
                        augmented_peaks
                    )

                    # R-peak timing differences
                    timing_differences = [
                        distance * 1000 / SAMPLING_RATE
                        for _, _, distance in matched
                    ]

                    # R-peak amplitude differences
                    amplitude_differences = [
                        abs(
                            original[original_peaks[oi]]
                            - augmented[augmented_peaks[ai]]
                        )
                        for oi, ai, _ in matched
                    ]

                    # QRS duration differences
                    qrs_differences = []

                    for oi, ai, _ in matched:

                        if (
                            oi < len(original_qrs)
                            and ai < len(augmented_qrs)
                            and np.isfinite(original_qrs[oi])
                            and np.isfinite(augmented_qrs[ai])
                        ):
                            qrs_differences.append(
                                abs(
                                    original_qrs[oi]
                                    - augmented_qrs[ai]
                                )
                            )

                    mae, rmse, correlation = calculate_signal_metrics(
                        original, augmented
                    )

                    total_original = len(original_peaks)

                    match_rate = (
                        len(matched) / total_original * 100
                        if total_original > 0
                        else np.nan
                    )

                    results.append({

                        "Record": index,
                        "Technique": technique,
                        "Strength": strength,

                        "Original_Peaks": total_original,
                        "Augmented_Peaks": len(augmented_peaks),
                        "Matched_Peaks": len(matched),

                        "Match_Rate_Percent": match_rate,

                        "Mean_Timing_Difference_ms":
                            np.mean(timing_differences)
                            if timing_differences else np.nan,

                        "Mean_QRS_Duration_Difference_ms":
                            np.mean(qrs_differences)
                            if qrs_differences else np.nan,

                        "Mean_R_Amplitude_Difference":
                            np.mean(amplitude_differences)
                            if amplitude_differences else np.nan,

                        "Unmatched_Original":
                            total_original - len(matched),

                        "Unmatched_Augmented":
                            len(augmented_peaks) - len(used_augmented),

                        "MAE": mae,
                        "RMSE": rmse,
                        "Correlation": correlation
                    })

        except Exception as error:

            print(f"Record {index} failed: {error}")

        print(f"Completed record {index + 1}/{count}")

    # ======================================
    # SAVE DETAILED RESULTS
    # ======================================

    Path("results").mkdir(exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        print("No records evaluated successfully.")
        return

    df.to_csv(
        "results/morphology_detailed_results.csv",
        index=False
    )

    # ======================================
    # SUMMARY
    # ======================================

    metrics = [
        "Match_Rate_Percent",
        "Mean_Timing_Difference_ms",
        "Mean_QRS_Duration_Difference_ms",
        "Mean_R_Amplitude_Difference",
        "Unmatched_Original",
        "Unmatched_Augmented",
        "MAE",
        "RMSE",
        "Correlation"
    ]

    summary = df.groupby(
        ["Technique", "Strength"]
    )[metrics].mean().reset_index()

    summary.to_csv(
        "results/morphology_summary.csv",
        index=False
    )

    print("\nMORPHOLOGY EVALUATION SUMMARY")
    print("=" * 85)

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nSaved files:")
    print("results/morphology_detailed_results.csv")
    print("results/morphology_summary.csv")


if __name__ == "__main__":
    main()
