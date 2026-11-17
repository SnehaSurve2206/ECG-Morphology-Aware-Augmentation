
import random
import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb
import torch

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = next(Path("ptb-xl-dataset").iterdir())

LABEL_FILE = Path("results/ptbxl_labels.csv")
METADATA_FILE = ROOT / "ptbxl_database.csv"

SAMPLING_RATE = 500
NUM_RECORDS = 100
MATCH_TOLERANCE_MS = 100

TECHNIQUES = ["noise", "scaling", "shifting", "warping"]
STRENGTHS = ["mild", "moderate", "strong"]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# AUGMENTATION SETTINGS
# ==========================================

CONFIG = {
    "mild": {
        "noise": 0.01,
        "scale": 0.05,
        "shift": 10,
        "warp": 25
    },
    "moderate": {
        "noise": 0.03,
        "scale": 0.10,
        "shift": 25,
        "warp": 50
    },
    "strong": {
        "noise": 0.05,
        "scale": 0.20,
        "shift": 50,
        "warp": 100
    }
}


# ==========================================
# AUGMENTATION FUNCTIONS
# Input: 12 leads x 5000 samples
# ==========================================

def augment(signal, technique, strength):

    config = CONFIG[strength]
    output = signal.copy()

    if technique == "noise":

        # Noise relative to each lead's standard deviation
        lead_std = np.std(output, axis=1, keepdims=True)

        noise = np.random.normal(
            0,
            config["noise"],
            size=output.shape
        ) * lead_std

        return output + noise

    elif technique == "scaling":

        factor = random.uniform(
            1 - config["scale"],
            1 + config["scale"]
        )

        return output * factor

    
    elif technique == "shifting":

        max_shift = config["shift"]

        shift = random.randint(
            -max_shift,
            max_shift
        )

        if shift == 0:
            return output.copy()

        shifted = np.zeros_like(output)

        for lead in range(output.shape[0]):

            padded = np.pad(
                output[lead],
                (max_shift, max_shift),
                mode="reflect"
            )

            start = max_shift - shift

            shifted[lead] = padded[
                start:start + output.shape[1]
            ]

        return shifted


    elif technique == "warping":

        length = output.shape[1]

        t = np.linspace(0, 2 * np.pi, length)

        amplitude = random.uniform(
            -config["warp"],
            config["warp"]
        )

        displacement = amplitude * np.sin(t)

        positions = np.arange(length) + displacement

        positions = np.clip(
            positions, 0, length - 1
        )

        warped = np.zeros_like(output)

        for lead in range(output.shape[0]):

            warped[lead] = np.interp(
                positions,
                np.arange(length),
                output[lead]
            )

        return warped

    raise ValueError("Unknown augmentation technique")


# ==========================================
# LOAD 500 Hz RECORD
# ==========================================

def load_record(filename):

    record_path = ROOT / filename

    record = wfdb.rdrecord(str(record_path))

    signal = record.p_signal.T.astype(np.float32)

    if record.fs != SAMPLING_RATE:
        raise ValueError(
            f"Unexpected sampling rate: {record.fs}"
        )

    if signal.shape != (12, 5000):
        raise ValueError(
            f"Unexpected signal shape: {signal.shape}"
        )

    return signal


# ==========================================
# ECG AND QRS ANALYSIS
# ==========================================

def analyze_ecg(signal):

    lead_ii = signal[1]

    cleaned = nk.ecg_clean(
        lead_ii,
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

    for i in range(min(len(peaks), len(onsets), len(offsets))):

        onset = onsets[i]
        offset = offsets[i]

        if (
            np.isfinite(onset)
            and np.isfinite(offset)
            and 0 <= onset < offset < len(lead_ii)
        ):

            durations[i] = (
                offset - onset
            ) * 1000 / SAMPLING_RATE

    return peaks, durations


# ==========================================
# ONE-TO-ONE R-PEAK MATCHING
# ==========================================

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
            matches.append((oi, ai, distance))

    return matches, used


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():

    labels = pd.read_csv(LABEL_FILE)
    metadata = pd.read_csv(METADATA_FILE)

    data = labels.merge(
        metadata[["ecg_id", "filename_hr"]],
        on="ecg_id",
        how="left",
        validate="one_to_one"
    )

    # Same eligible records and official training folds
    data = data[
        (data["strat_fold"] <= 8) &
        (data[["NORM", "MI", "STTC", "CD", "HYP"]].sum(axis=1) > 0)
    ]

    data = data.head(NUM_RECORDS).reset_index(drop=True)

    results = []

    print("500 Hz MORPHOLOGY EVALUATION")
    print("=" * 55)
    print("Records:", len(data))
    print("Sampling rate:", SAMPLING_RATE, "Hz")

    for index, row in data.iterrows():

        try:

            original = load_record(row["filename_hr"])

            original_peaks, original_qrs = analyze_ecg(
                original
            )

            for technique in TECHNIQUES:

                for strength in STRENGTHS:

                    augmented = augment(
                        original.copy(),
                        technique,
                        strength
                    )

                    augmented_peaks, augmented_qrs = analyze_ecg(
                        augmented
                    )

                    matches, used = match_peaks(
                        original_peaks,
                        augmented_peaks
                    )

                    timing = []
                    qrs_diff = []
                    amplitude_diff = []

                    for oi, ai, distance in matches:

                        timing.append(
                            distance * 1000 / SAMPLING_RATE
                        )

                        if (
                            np.isfinite(original_qrs[oi])
                            and np.isfinite(augmented_qrs[ai])
                        ):

                            qrs_diff.append(
                                abs(
                                    original_qrs[oi]
                                    - augmented_qrs[ai]
                                )
                            )

                        amplitude_diff.append(
                            abs(
                                original[1, original_peaks[oi]]
                                - augmented[1, augmented_peaks[ai]]
                            )
                        )

                    valid_original_qrs = np.isfinite(
                        original_qrs
                    ).sum()

                    valid_augmented_qrs = np.isfinite(
                        augmented_qrs
                    ).sum()

                    results.append({

                        "ECG_ID": row["ecg_id"],
                        "Technique": technique,
                        "Strength": strength,

                        "Original_Peaks": len(original_peaks),
                        "Augmented_Peaks": len(augmented_peaks),
                        "Matched_Peaks": len(matches),

                        "Match_Rate_Percent":
                            len(matches) / len(original_peaks) * 100
                            if len(original_peaks) else np.nan,

                        "Original_Valid_QRS": valid_original_qrs,
                        "Augmented_Valid_QRS": valid_augmented_qrs,

                        "Valid_Matched_QRS": len(qrs_diff),

                        "Mean_Timing_Difference_ms":
                            np.mean(timing) if timing else np.nan,

                        "Mean_QRS_Difference_ms":
                            np.mean(qrs_diff) if qrs_diff else np.nan,

                        "Median_QRS_Difference_ms":
                            np.median(qrs_diff) if qrs_diff else np.nan,

                        "Mean_R_Amplitude_Difference":
                            np.mean(amplitude_diff)
                            if amplitude_diff else np.nan,

                        "MAE": np.mean(np.abs(original - augmented)),

                        "RMSE": np.sqrt(
                            np.mean((original - augmented) ** 2)
                        ),

                        "Correlation": np.corrcoef(
                            original.flatten(),
                            augmented.flatten()
                        )[0, 1]
                    })

        except Exception as error:

            print(
                f"ECG {row['ecg_id']} failed: {error}"
            )

        print(
            f"Completed {index + 1}/{len(data)}"
        )

    # ======================================
    # SAVE RESULTS
    # ======================================

    Path("results").mkdir(exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty:
        print("No results generated.")
        return

    df.to_csv(
        "results/morphology_500hz_detailed.csv",
        index=False
    )

    metrics = [
        "Match_Rate_Percent",
        "Original_Valid_QRS",
        "Augmented_Valid_QRS",
        "Valid_Matched_QRS",
        "Mean_Timing_Difference_ms",
        "Mean_QRS_Difference_ms",
        "Median_QRS_Difference_ms",
        "Mean_R_Amplitude_Difference",
        "MAE",
        "RMSE",
        "Correlation"
    ]

    summary = df.groupby(
        ["Technique", "Strength"]
    )[metrics].mean().reset_index()

    summary.to_csv(
        "results/morphology_500hz_summary.csv",
        index=False
    )

    print("\n500 Hz MORPHOLOGY SUMMARY")
    print("=" * 100)
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print("results/morphology_500hz_detailed.csv")
    print("results/morphology_500hz_summary.csv")


if __name__ == "__main__":
    main()
