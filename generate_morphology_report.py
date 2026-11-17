
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "results/morphology_500hz_detailed.csv"
OUTPUT_DIR = Path("results/morphology_report")

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42

TECHNIQUES = ["noise", "scaling", "shifting", "warping"]
STRENGTHS = ["mild", "moderate", "strong"]

np.random.seed(RANDOM_SEED)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    df = pd.read_csv(INPUT_FILE)

    required = [
        "ECG_ID",
        "Technique",
        "Strength",
        "Match_Rate_Percent",
        "Valid_Matched_QRS",
        "Matched_Peaks",
        "Original_Peaks",
        "Original_Valid_QRS",
        "Augmented_Valid_QRS",
        "Mean_Timing_Difference_ms",
        "Mean_QRS_Difference_ms",
        "Median_QRS_Difference_ms",
        "Mean_R_Amplitude_Difference",
        "MAE",
        "RMSE",
        "Correlation"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ==========================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ==========================================

def bootstrap_ci(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(RANDOM_SEED)

    samples = rng.choice(
        values,
        size=(BOOTSTRAP_ITERATIONS, len(values)),
        replace=True
    )

    bootstrap_means = samples.mean(axis=1)

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return lower, upper


# ==========================================
# ADD COVERAGE METRICS
# ==========================================

def add_coverage_metrics(df):

    df = df.copy()

    df["R_Peak_Coverage_Percent"] = (
        df["Matched_Peaks"]
        / df["Original_Peaks"].replace(0, np.nan)
    ) * 100

    df["Original_QRS_Coverage_Percent"] = (
        df["Original_Valid_QRS"]
        / df["Original_Peaks"].replace(0, np.nan)
    ) * 100

    df["Augmented_QRS_Coverage_Percent"] = (
        df["Augmented_Valid_QRS"]
        / df["Augmented_Peaks"].replace(0, np.nan)
    ) * 100

    df["Matched_QRS_Coverage_Percent"] = (
        df["Valid_Matched_QRS"]
        / df["Matched_Peaks"].replace(0, np.nan)
    ) * 100

    return df


# ==========================================
# STATISTICAL SUMMARY
# ==========================================

def calculate_summary(df):

    metrics = [
        "Match_Rate_Percent",
        "R_Peak_Coverage_Percent",
        "Original_QRS_Coverage_Percent",
        "Augmented_QRS_Coverage_Percent",
        "Matched_QRS_Coverage_Percent",
        "Mean_Timing_Difference_ms",
        "Mean_QRS_Difference_ms",
        "Median_QRS_Difference_ms",
        "Mean_R_Amplitude_Difference",
        "MAE",
        "RMSE",
        "Correlation"
    ]

    rows = []

    for technique in TECHNIQUES:

        for strength in STRENGTHS:

            group = df[
                (df["Technique"] == technique)
                & (df["Strength"] == strength)
            ]

            for metric in metrics:

                values = group[metric].dropna().values

                if len(values) == 0:
                    continue

                lower, upper = bootstrap_ci(values)

                rows.append({

                    "Technique": technique,
                    "Strength": strength,
                    "Metric": metric,

                    "Valid_Records": len(values),

                    "Mean": np.mean(values),
                    "Median": np.median(values),
                    "Std": np.std(values, ddof=1)
                    if len(values) > 1 else np.nan,

                    "CI_95_Lower": lower,
                    "CI_95_Upper": upper

                })

    return pd.DataFrame(rows)


# ==========================================
# PLOTTING FUNCTION
# ==========================================

def create_plot(
    df,
    metric,
    title,
    ylabel,
    filename
):

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(STRENGTHS))
    width = 0.18

    for index, technique in enumerate(TECHNIQUES):

        means = []
        lower_errors = []
        upper_errors = []

        for strength in STRENGTHS:

            group = df[
                (df["Technique"] == technique)
                & (df["Strength"] == strength)
            ]

            values = group[metric].dropna().values

            if len(values) == 0:

                means.append(np.nan)
                lower_errors.append(0)
                upper_errors.append(0)

                continue

            mean = np.mean(values)
            lower, upper = bootstrap_ci(values)

            means.append(mean)
            lower_errors.append(max(0, mean - lower))
            upper_errors.append(max(0, upper - mean))

        position = x + (
            index - (len(TECHNIQUES) - 1) / 2
        ) * width

        ax.bar(
            position,
            means,
            width,
            yerr=[lower_errors, upper_errors],
            capsize=3,
            label=technique
        )

    ax.set_title(title)
    ax.set_xlabel("Augmentation intensity")
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(STRENGTHS)

    ax.legend(title="Technique")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# GENERATE REPORT
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    df = add_coverage_metrics(df)

    print("500 Hz MORPHOLOGY ANALYSIS")
    print("=" * 60)

    print("Total experimental observations:", len(df))
    print("Unique ECG records:", df["ECG_ID"].nunique())

    # Detailed results with coverage
    df.to_csv(
        OUTPUT_DIR / "detailed_results.csv",
        index=False
    )

    # Statistical summary
    summary = calculate_summary(df)

    summary.to_csv(
        OUTPUT_DIR / "statistical_summary.csv",
        index=False
    )

    # ======================================
    # GENERATE FOUR GRAPHS
    # ======================================

    create_plot(
        df,
        "Match_Rate_Percent",
        "R-Peak Matching Rate",
        "Matching rate (%)",
        "01_rpeak_matching.png"
    )

    create_plot(
        df,
        "Mean_QRS_Difference_ms",
        "QRS Duration Difference",
        "Absolute difference (ms)",
        "02_qrs_difference.png"
    )

    create_plot(
        df,
        "Correlation",
        "ECG Signal Correlation",
        "Pearson correlation",
        "03_signal_correlation.png"
    )

    create_plot(
        df,
        "Matched_QRS_Coverage_Percent",
        "Valid Matched QRS Coverage",
        "Coverage (%)",
        "04_qrs_coverage.png"
    )

    # ======================================
    # PRINT RESULTS
    # ======================================

    print("\nKEY RESULTS")
    print("=" * 60)

    important_metrics = [
        "Match_Rate_Percent",
        "Matched_QRS_Coverage_Percent",
        "Mean_QRS_Difference_ms",
        "Correlation"
    ]

    for metric in important_metrics:

        print(f"\n{metric}")

        selected = summary[
            summary["Metric"] == metric
        ]

        print(
            selected[
                [
                    "Technique",
                    "Strength",
                    "Valid_Records",
                    "Mean",
                    "Std",
                    "CI_95_Lower",
                    "CI_95_Upper"
                ]
            ].round(4).to_string(index=False)
        )

    print("\nREPORT GENERATED SUCCESSFULLY")

    print("\nOutput folder:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
