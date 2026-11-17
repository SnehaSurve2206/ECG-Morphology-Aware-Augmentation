
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import ECGDataset
from augmentations import apply_augmentation

# Load one real training ECG
dataset = ECGDataset("train")
signal, labels = dataset[0]

# Use Lead II for visual comparison
original = signal[1].numpy()

techniques = ["noise", "scaling", "shifting", "warping"]
strengths = ["mild", "moderate", "strong"]

fig, axes = plt.subplots(4, 3, figsize=(16, 12))

for row, technique in enumerate(techniques):
    for col, strength in enumerate(strengths):

        augmented = apply_augmentation(
            signal.clone(), technique, strength
        )

        axes[row, col].plot(
            original, label="Original", linewidth=1
        )

        axes[row, col].plot(
            augmented[1].numpy(),
            label="Augmented",
            alpha=0.75,
            linewidth=1
        )

        axes[row, col].set_title(
            f"{technique.title()} - {strength.title()}"
        )
        axes[row, col].set_xlabel("Samples")
        axes[row, col].set_ylabel("Amplitude")
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(alpha=0.2)

plt.tight_layout()

Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/augmentation_comparison.png",
    dpi=200
)

plt.show()

print("Saved: results/augmentation_comparison.png")
