
import torch
import torch.nn.functional as F
import random


# Intensity settings
AUGMENTATION_CONFIG = {
    "mild": {
        "noise": 0.01,
        "scale": 0.05,
        "shift": 2,
        "warp": 5
    },
    "moderate": {
        "noise": 0.03,
        "scale": 0.10,
        "shift": 5,
        "warp": 10
    },
    "strong": {
        "noise": 0.05,
        "scale": 0.20,
        "shift": 10,
        "warp": 20
    }
}


def add_gaussian_noise(signal, strength):
    config = AUGMENTATION_CONFIG[strength]
    noise = torch.randn_like(signal) * config["noise"]
    return signal + noise


def amplitude_scaling(signal, strength):
    config = AUGMENTATION_CONFIG[strength]
    amount = config["scale"]

    factor = random.uniform(1 - amount, 1 + amount)
    return signal * factor


def temporal_shifting(signal, strength):
    config = AUGMENTATION_CONFIG[strength]
    max_shift = config["shift"]

    shift = random.randint(-max_shift, max_shift)

    if shift == 0:
        return signal.clone()

    padded = F.pad(
        signal.unsqueeze(0),
        (max_shift, max_shift),
        mode="reflect"
    ).squeeze(0)

    start = max_shift - shift

    return padded[:, start:start + signal.shape[1]]

def temporal_warping(signal, strength):
    config = AUGMENTATION_CONFIG[strength]
    max_displacement = config["warp"]

    channels, length = signal.shape

    # Smooth temporal displacement
    t = torch.linspace(
        0, 2 * torch.pi, length,
        device=signal.device
    )

    amplitude = random.uniform(
        -max_displacement,
        max_displacement
    )

    displacement = amplitude * torch.sin(t)

    positions = torch.arange(
        length, device=signal.device
    ).float() + displacement

    positions = positions.clamp(0, length - 1)

    # Linear interpolation
    left = positions.floor().long()
    right = (left + 1).clamp(max=length - 1)

    weight = (positions - left).unsqueeze(0)

    warped = (
        signal[:, left] * (1 - weight)
        + signal[:, right] * weight
    )

    return warped


def apply_augmentation(signal, technique, strength):
    if technique == "noise":
        return add_gaussian_noise(signal, strength)

    elif technique == "scaling":
        return amplitude_scaling(signal, strength)

    elif technique == "shifting":
        return temporal_shifting(signal, strength)

    elif technique == "warping":
        return temporal_warping(signal, strength)

    else:
        raise ValueError("Unknown augmentation technique")


if __name__ == "__main__":

    sample = torch.randn(12, 1000)

    techniques = [
        "noise",
        "scaling",
        "shifting",
        "warping"
    ]

    strengths = ["mild", "moderate", "strong"]

    print("AUGMENTATION TEST")
    print("=" * 40)

    for technique in techniques:
        for strength in strengths:

            augmented = apply_augmentation(
                sample, technique, strength
            )

            print(
                technique,
                strength,
                augmented.shape,
                torch.isfinite(augmented).all().item()
            )
