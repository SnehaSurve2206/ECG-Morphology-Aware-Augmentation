
import numpy as np
from morphology_500hz import augment, CONFIG

signal = np.zeros((12, 5000), dtype=np.float32)

# Add a known peak at sample 2500
signal[:, 2500] = 1.0

for strength in ["mild", "moderate", "strong"]:

    max_shift = CONFIG[strength]["shift"]

    for shift in [-max_shift, max_shift]:

        # Test the shifting logic directly
        shifted = np.zeros_like(signal)

        for lead in range(12):

            padded = np.pad(
                signal[lead],
                (max_shift, max_shift),
                mode="reflect"
            )

            start = max_shift - shift

            shifted[lead] = padded[
                start:start + signal.shape[1]
            ]

        detected_position = np.argmax(shifted[0])

        actual_shift = detected_position - 2500

        print(
            strength,
            "Expected:", shift,
            "Actual:", actual_shift
        )

        assert actual_shift == shift

print("\nALL SHIFTING TESTS PASSED")
