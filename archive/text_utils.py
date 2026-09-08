"""
Basic utilities for processing the 16-segment display.
"""

from pathlib import Path

import cv2
import numpy as np


def load_display_image(path: str | Path) -> np.ndarray:
    """
    Load a BMP display image.
    """

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


def preprocess_display(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert the display image into a binary representation.

    The LED segments are bright blue/cyan, while the background
    is dark.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hue, saturation, value = cv2.split(hsv)

    # Bright, saturated pixels correspond to the LED segments.
    mask = (
        (saturation > 80)
        & (value > 80)
    ).astype(
        np.uint8
    ) * 255

    # Remove small isolated noise.
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


def save_preprocessed(
    image_path: str | Path,
    output_path: str | Path,
):
    """
    Save the thresholded display image.
    """

    image = load_display_image(
        image_path
    )

    mask = preprocess_display(
        image
    )

    cv2.imwrite(
        str(output_path),
        mask
    )