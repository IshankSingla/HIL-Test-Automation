"""
Needle gauge calibration.

The assignment provides five reference images with known positions:
0, 900, 1500, 1900 and 3000.
"""

from pathlib import Path

import cv2

from .needle_gauge import (
    detect_needle_angle,
    build_calibration,
)


# Project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "needle_gauge"
)


# Ground-truth values supplied by the assignment.
REFERENCE_IMAGES = [
    (
        "1.59592113294115.jpg",
        0
    ),
    (
        "895.590073713866.jpg",
        900
    ),
    (
        "1463.jpg",
        1500
    ),
    (
        "1880.jpg",
        1900
    ),
    (
        "2984.jpg",
        3000
    ),
]


def create_gauge_calibration():
    """
    Detect the angle from each reference image and create
    the angle -> gauge-value mapping.
    """

    angles = []
    values = []

    print("Calibration data")
    print("-" * 60)

    for filename, expected_value in REFERENCE_IMAGES:

        path = IMAGE_DIR / filename

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {path}"
            )

        angle, _ = detect_needle_angle(
            image
        )

        angles.append(angle)
        values.append(expected_value)

        print(
            f"{filename:28s} "
            f"angle={angle:8.2f}° "
            f"value={expected_value}"
        )

    mapper = build_calibration(
        values,
        angles
    )

    return mapper


if __name__ == "__main__":

    create_gauge_calibration()