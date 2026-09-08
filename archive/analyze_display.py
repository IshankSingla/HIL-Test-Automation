"""
Analyze the geometry of the 16-segment display.

This program examines a preprocessed display frame and finds
the horizontal regions containing illuminated pixels.
"""

from pathlib import Path

import cv2
import numpy as np

from .text_utils import (
    load_display_image,
    preprocess_display,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "text"
)


def find_active_columns(
    binary_image: np.ndarray,
):
    """
    Find horizontal regions containing illuminated pixels.
    """

    # Count white pixels in each column.
    column_counts = np.sum(
        binary_image > 0,
        axis=0
    )

    # A column is considered active when it contains
    # at least a few illuminated pixels.
    active = column_counts > 2

    regions = []

    start = None

    for x, is_active in enumerate(active):

        if is_active and start is None:
            start = x

        elif not is_active and start is not None:

            end = x - 1

            if end - start >= 2:
                regions.append(
                    (start, end)
                )

            start = None

    # Handle a region reaching the right edge.
    if start is not None:

        end = len(active) - 1

        if end - start >= 2:
            regions.append(
                (start, end)
            )

    return regions


def main():

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:
        raise RuntimeError(
            "No BMP images found in data/text/"
        )

    # Analyze the first frame.
    image_path = images[0]

    print(
        f"Analyzing: {image_path.name}"
    )

    image = load_display_image(
        image_path
    )

    binary = preprocess_display(
        image
    )

    regions = find_active_columns(
        binary
    )

    print()
    print("Active horizontal regions")
    print("-" * 60)

    for index, (start, end) in enumerate(
        regions,
        start=1
    ):

        width = end - start + 1

        print(
            f"Region {index:2d}: "
            f"x={start:3d} to {end:3d} "
            f"width={width:3d}"
        )


if __name__ == "__main__":
    main()