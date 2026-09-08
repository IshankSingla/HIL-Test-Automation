"""
Analyze the geometry of the scrolling 16-segment display.

The purpose of this script is to determine:

1. Display dimensions
2. Active text area
3. Horizontal scrolling movement
4. Approximate character pitch
5. Blank frames
"""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "text"
)


def get_active_bounds(image):
    """
    Find the horizontal bounds of illuminated pixels.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hue, saturation, value = cv2.split(hsv)

    mask = (
        (saturation > 80)
        & (value > 80)
    )

    column_counts = np.sum(
        mask,
        axis=0
    )

    active_columns = np.where(
        column_counts > 2
    )[0]

    if len(active_columns) == 0:
        return None

    return (
        int(active_columns[0]),
        int(active_columns[-1])
    )


def main():

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            "No BMP images found in data/text/"
        )

    print(
        f"Total BMP images: {len(images)}"
    )

    print(
        f"Image size: "
        f"{cv2.imread(str(images[0])).shape}"
    )

    print()
    print("=" * 75)
    print("DISPLAY GEOMETRY ANALYSIS")
    print("=" * 75)

    bounds = []

    for index, image_path in enumerate(
        images,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        result = get_active_bounds(
            image
        )

        bounds.append(result)

        if result is None:

            print(
                f"Frame {index:3d} "
                f"{image_path.name} "
                f"-> BLANK"
            )

        else:

            x1, x2 = result

            print(
                f"Frame {index:3d} "
                f"{image_path.name} "
                f"-> "
                f"x={x1:3d} to {x2:3d}"
            )

    # ---------------------------------------------------------
    # Calculate movement between consecutive non-blank frames.
    # ---------------------------------------------------------

    movements = []

    previous_x = None

    for result in bounds:

        if result is None:

            previous_x = None
            continue

        current_x = result[0]

        if previous_x is not None:

            movement = (
                current_x
                - previous_x
            )

            movements.append(
                movement
            )

        previous_x = current_x

    print()
    print("=" * 75)
    print("SCROLLING ANALYSIS")
    print("=" * 75)

    if movements:

        movements_array = np.asarray(
            movements,
            dtype=float
        )

        print(
            "Measured horizontal movements:"
        )

        print(
            movements_array
        )

        print()

        print(
            "Median movement: "
            f"{np.median(movements_array):.2f} pixels"
        )

        print(
            "Mean movement: "
            f"{np.mean(movements_array):.2f} pixels"
        )

        print(
            "Minimum movement: "
            f"{np.min(movements_array):.2f} pixels"
        )

        print(
            "Maximum movement: "
            f"{np.max(movements_array):.2f} pixels"
        )

    # ---------------------------------------------------------
    # Identify blank frame numbers.
    # ---------------------------------------------------------

    blank_frames = [
        index + 1
        for index, result in enumerate(bounds)
        if result is None
    ]

    print()
    print("=" * 75)
    print("BLANK FRAMES")
    print("=" * 75)

    print(blank_frames)


if __name__ == "__main__":

    main()