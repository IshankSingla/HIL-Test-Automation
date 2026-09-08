"""
Inspect the actual 16-segment patterns of characters.

We inspect W, P and J at the character-cell positions
where they are actually visible in the supplied frames.
"""

from pathlib import Path

import cv2

from .segment_decoder import (
    create_display_mask,
    get_segment_scores,
    scores_to_pattern,
    SEGMENT_NAMES,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "text"
)


# ------------------------------------------------------------
# CASES
#
# frame number
# expected character
# character-cell x position
#
# W:
# Frame 1 starts around x=696, so W is at the rightmost
# character position around x=694.
#
# P:
# Frame 51 starts around x=757, so P is around x=756.
#
# J:
# Frame 68 contains approximately:
#
# CONTACT JCB
#
# Character positions are approximately:
#
# C -> 18
# O -> 80
# N -> 141
# T -> 202
# A -> 264
# C -> 326
# T -> 387
# space -> 448
# J -> 510
# C -> 572
# B -> 633
# ------------------------------------------------------------

CASES = [
    (1, "W", 694),
    (51, "P", 756),
    (68, "J", 572),
]


def inspect_character(
    frame_number: int,
    expected_character: str,
    cell_x: int,
):

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            "No BMP files found in data/text/"
        )

    image_path = images[
        frame_number - 1
    ]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read {image_path}"
        )

    mask = create_display_mask(
        image
    )

    scores = get_segment_scores(
        mask,
        cell_x,
    )

    pattern = scores_to_pattern(
        scores,
        threshold=0.55,
    )

    print()
    print("=" * 75)

    print(
        f"FRAME {frame_number}"
    )

    print(
        f"EXPECTED CHARACTER: {expected_character}"
    )

    print(
        f"CELL X POSITION: {cell_x}"
    )

    print("=" * 75)

    print()

    print(
        f"Image: {image_path.name}"
    )

    print()

    print(
        "Segment scores:"
    )

    print()

    for name, score in zip(
        SEGMENT_NAMES,
        scores,
    ):

        state = (
            "ON"
            if score >= 0.55
            else "OFF"
        )

        print(
            f"{name:3s} "
            f"score={score:.3f} "
            f"{state}"
        )

    print()

    print(
        "16-segment pattern:"
    )

    print(
        "".join(
            str(value)
            for value in pattern
        )
    )


def main():

    for (
        frame_number,
        expected_character,
        cell_x,
    ) in CASES:

        inspect_character(
            frame_number,
            expected_character,
            cell_x,
        )


if __name__ == "__main__":

    main()