"""
Test the custom 16-segment decoder on a known frame.
"""

from pathlib import Path

import cv2

from .segment_decoder import (
    decode_frame,
    decode_text,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "text"
)


def main():

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            "No BMP images found."
        )

    # Frame 40 is known to contain:
    #
    # ER IS BLOCKED
    #
    # It is a good validation frame because it contains
    # many different characters.

    image_path = images[39]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read {image_path}"
        )

    results = decode_frame(
        image
    )

    print()
    print("=" * 70)
    print("16-SEGMENT DECODER TEST")
    print("=" * 70)

    print()
    print(
        f"Image: {image_path.name}"
    )

    print()

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"Cell {index:2d}: "
            f"'{result.character}' "
            f"confidence={result.confidence:.2f}"
        )

    print()
    print(
        "Decoded text:"
    )

    print(
        decode_text(image)
    )


if __name__ == "__main__":
    main()