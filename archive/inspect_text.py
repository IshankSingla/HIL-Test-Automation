"""
Inspect one 16-segment display frame.
"""

from pathlib import Path

import cv2

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "text"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Get the first BMP file.
    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:
        raise RuntimeError(
            "No BMP images found in data/text/"
        )

    image_path = images[0]

    print(
        f"Inspecting: {image_path.name}"
    )

    image = load_display_image(
        image_path
    )

    mask = preprocess_display(
        image
    )

    output_path = (
        OUTPUT_DIR
        / "preprocessed_first_frame.png"
    )

    cv2.imwrite(
        str(output_path),
        mask
    )

    print(
        f"Original shape: {image.shape}"
    )

    print(
        f"Preprocessed image saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()