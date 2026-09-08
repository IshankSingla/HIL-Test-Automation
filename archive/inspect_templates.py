from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "templates"
)


CHARACTERS = [
    "B",
    "D",
    "J",
    "K",
]


def main():

    images = []

    for character in CHARACTERS:

        path = (
            TEMPLATE_DIR
            / f"{character}.png"
        )

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            print(
                f"Could not load {path}"
            )

            continue

        # Add character label area.
        canvas = np.zeros(
            (100, 80),
            dtype=np.uint8
        )

        # Resize template for easier viewing.
        resized = cv2.resize(
            image,
            (60, 90),
            interpolation=cv2.INTER_NEAREST
        )

        canvas[
            5:95,
            10:70
        ] = resized

        images.append(
            canvas
        )

    if not images:

        print(
            "No templates found."
        )

        return

    comparison = np.hstack(
        images
    )

    output = (
        TEMPLATE_DIR
        / "missing_templates_comparison.png"
    )

    cv2.imwrite(
        str(output),
        comparison
    )

    print()
    print(
        "Created:"
    )
    print(output)
    print()
    print(
        "Order:"
    )
    print(
        "B | D | J | K"
    )


if __name__ == "__main__":
    main()