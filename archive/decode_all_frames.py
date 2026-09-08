"""
Decode all 103 frames and save the results to CSV.
"""

from pathlib import Path
import csv

import cv2

from .segment_decoder import decode_text


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

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            "No BMP images found in data/text/"
        )

    results = []

    print(
        f"Found {len(images)} BMP images."
    )

    print()
    print("=" * 80)
    print("16-SEGMENT DECODER — ALL FRAMES")
    print("=" * 80)

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Frame {index:3d}: "
                f"ERROR"
            )

            continue

        text = decode_text(
            image
        )

        results.append(
            {
                "frame": index,
                "filename": image_path.name,
                "text": text,
            }
        )

        print(
            f"Frame {index:3d} | "
            f"{image_path.name} | "
            f"'{text}'"
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "segment_decoder_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "filename",
                "text",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print(
        f"Results saved to:"
    )

    print(
        csv_path
    )


if __name__ == "__main__":

    main()