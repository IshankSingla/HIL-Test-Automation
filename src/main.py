"""
Main program for Needle Gauge detection.
"""

from pathlib import Path

import cv2

from .calibration import (
    create_gauge_calibration,
    REFERENCE_IMAGES,
    IMAGE_DIR,
)

from .needle_gauge import (
    detect_value,
    draw_detection,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "needle_gauge"
)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create calibration using the five
    # supplied reference images.
    mapper = create_gauge_calibration()

    print()
    print("=" * 70)
    print("NEEDLE GAUGE DETECTION")
    print("=" * 70)

    for filename, expected_value in REFERENCE_IMAGES:

        image_path = (
            IMAGE_DIR
            / filename
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"ERROR: Could not read {image_path}"
            )
            continue

        # Detect gauge value.
        result = detect_value(
            image,
            mapper
        )

        error = abs(
            result.value
            - expected_value
        )

        print()
        print(
            f"Image          : {filename}"
        )

        print(
            f"Expected value : {expected_value}"
        )

        print(
            f"Detected value : {result.value:.2f}"
        )

        print(
            f"Needle angle   : "
            f"{result.angle_deg:.2f}°"
        )

        print(
            f"Absolute error : "
            f"{error:.2f}"
        )

        # Create annotated image.
        annotated = draw_detection(
            image,
            result
        )

        output_path = (
            OUTPUT_DIR
            / f"{Path(filename).stem}_detected.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print(
            f"Output saved   : {output_path}"
        )


if __name__ == "__main__":
    main()