"""
Method 2: Build Character Templates

The existing Method 1 decoder is used ONLY to identify
which character is present in each real display cell.

After the character is identified, we save the actual
image appearance of that character.

IMPORTANT:
Method 2 recognition itself will NOT use the 16-segment
patterns. It will use image/template similarity.
"""

from pathlib import Path

import cv2
import numpy as np

from segment_decoder import (
    CELL_STARTS,
    CELL_WIDTH,
    decode_frame,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = PROJECT_ROOT / "data" / "text"

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "templates"
)


# ============================================================
# TEMPLATE SIZE
# ============================================================

TEMPLATE_WIDTH = 50
TEMPLATE_HEIGHT = 76


# ============================================================
# REQUIRED CHARACTERS
# ============================================================

REQUIRED_CHARACTERS = sorted(
    {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "I",
        "J",
        "K",
        "L",
        "N",
        "O",
        "P",
        "R",
        "S",
        "T",
        "U",
        "W",
    }
)


# ============================================================
# DISPLAY MASK
# ============================================================

def create_display_mask(image):
    """
    Create a binary image containing the illuminated display
    pixels.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    _, saturation, value = cv2.split(hsv)

    mask = (
        (saturation > 80)
        & (value > 80)
    ).astype(np.uint8) * 255

    return mask


# ============================================================
# NORMALIZE CHARACTER IMAGE
# ============================================================

def normalize_character(cell):
    """
    Convert a real character cell into a normalized image.

    The illuminated pixels are cropped to their bounding box,
    padded, and resized to a common template size.
    """

    if cell is None or cell.size == 0:
        return None

    points = cv2.findNonZero(cell)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    # Reject very small/noisy regions.
    if w < 5 or h < 5:
        return None

    cropped = cell[
        y:y + h,
        x:x + w
    ]

    padding = 5

    cropped = cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    normalized = cv2.resize(
        cropped,
        (
            TEMPLATE_WIDTH,
            TEMPLATE_HEIGHT,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return normalized


# ============================================================
# TEMPLATE QUALITY
# ============================================================

def calculate_quality(cell):
    """
    Estimate how much of the character is illuminated.

    Higher values generally mean a more complete character.
    """

    if cell is None:
        return 0

    return cv2.countNonZero(cell)


# ============================================================
# EXTRACT CELL
# ============================================================

def extract_cell(mask, cell_x):
    """
    Extract one of the 13 display cells using the exact
    geometry already validated in Method 1.
    """

    x1 = int(cell_x)

    x2 = min(
        x1 + CELL_WIDTH,
        mask.shape[1],
    )

    if x1 >= mask.shape[1]:
        return None

    cell = mask[
        0:86,
        x1:x2,
    ]

    return cell


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("METHOD 2 — CHARACTER TEMPLATE BUILDER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load all images.
    # --------------------------------------------------------

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            f"No BMP files found in {TEXT_DIR}"
        )

    print(
        f"Found {len(images)} BMP images."
    )

    print()

    print(
        "Using Method 1 only to identify "
        "the character labels."
    )

    print()

    # --------------------------------------------------------
    # Store candidate templates.
    #
    # character -> list of:
    #
    # (normalized_image, frame_number, cell_number, quality)
    # --------------------------------------------------------

    candidates = {
        character: []
        for character in REQUIRED_CHARACTERS
    }

    # --------------------------------------------------------
    # Process every frame.
    # --------------------------------------------------------

    for frame_number, image_path in enumerate(
        images,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not read "
                f"{image_path.name}"
            )

            continue

        # ----------------------------------------------------
        # Method 1 identifies characters.
        # ----------------------------------------------------

        results = decode_frame(
            image,
            threshold=0.55,
        )

        # ----------------------------------------------------
        # Create the binary image used to obtain the actual
        # character pixels.
        # ----------------------------------------------------

        mask = create_display_mask(
            image
        )

        # ----------------------------------------------------
        # Examine each of the 13 cells.
        # ----------------------------------------------------

        for cell_number, result in enumerate(
            results
        ):

            character = result.character

            # We only need the required characters.
            if character not in candidates:
                continue

            cell_x = CELL_STARTS[
                cell_number
            ]

            cell = extract_cell(
                mask,
                cell_x
            )

            if cell is None:
                continue

            quality = calculate_quality(
                cell
            )

            # Ignore very weak/clipped cells.
            if quality < 500:
                continue

            normalized = normalize_character(
                cell
            )

            if normalized is None:
                continue

            candidates[
                character
            ].append(
                (
                    normalized,
                    frame_number,
                    cell_number,
                    quality,
                    result.confidence,
                )
            )

    # ========================================================
    # SELECT BEST TEMPLATE
    # ========================================================

    print()
    print("=" * 70)
    print("SELECTING BEST TEMPLATES")
    print("=" * 70)
    print()

    TEMPLATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for character in REQUIRED_CHARACTERS:

        character_candidates = candidates[
            character
        ]

        if not character_candidates:

            print(
                f"WARNING: No template found "
                f"for '{character}'"
            )

            continue

        # ----------------------------------------------------
        # Prefer:
        #
        # 1. high pixel count
        # 2. high Method 1 confidence
        #
        # This helps avoid clipped edge characters.
        # ----------------------------------------------------

        best = max(
            character_candidates,
            key=lambda item: (
                item[3],
                item[4],
            ),
        )

        (
            template,
            frame_number,
            cell_number,
            quality,
            confidence,
        ) = best

        output_path = (
            TEMPLATE_DIR
            / f"{character}.png"
        )

        cv2.imwrite(
            str(output_path),
            template,
        )

        print(
            f"Saved {character}.png "
            f"| frame={frame_number} "
            f"| cell={cell_number + 1} "
            f"| quality={quality} "
            f"| confidence={confidence:.2f}"
        )

        saved_count += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TEMPLATE BUILD COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Templates created: "
        f"{saved_count} / "
        f"{len(REQUIRED_CHARACTERS)}"
    )

    print()

    print(
        f"Templates saved in:"
    )

    print(
        TEMPLATE_DIR
    )

    print()

    if saved_count == len(
        REQUIRED_CHARACTERS
    ):

        print(
            "SUCCESS: All required character "
            "templates were created."
        )

    else:

        missing = [
            character
            for character in REQUIRED_CHARACTERS
            if not (
                TEMPLATE_DIR
                / f"{character}.png"
            ).exists()
        ]

        print(
            "Missing:"
        )

        print(
            " ".join(missing)
        )

    print()


if __name__ == "__main__":
    main()