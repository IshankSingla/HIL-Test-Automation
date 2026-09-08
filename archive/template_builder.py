"""
Method 2: Image Template Builder

This method recognizes characters using their actual image
appearance rather than their 16-segment ON/OFF pattern.

Method 1:
    Segment state -> character

Method 2:
    Character image -> template similarity
"""

from pathlib import Path

import cv2
import numpy as np


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
# DISPLAY INFORMATION
# ============================================================

IMAGE_WIDTH = 825
IMAGE_HEIGHT = 86

# Nominal character pitch from the supplied display.
CHARACTER_PITCH = 61

# Approximate character width.
CHARACTER_WIDTH = 55


# ============================================================
# KNOWN MESSAGES
# ============================================================

MESSAGES = [
    "WATER IN FLUE",
    "AIR FILTER IS BLOCKED",
    "PLEASE CONTACT JCB DEALER",
]


# ============================================================
# REQUIRED CHARACTERS
# ============================================================

REQUIRED_CHARACTERS = sorted(
    {
        character
        for message in MESSAGES
        for character in message
        if character != " "
    }
)


# ============================================================
# DISPLAY MASK
# ============================================================

def create_display_mask(image):
    """
    Convert the display image into a binary LED mask.

    The supplied display uses bright saturated blue/cyan
    illuminated segments on a dark background.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(hsv)

    mask = (
        (saturation > 80)
        & (value > 80)
    ).astype(np.uint8) * 255

    return mask


# ============================================================
# CHARACTER IMAGE EXTRACTION
# ============================================================

def extract_character_image(
    mask,
    center_x,
):
    """
    Extract a character image around a predicted character
    center position.

    center_x is allowed to be fractional because the display
    scrolls continuously.
    """

    x1 = int(round(
        center_x - CHARACTER_WIDTH / 2
    ))

    x2 = x1 + CHARACTER_WIDTH

    # Character must be sufficiently inside the image.
    if x1 < 0 or x2 > mask.shape[1]:
        return None

    return mask[
        0:IMAGE_HEIGHT,
        x1:x2
    ]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_character(
    character_image,
):
    """
    Convert a character image into a normalized template.

    The active pixels are cropped to their bounding box and
    then padded and resized to a fixed size.
    """

    if (
        character_image is None
        or character_image.size == 0
    ):
        return None

    points = cv2.findNonZero(
        character_image
    )

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(
        points
    )

    # Reject extremely small fragments.
    if w < 5 or h < 5:
        return None

    cropped = character_image[
        y:y + h,
        x:x + w
    ]

    # Add padding around the character.
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

    # Standard template size.
    normalized = cv2.resize(
        cropped,
        (50, 76),
        interpolation=cv2.INTER_NEAREST,
    )

    return normalized


# ============================================================
# TEMPLATE QUALITY
# ============================================================

def template_quality(
    character_image,
):
    """
    Estimate how complete a character is.

    A fully visible character normally contains more
    illuminated pixels than a clipped edge character.

    This score is only used for selecting templates.
    """

    if character_image is None:
        return 0

    return cv2.countNonZero(
        character_image
    )


# ============================================================
# SAVE TEMPLATE
# ============================================================

def save_template(
    character,
    template,
    frame_number,
    quality,
):
    """
    Save one template image.
    """

    TEMPLATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TEMPLATE_DIR
        / f"{character}.png"
    )

    cv2.imwrite(
        str(output_path),
        template,
    )

    print(
        f"Saved template: "
        f"{character} "
        f"(frame {frame_number}, "
        f"quality={quality})"
    )


# ============================================================
# FIND MESSAGE ALIGNMENT
# ============================================================

def find_message_alignments(
    mask,
    message,
):
    """
    Estimate possible horizontal positions of a known message.

    The display scrolls horizontally, so the first character
    is not assumed to start at a fixed CELL_START position.

    We try multiple possible positions for the first character.
    """

    alignments = []

    # --------------------------------------------------------
    # Search the possible center position of the first
    # character.
    #
    # The display is 825 pixels wide and the character pitch
    # is approximately 61 pixels.
    # --------------------------------------------------------

    max_start = IMAGE_WIDTH - CHARACTER_WIDTH

    for first_center in np.arange(
        CHARACTER_WIDTH / 2,
        max_start,
        1.0,
    ):

        centers = []

        for index in range(
            len(message)
        ):

            center = (
                first_center
                + index * CHARACTER_PITCH
            )

            centers.append(
                center
            )

        # ----------------------------------------------------
        # Only keep alignments where at least some characters
        # are visible.
        # ----------------------------------------------------

        visible = 0

        for center in centers:

            x1 = int(
                round(
                    center
                    - CHARACTER_WIDTH / 2
                )
            )

            x2 = x1 + CHARACTER_WIDTH

            if (
                x1 >= 0
                and x2 <= IMAGE_WIDTH
            ):
                visible += 1

        if visible >= 3:

            alignments.append(
                (
                    first_center,
                    centers,
                )
            )

    return alignments


# ============================================================
# SCORE ALIGNMENT
# ============================================================

def score_alignment(
    mask,
    message,
    centers,
):
    """
    Score how well a known message alignment fits the image.

    For each expected non-space character we check whether
    the corresponding image cell contains illuminated pixels.

    Spaces are expected to be mostly dark.
    """

    score = 0.0
    count = 0

    for character, center in zip(
        message,
        centers,
    ):

        cell = extract_character_image(
            mask,
            center,
        )

        if cell is None:
            continue

        illuminated = (
            cv2.countNonZero(cell)
        )

        if character == " ":

            # A space should be relatively dark.
            if illuminated < 300:
                score += 1.0
            else:
                score += 0.0

        else:

            # A real character should contain
            # illuminated pixels.
            if illuminated > 80:
                score += 1.0

        count += 1

    if count == 0:
        return 0.0

    return score / count


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("METHOD 2 — IMAGE TEMPLATE BUILDER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load all frames
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
        "Required characters:"
    )

    print(
        " ".join(
            REQUIRED_CHARACTERS
        )
    )

    print()

    # --------------------------------------------------------
    # Candidate templates.
    #
    # Each character will have multiple real examples.
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

        mask = create_display_mask(
            image
        )

        # ----------------------------------------------------
        # Try every known message.
        # ----------------------------------------------------

        for message in MESSAGES:

            alignments = (
                find_message_alignments(
                    mask,
                    message,
                )
            )

            # ------------------------------------------------
            # Find the best alignment for this frame/message.
            # ------------------------------------------------

            best_alignment = None
            best_score = 0.0

            for (
                first_center,
                centers,
            ) in alignments:

                score = score_alignment(
                    mask,
                    message,
                    centers,
                )

                if score > best_score:

                    best_score = score
                    best_alignment = (
                        first_center,
                        centers,
                    )

            if best_alignment is None:
                continue

            first_center, centers = (
                best_alignment
            )

            # ------------------------------------------------
            # Only accept a reasonably good alignment.
            # ------------------------------------------------

            if best_score < 0.55:
                continue

            # ------------------------------------------------
            # Extract character images from this alignment.
            # ------------------------------------------------

            for character, center in zip(
                message,
                centers,
            ):

                if character == " ":
                    continue

                cell = extract_character_image(
                    mask,
                    center,
                )

                normalized = normalize_character(
                    cell
                )

                if normalized is None:
                    continue

                quality = template_quality(
                    cell
                )

                # Ignore tiny/weak fragments.
                if quality < 80:
                    continue

                candidates[
                    character
                ].append(
                    (
                        normalized,
                        frame_number,
                        quality,
                        best_score,
                    )
                )

    # ========================================================
    # SELECT BEST TEMPLATE FOR EACH CHARACTER
    # ========================================================

    print()
    print("=" * 70)
    print("SELECTING BEST CHARACTER TEMPLATES")
    print("=" * 70)
    print()

    TEMPLATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_count = 0

    for character in REQUIRED_CHARACTERS:

        character_candidates = (
            candidates[
                character
            ]
        )

        if not character_candidates:

            print(
                f"WARNING: No template found "
                f"for '{character}'"
            )

            continue

        # ----------------------------------------------------
        # Prefer candidates with:
        #
        # 1. Strong character pixels
        # 2. Good message alignment
        #
        # The quality score is the primary criterion.
        # ----------------------------------------------------

        best = max(
            character_candidates,
            key=lambda item: (
                item[2],
                item[3],
            ),
        )

        template = best[0]
        frame_number = best[1]
        quality = best[2]

        save_template(
            character,
            template,
            frame_number,
            quality,
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
        "Output directory:"
    )

    print(
        TEMPLATE_DIR
    )

    print()

    if saved_count == len(
        REQUIRED_CHARACTERS
    ):

        print(
            "SUCCESS: A template was created "
            "for every required character."
        )

    else:

        print(
            "WARNING: Some characters are still "
            "missing. We will inspect them before "
            "building the decoder."
        )

    print()


if __name__ == "__main__":
    main()