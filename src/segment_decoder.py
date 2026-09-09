"""
Method 1: 16-Segment Display Decoder

This module detects characters by checking which of the
16 physical display segments are illuminated.

No OCR or machine-learning model is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


# ------------------------------------------------------------
# DISPLAY GEOMETRY
# ------------------------------------------------------------

# Approximate character-cell starting positions measured from
# the supplied 825 x 86 display frames.
#
# There are 13 character positions across the display.

CELL_STARTS = [
    18,
    80,
    141,
    202,
    264,
    326,
    387,
    448,
    510,
    572,
    633,
    694,
    756,
]

# Character width.
CELL_WIDTH = 55

# The segment geometry inside a character cell.
#
# Coordinates are relative to the character-cell origin.
#
# The 16 segments are:
#
# a1, a2  -> top-left / top-right
# b, c    -> upper-right / lower-right
# d1, d2  -> bottom-right / bottom-left
# e, f    -> lower-left / upper-left
# g1, g2  -> middle-left / middle-right
# h, i    -> upper diagonals
# j, k    -> lower diagonals
# l, m    -> upper/lower center vertical
#
# These coordinates were chosen from the actual supplied
# display geometry.

SEGMENT_NAMES = [
    "a1",
    "a2",
    "b",
    "c",
    "d1",
    "d2",
    "e",
    "f",
    "g1",
    "g2",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
]


SEGMENTS = [
    # Top
    ((8, 10), (27, 10)),       # a1
    ((29, 10), (50, 10)),       # a2

    # Right vertical
    ((52, 14), (52, 36)),       # b
    ((52, 46), (52, 70)),       # c

    # Bottom
    ((29, 72), (50, 72)),       # d1
    ((8, 72), (27, 72)),        # d2

    # Left vertical
    ((3, 46), (3, 70)),         # e
    ((3, 14), (3, 36)),         # f

    # Middle
    ((8, 42), (27, 42)),         # g1
    ((29, 42), (50, 42)),        # g2

    # Upper diagonals
    ((8, 14), (27, 36)),         # h
    ((50, 14), (32, 36)),        # i

    # Lower diagonals
    ((8, 70), (27, 46)),         # j
    ((50, 70), (32, 46)),        # k

    # Center vertical
    ((28, 14), (28, 36)),        # l
    ((28, 46), (28, 70)),        # m
]


# ------------------------------------------------------------
# CHARACTER PATTERNS
# ------------------------------------------------------------

# Each character is represented by the segments that should
# be illuminated.
#
# These patterns are based on the actual 16-segment display
# used in the supplied images.
#
# Characters needed by the supplied messages are explicitly
# defined first. Additional common characters are included
# for reuse.

CHARACTER_SEGMENTS = {

    "A": {
        "a1", "a2",
        "b", "c",
        "e", "f",
        "g1", "g2",
    },

    "B": {
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "g2",
        "l", "m",
    },

    "C": {
        "a1", "a2",
        "d1", "d2",
        "e", "f",
    },

    "D": {
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "l", "m",
    },

    "E": {
        "a1", "a2",
        "d1", "d2",
        "e", "f",
        "g1", "g2",
    },

    "F": {
        "a1", "a2",
        "e", "f",
        "g1", "g2",
    },

    "I": {
        "a1", "a2",
        "d1", "d2",
        "l", "m",
    },

    "J": {
        "b",
        "c",
        "d1",
        "d2",
        "e",
    },

    "K": {
        "e", "f",
        "g1",
        "i", "k",
    },

    "L": {
        "d1", "d2",
        "e", "f",
    },

    "N": {
        "e", "f",
        "b", "c",
        "h", "k",
    },

    "O": {
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "e", "f",
    },

    "P": {
        "a1",
        "a2",
        "b",
        "e",
        "f",
        "g1",
        "g2",
    },

    "R": {
        "a1", "a2",
        "b",
        "e", "f",
        "g1", "g2",
        "k",
    },

    "S": {
        "a1", "a2",
        "c",
        "d1", "d2",
        "g2",
        "h",
    },

    "T": {
        "a1", "a2",
        "l", "m",
    },

    "U": {
        "b", "c",
        "d1", "d2",
        "e", "f",
    },

    "W": {
        "b",
        "c",
        "e",
        "f",
        "j",
        "k",
    },
}


# ------------------------------------------------------------
# DATA STRUCTURE
# ------------------------------------------------------------

@dataclass
class CharacterResult:
    """
    Result for one character cell.
    """

    character: str
    pattern: tuple[int, ...]
    confidence: float


# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

def create_display_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert the display image into a binary LED mask.

    The display LEDs are bright and saturated blue/cyan,
    while the background is dark.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    _, saturation, value = cv2.split(hsv)

    mask = (
        (saturation > 80)
        & (value > 80)
    ).astype(
        np.uint8
    ) * 255

    return mask


# ------------------------------------------------------------
# SEGMENT SAMPLING
# ------------------------------------------------------------

def sample_segment(
    mask: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    cell_x: int,
    y_offset: int = 5,
    radius: int = 2,
) -> float:
    """
    Calculate the percentage of an individual segment that
    contains illuminated pixels.

    Returns:
        value between 0 and 1.
    """

    x1 = p1[0] + cell_x
    y1 = p1[1] + y_offset

    x2 = p2[0] + cell_x
    y2 = p2[1] + y_offset

    length = int(
        math.hypot(
            x2 - x1,
            y2 - y1,
        )
    ) + 1

    xs = np.linspace(
        x1,
        x2,
        length,
    )

    ys = np.linspace(
        y1,
        y2,
        length,
    )

    samples = []

    for x, y in zip(xs, ys):

        xi = int(round(x))
        yi = int(round(y))

        x_min = max(
            0,
            xi - radius,
        )

        x_max = min(
            mask.shape[1],
            xi + radius + 1,
        )

        y_min = max(
            0,
            yi - radius,
        )

        y_max = min(
            mask.shape[0],
            yi + radius + 1,
        )

        region = mask[
            y_min:y_max,
            x_min:x_max,
        ]

        if region.size:
            samples.append(
                np.mean(region > 0)
            )

    if not samples:
        return 0.0

    return float(
        np.mean(samples)
    )


def get_segment_scores(
    mask: np.ndarray,
    cell_x: int,
) -> list[float]:
    """
    Measure all 16 segments in one character cell.
    """

    scores = []

    for p1, p2 in SEGMENTS:

        score = sample_segment(
            mask,
            p1,
            p2,
            cell_x,
        )

        scores.append(
            score
        )

    return scores


# ------------------------------------------------------------
# PATTERN CREATION
# ------------------------------------------------------------

def scores_to_pattern(
    scores: list[float],
    threshold: float = 0.55,
) -> tuple[int, ...]:
    """
    Convert segment brightness scores into an ON/OFF pattern.
    """

    return tuple(
        1 if score >= threshold else 0
        for score in scores
    )


def character_to_pattern(
    character: str,
) -> tuple[int, ...]:
    """
    Convert a character into its expected 16-segment pattern.
    """

    character = character.upper()

    active_segments = CHARACTER_SEGMENTS.get(
        character,
        set(),
    )

    return tuple(
        1 if name in active_segments else 0
        for name in SEGMENT_NAMES
    )


# ------------------------------------------------------------
# CHARACTER MATCHING
# ------------------------------------------------------------

def pattern_distance(
    observed: tuple[int, ...],
    expected: tuple[int, ...],
) -> float:
    """
    Calculate normalized Hamming distance between two patterns.
    """

    if len(observed) != len(expected):
        raise ValueError(
            "Patterns must have the same length."
        )

    differences = sum(
        a != b
        for a, b in zip(
            observed,
            expected,
        )
    )

    return differences / len(
        observed
    )


def decode_character(
    pattern: tuple[int, ...],
) -> tuple[str, float]:
    """
    Find the character whose segment pattern is closest
    to the observed pattern.

    Returns:
        character
        confidence
    """

    best_character = "?"
    best_distance = float("inf")

    for character in CHARACTER_SEGMENTS:

        expected = character_to_pattern(
            character
        )

        distance = pattern_distance(
            pattern,
            expected,
        )

        if distance < best_distance:

            best_distance = distance
            best_character = character

    # Confidence is the inverse of normalized error.
    confidence = 1.0 - best_distance

    return (
        best_character,
        confidence,
    )


# ------------------------------------------------------------
# COMPLETE FRAME DECODING
# ------------------------------------------------------------

def decode_frame(
    image: np.ndarray,
    threshold: float = 0.55,
) -> list[CharacterResult]:
    """
    Decode all character cells in one display frame.
    """

    mask = create_display_mask(
        image
    )

    results = []

    for cell_x in CELL_STARTS:

        scores = get_segment_scores(
            mask,
            cell_x,
        )

        pattern = scores_to_pattern(
            scores,
            threshold,
        )

        # Blank character detection.
        max_score = max(scores)

        if max_score < 0.30:

            result = CharacterResult(
                character=" ",
                pattern=pattern,
                confidence=1.0,
            )

        else:

            character, confidence = (
                decode_character(
                    pattern
                )
            )

            result = CharacterResult(
                character=character,
                pattern=pattern,
                confidence=confidence,
            )

        results.append(
            result
        )

    return results


def decode_text(
    image: np.ndarray,
    threshold: float = 0.55,
    min_confidence: float = 0.75,
) -> str:
    """
    Decode the visible text.

    Characters with low confidence are represented as '?'.
    This prevents uncertain edge characters from being silently
    treated as reliable detections.
    """

    results = decode_frame(
        image,
        threshold,
    )

    decoded = []

    for result in results:

        # Preserve blank cells.
        if result.character == " ":

            decoded.append(" ")
            continue

        # Reject uncertain classifications.
        if result.confidence < min_confidence:

            decoded.append("?")

        else:

            decoded.append(
                result.character
            )

    return "".join(
        decoded
    ).strip()
if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    text_dir = project_root / "data" / "text"

    images = sorted(text_dir.glob("*.bmp"))

    if not images:
        raise RuntimeError("No BMP images found.")

    print()
    print("=" * 70)
    print("METHOD 1 - 16-SEGMENT DISPLAY DECODER")
    print("=" * 70)
    print()
    print(f"Total images: {len(images)}")
    print()

    for index, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Frame {index:03d}: "
                "ERROR - could not read image"
            )
            continue

        text = decode_text(image)

        print(
            f"Frame {index:03d}: "
            f"{text}"
        )