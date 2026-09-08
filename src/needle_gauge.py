"""
Needle Gauge Detection

Detects the colored needle in a gauge image, calculates its angle,
and converts that angle into a gauge value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


# Approximate center/pivot of the gauge in the supplied images.
# Image size: 949 x 665.
DEFAULT_CENTER = (505.0, 360.0)


@dataclass
class NeedleResult:
    """Result returned by the needle detector."""

    angle_deg: float
    line: tuple[int, int, int, int]
    value: float | None = None


def create_needle_mask(image: np.ndarray) -> np.ndarray:
    """
    Isolate the yellow/orange/red needle.

    The needle has high saturation and high brightness compared
    with the mostly black/white gauge background.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue, saturation, value = cv2.split(hsv)

    # Yellow/orange/red region.
    mask = (
        (saturation > 150)
        & (value > 150)
        & (hue < 45)
    ).astype(np.uint8) * 255

    # Remove irrelevant image borders.
    mask[:80, :] = 0
    mask[610:, :] = 0
    mask[:, :180] = 0
    mask[:, 900:] = 0

    # Remove small noise.
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


def detect_needle_angle(
    image: np.ndarray,
    center: tuple[float, float] = DEFAULT_CENTER,
) -> tuple[float, tuple[int, int, int, int]]:
    """
    Detect the gauge needle and return:

        angle in degrees
        detected line coordinates

    The implementation is deliberately robust to the different
    array shapes returned by different OpenCV versions.
    """

    mask = create_needle_mask(image)

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,
        threshold=80,
        minLineLength=100,
        maxLineGap=40,
    )

    if lines is None:
        raise RuntimeError(
            "Could not detect the gauge needle."
        )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # OpenCV normally returns:
    #
    #     [[[x1, y1, x2, y2]],
    #      [[x1, y1, x2, y2]],
    #      ...]
    #
    # But depending on the OpenCV build, the returned shape
    # can vary.
    #
    # Flatten everything into individual groups of 4 values.
    # ---------------------------------------------------------

    lines = np.asarray(lines)

    if lines.size < 4:
        raise RuntimeError(
            "Hough transform returned insufficient line data."
        )

    lines = lines.reshape(-1, 4)

    cx, cy = center

    candidates = []

    for line in lines:

        # Convert the four coordinates individually.
        x1 = int(line[0])
        y1 = int(line[1])
        x2 = int(line[2])
        y2 = int(line[3])

        line_length = math.hypot(
            x2 - x1,
            y2 - y1
        )

        # Ignore short lines.
        if line_length < 100:
            continue

        vx = x2 - x1
        vy = y2 - y1

        # -----------------------------------------------------
        # Calculate distance between the gauge center and the
        # candidate line.
        # -----------------------------------------------------

        distance_from_center = abs(
            vx * (cy - y1)
            - (cx - x1) * vy
        ) / max(
            line_length,
            1e-9
        )

        # The needle should originate close to the pivot.
        if distance_from_center > 80:
            continue

        # -----------------------------------------------------
        # Determine which endpoint is farther from the pivot.
        # That endpoint represents the needle direction.
        # -----------------------------------------------------

        distance_1 = math.hypot(
            x1 - cx,
            y1 - cy
        )

        distance_2 = math.hypot(
            x2 - cx,
            y2 - cy
        )

        if distance_1 > distance_2:

            tip_x = x1
            tip_y = y1

        else:

            tip_x = x2
            tip_y = y2

        # -----------------------------------------------------
        # Calculate clockwise angle.
        #
        #             0°
        #              |
        #              |
        #       270° --+-- 90°
        #              |
        #              |
        #            180°
        # -----------------------------------------------------

        angle = (
            math.degrees(
                math.atan2(
                    tip_x - cx,
                    cy - tip_y
                )
            )
            + 360
        ) % 360

        # -----------------------------------------------------
        # Candidate score.
        #
        # Longer line = better.
        # Closer to pivot = better.
        # -----------------------------------------------------

        score = (
            line_length
            - 2.0 * distance_from_center
        )

        candidates.append(
            (
                score,
                angle,
                (x1, y1, x2, y2)
            )
        )

    if not candidates:

        raise RuntimeError(
            "No valid needle line was found. "
            "Try adjusting the color mask or Hough parameters."
        )

    # Select the strongest candidate.
    best_candidate = max(
        candidates,
        key=lambda item: item[0]
    )

    _, angle, line = best_candidate

    return angle, line


def unwrap_angles(
    angles: list[float]
) -> np.ndarray:
    """
    Convert circular angles into a continuous sequence.

    Example:

        233°
        305°
        2°

    becomes:

        233°
        305°
        362°
    """

    result = np.asarray(
        angles,
        dtype=float
    ).copy()

    for i in range(1, len(result)):

        while result[i] < result[i - 1]:
            result[i] += 360

    return result


def build_calibration(
    known_values: list[float],
    known_angles: list[float],
):
    """
    Build a piecewise-linear calibration function.

    The supplied reference images give us known gauge values:

        0
        900
        1500
        1900
        3000
    """

    if len(known_values) != len(known_angles):
        raise ValueError(
            "Number of values and angles must match."
        )

    if len(known_values) < 2:
        raise ValueError(
            "At least two calibration points are required."
        )

    angles = unwrap_angles(
        known_angles
    )

    values = np.asarray(
        known_values,
        dtype=float
    )

    # Sort according to angle.
    order = np.argsort(angles)

    angles = angles[order]
    values = values[order]

    def mapper(angle: float) -> float:
        """
        Convert a detected angle into a gauge value.
        """

        normalized_angle = angle % 360

        # Move the angle onto the same continuous
        # branch as the calibration angles.
        while normalized_angle < angles[0]:
            normalized_angle += 360

        # Linear interpolation.
        value = np.interp(
            normalized_angle,
            angles,
            values
        )

        return float(value)

    return mapper


def detect_value(
    image: np.ndarray,
    mapper,
    center: tuple[float, float] = DEFAULT_CENTER,
) -> NeedleResult:
    """
    Detect needle and convert its angle to gauge value.
    """

    angle, line = detect_needle_angle(
        image,
        center
    )

    value = mapper(angle)

    return NeedleResult(
        angle_deg=angle,
        line=line,
        value=value
    )


def draw_detection(
    image: np.ndarray,
    result: NeedleResult,
    center: tuple[float, float] = DEFAULT_CENTER,
) -> np.ndarray:
    """
    Draw the detected needle and result on the image.

    Used for visual verification.
    """

    output = image.copy()

    x1, y1, x2, y2 = result.line

    cx, cy = map(
        int,
        center
    )

    # Detected needle.
    cv2.line(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 255),
        4
    )

    # Gauge pivot.
    cv2.circle(
        output,
        (cx, cy),
        7,
        (255, 0, 255),
        -1
    )

    # Result text.
    text = (
        f"Value: {result.value:.0f}    "
        f"Angle: {result.angle_deg:.2f} deg"
    )

    cv2.putText(
        output,
        text,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 255),
        2,
        cv2.LINE_AA
    )

    return output