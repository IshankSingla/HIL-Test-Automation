"""
Evaluate Method 1 on the supplied BMP dataset.

The dataset contains scrolling messages such as:

    WATER IN FLUE
    AIR FILTER IS BLOCKED
    PLEASE CONTACT JCB DEALER

This script:
    1. Reads all BMP images
    2. Extracts timestamps from filenames
    3. Decodes each frame using the 16-segment decoder
    4. Finds the AIR FILTER IS BLOCKED message interval
    5. Calculates its display duration
"""

from pathlib import Path
from datetime import datetime

import cv2

from .segment_decoder import decode_text


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "text"
)


# ============================================================
# TARGET MESSAGE
# ============================================================

TARGET_TEXT = "AIR FILTER IS BLOCKED"


# ============================================================
# TIMESTAMP
# ============================================================

def extract_timestamp(filename: str) -> datetime:
    """
    Extract timestamp from filenames such as:

        20250620_095923806_.bmp

    Format:

        YYYYMMDD_HHMMSSmmm_.bmp
    """

    stem = Path(filename).stem

    # Remove trailing underscore
    stem = stem.rstrip("_")

    parts = stem.split("_")

    date_part = parts[0]
    time_part = parts[1]

    year = int(date_part[0:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])

    hour = int(time_part[0:2])
    minute = int(time_part[2:4])
    second = int(time_part[4:6])
    millisecond = int(time_part[6:9])

    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        millisecond * 1000,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Find BMP files
    # --------------------------------------------------------

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            f"No BMP images found in: {TEXT_DIR}"
        )

    print()
    print("=" * 80)
    print("TEXT DETECTION EVALUATION")
    print("=" * 80)

    print()
    print(
        f"Total images: {len(images)}"
    )

    # --------------------------------------------------------
    # Decode every frame
    #
    # IMPORTANT:
    # `decoded` is created here inside main() and remains
    # available to all analysis below.
    # --------------------------------------------------------

    decoded = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Frame {index:3d} | "
                f"ERROR reading image"
            )

            continue

        # Decode using Method 1
        text = decode_text(
            image
        )

        # Extract timestamp
        timestamp = extract_timestamp(
            image_path.name
        )

        # Store complete result
        decoded.append(
            {
                "frame": index,
                "filename": image_path.name,
                "timestamp": timestamp,
                "text": text,
            }
        )

    # --------------------------------------------------------
    # Print all decoded frames
    # --------------------------------------------------------

    print()

    for item in decoded:

        timestamp_string = (
            item["timestamp"]
            .strftime(
                "%H:%M:%S.%f"
            )[:-3]
        )

        print(
            f"Frame {item['frame']:3d} | "
            f"{timestamp_string} | "
            f"{item['text']!r}"
        )

    # ========================================================
    # TARGET MESSAGE ANALYSIS
    # ========================================================

    print()
    print("=" * 80)
    print("TARGET MESSAGE DISPLAY TIME")
    print("=" * 80)

    # --------------------------------------------------------
    # Find first frame containing "AIR"
    # --------------------------------------------------------

    air_frame_index = None

    for i, item in enumerate(decoded):

        text = (
            item["text"]
            .replace("?", "")
            .strip()
        )

        if "AIR" in text:

            air_frame_index = i

            break

    # --------------------------------------------------------
    # If target was not found
    # --------------------------------------------------------

    if air_frame_index is None:

        print()
        print(
            "Target message was not detected."
        )

        return

    # --------------------------------------------------------
    # Find beginning of message
    #
    # Move backwards until the previous frame is blank.
    # --------------------------------------------------------

    start_index = air_frame_index

    while start_index > 0:

        previous = decoded[
            start_index - 1
        ]

        previous_text = (
            previous["text"]
            .replace("?", "")
            .strip()
        )

        if previous_text == "":

            break

        start_index -= 1

    # --------------------------------------------------------
    # Find end of message
    #
    # Continue until the first blank frame.
    # --------------------------------------------------------

    end_index = air_frame_index

    while (
        end_index + 1
        < len(decoded)
    ):

        next_item = decoded[
            end_index + 1
        ]

        next_text = (
            next_item["text"]
            .replace("?", "")
            .strip()
        )

        if next_text == "":

            break

        end_index += 1

    # --------------------------------------------------------
    # Get start and last visible frame
    # --------------------------------------------------------

    start_frame = decoded[
        start_index
    ]

    last_visible_frame = decoded[
        end_index
    ]

    # --------------------------------------------------------
    # First blank frame after target
    # --------------------------------------------------------

    blank_after_index = (
        end_index + 1
    )

    if (
        blank_after_index
        < len(decoded)
    ):

        end_frame = decoded[
            blank_after_index
        ]

    else:

        end_frame = last_visible_frame

    # --------------------------------------------------------
    # Calculate duration
    # --------------------------------------------------------

    duration = (
        end_frame["timestamp"]
        - start_frame["timestamp"]
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()

    print(
        f"Message                  : "
        f"'{TARGET_TEXT}.'"
    )

    print(
        f"First visible frame      : "
        f"{start_frame['frame']}"
    )

    print(
        f"First visible timestamp  : "
        f"{start_frame['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
    )

    print(
        f"Last visible frame       : "
        f"{last_visible_frame['frame']}"
    )

    print(
        f"Last visible timestamp   : "
        f"{last_visible_frame['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
    )

    print(
        f"First blank after message: "
        f"{end_frame['frame']}"
    )

    print(
        f"End timestamp            : "
        f"{end_frame['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
    )

    print()

    print(
        f"Display duration         : "
        f"{duration.total_seconds():.3f} seconds"
    )

    # ========================================================
    # BLANK FRAME SUMMARY
    # ========================================================

    blank_frames = [
        item["frame"]
        for item in decoded
        if not item["text"].strip()
    ]

    print()
    print("=" * 80)
    print("BLANK FRAME SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Blank frames: {blank_frames}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()