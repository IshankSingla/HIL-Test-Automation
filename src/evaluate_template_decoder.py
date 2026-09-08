"""
Method 2: Template Matching Evaluation

This script evaluates the template-matching decoder across
all supplied frames.

It identifies the target message by checking whether the
decoded text is a contiguous visible portion of:

    AIR FILTER IS BLOCKED

This avoids confusing unrelated messages such as:

    WATER IN FLUE
    PLEASE CONTACT JCB DEALER
"""

from pathlib import Path
from datetime import datetime

import cv2

from template_decoder import (
    load_templates,
    decode_frame,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = PROJECT_ROOT / "data" / "text"


# ============================================================
# TARGET MESSAGE
# ============================================================

TARGET = "AIR FILTER IS BLOCKED"


# ============================================================
# TIMESTAMP
# ============================================================

def timestamp_from_filename(filename):
    """
    Convert a filename such as:

        20250620_095923806_.bmp

    into a datetime object.
    """

    name = filename.stem

    timestamp = name.strip("_")

    return datetime.strptime(
        timestamp,
        "%Y%m%d_%H%M%S%f",
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize spaces and capitalization.
    """

    return " ".join(
        text.upper().split()
    )


# ============================================================
# TARGET MATCH
# ============================================================

def is_target_fragment(text):
    """
    Determine whether the decoded text is a visible fragment
    of the target message.

    Examples that should match:

        A
        AI
        AIR
        AIR FILTER
        AIR FILTER IS
        R FILTER IS
        FILTER IS BL
        TER IS BLOCKE
        ER IS BLOCKED

    Examples that should NOT match:

        WATER
        WATER IN
        PLEASE
        CONTACT JCB
    """

    text = normalize_text(
        text
    )

    if not text:
        return False

    # --------------------------------------------------------
    # A decoded frame must be an exact contiguous substring
    # of the target message.
    # --------------------------------------------------------

    return text in TARGET


# ============================================================
# FIND TARGET RUN
# ============================================================

def find_target_run(decoded_frames):
    """
    Find the continuous sequence of frames belonging to the
    target message.

    The display contains blank intervals separating messages.

    We therefore:
        1. identify non-blank runs;
        2. count target fragments in each run;
        3. choose the run with the strongest target evidence.
    """

    runs = []

    current_run = []

    for item in decoded_frames:

        text = normalize_text(
            item["text"]
        )

        if text:

            current_run.append(
                item
            )

        else:

            if current_run:

                runs.append(
                    current_run
                )

                current_run = []

    # Handle a run that reaches the end.
    if current_run:

        runs.append(
            current_run
        )

    # --------------------------------------------------------
    # Evaluate each run.
    # --------------------------------------------------------

    best_run = None
    best_score = -1

    for run in runs:

        target_frames = [
            item
            for item in run
            if is_target_fragment(
                item["text"]
            )
        ]

        if not target_frames:
            continue

        # ----------------------------------------------------
        # Score based on:
        #
        # 1. number of target fragments
        # 2. total length of matching text
        # ----------------------------------------------------

        total_length = sum(
            len(
                normalize_text(
                    item["text"]
                )
            )
            for item in target_frames
        )

        score = (
            len(target_frames) * 1000
            + total_length
        )

        if score > best_score:

            best_score = score
            best_run = (
                run,
                target_frames,
            )

    return best_run


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("METHOD 2 — TEMPLATE MATCHING EVALUATION")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Load templates.
    # --------------------------------------------------------

    templates = load_templates()

    print(
        f"Loaded {len(templates)} templates."
    )

    # --------------------------------------------------------
    # Load frames.
    # --------------------------------------------------------

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            f"No BMP images found in {TEXT_DIR}"
        )

    print(
        f"Found {len(images)} frames."
    )

    print()

    # ========================================================
    # DECODE ALL FRAMES
    # ========================================================

    decoded_frames = []

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

        text, confidences = decode_frame(
            image,
            templates,
        )

        decoded_frames.append(
            {
                "frame": frame_number,
                "filename": image_path.name,
                "text": text,
                "timestamp": timestamp_from_filename(
                    image_path
                ),
                "confidences": confidences,
            }
        )

    # ========================================================
    # PRINT FRAME DECODING
    # ========================================================

    print("=" * 80)
    print("FRAME DECODING")
    print("=" * 80)
    print()

    for item in decoded_frames:

        print(
            f"Frame {item['frame']:03d}: "
            f"{item['text']}"
        )

    # ========================================================
    # FIND TARGET
    # ========================================================

    result = find_target_run(
        decoded_frames
    )

    if result is None:

        print()
        print(
            "ERROR: Target message was not detected."
        )

        return

    target_run, target_frames = result

    # --------------------------------------------------------
    # First and last frame of the actual target run.
    # --------------------------------------------------------

    first_target = target_run[0]
    last_target = target_run[-1]

    # --------------------------------------------------------
    # Find first blank after the target run.
    # --------------------------------------------------------

    end_item = None

    for item in decoded_frames:

        if item["frame"] <= last_target["frame"]:
            continue

        if not normalize_text(
            item["text"]
        ):

            end_item = item
            break

    if end_item is None:

        end_item = last_target

    # ========================================================
    # DISPLAY DURATION
    # ========================================================

    start_time = first_target[
        "timestamp"
    ]

    end_time = end_item[
        "timestamp"
    ]

    duration = (
        end_time - start_time
    ).total_seconds()

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("TARGET MESSAGE DISPLAY TIME")
    print("=" * 80)
    print()

    print(
        f"Message                  : "
        f"'{TARGET}.'"
    )

    print(
        f"First visible frame      : "
        f"{first_target['frame']}"
    )

    print(
        f"First visible timestamp  : "
        f"{first_target['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
    )

    print(
        f"Last visible frame       : "
        f"{last_target['frame']}"
    )

    print(
        f"Last visible timestamp   : "
        f"{last_target['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
    )

    print(
        f"First blank after message: "
        f"{end_item['frame']}"
    )

    print(
        f"End timestamp            : "
        f"{end_item['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"
    )

    print(
        f"Display duration         : "
        f"{duration:.3f} seconds"
    )

    print()
    print("=" * 80)
    print("METHOD 2 EVALUATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()