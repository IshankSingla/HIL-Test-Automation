import cv2
import easyocr
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "data" / "text"

TARGET_TEXT = "AIR FILTER IS BLOCKED"


# ============================================================
# OCR READER
# ============================================================

def create_reader():
    """
    Create EasyOCR reader.

    English is sufficient for this dataset.
    GPU is disabled so this works on CPU.
    """

    print("Initializing EasyOCR...")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("EasyOCR initialized.")

    return reader


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepare the 16-segment display image for OCR.

    Several versions are generated because OCR can behave
    differently depending on thresholding.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Otsu threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # HSV-based display mask
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(hsv)

    display_mask = (
        (saturation > 80) &
        (value > 80)
    ).astype(np.uint8) * 255

    # Enlarge image for OCR
    binary_big = cv2.resize(
        binary,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    display_big = cv2.resize(
        display_mask,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST
    )

    return [
        image,
        binary_big,
        display_big
    ]


# ============================================================
# OCR CLEANUP
# ============================================================

def clean_ocr_text(text):

    text = text.upper()

    # Common OCR mistakes for this display
    replacements = {
        "|": "I",
        "1": "I",
        "0": "O",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Keep useful characters
    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    )

    text = "".join(
        ch if ch in allowed else " "
        for ch in text
    )

    # Remove repeated spaces
    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# OCR SINGLE IMAGE
# ============================================================

def run_ocr(reader, image):

    candidates = []

    processed_images = preprocess_image(
        image
    )

    for processed in processed_images:

        try:

            results = reader.readtext(
                processed,
                detail=1,
                paragraph=False
            )

        except Exception as error:

            print(
                f"OCR error: {error}"
            )

            continue

        for result in results:

            if len(result) < 3:
                continue

            text = result[1]
            confidence = float(result[2])

            text = clean_ocr_text(
                text
            )

            if not text:
                continue

            candidates.append(
                (
                    text,
                    confidence
                )
            )

    if not candidates:
        return "", 0.0

    # Prefer the longest reasonable OCR result,
    # then use confidence as a tie-breaker.
    candidates.sort(
        key=lambda item: (
            len(item[0]),
            item[1]
        ),
        reverse=True
    )

    best_text, best_confidence = candidates[0]

    return (
        best_text,
        best_confidence
    )


# ============================================================
# TARGET FRAGMENT CHECK
# ============================================================

def normalize(text):

    return (
        text.upper()
        .replace(" ", "")
        .strip()
    )


def target_fragment(text):

    text_normalized = normalize(text)

    target_normalized = normalize(
        TARGET_TEXT
    )

    if not text_normalized:
        return False

    return (
        text_normalized in
        target_normalized
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("METHOD 3 - EASYOCR")
    print("=" * 60)

    print(
        f"\nDataset: {TEXT_DIR}"
    )

    if not TEXT_DIR.exists():

        print(
            "ERROR: Dataset folder not found."
        )

        return

    files = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    print(
        f"Found {len(files)} BMP frames."
    )

    if not files:

        print(
            "ERROR: No BMP files found."
        )

        return

    reader = create_reader()

    print(
        "\nRunning OCR on frames...\n"
    )

    results = []

    for frame_number, path in enumerate(
        files,
        start=1
    ):

        image = cv2.imread(
            str(path)
        )

        if image is None:

            print(
                f"Frame {frame_number:03d}: "
                "IMAGE ERROR"
            )

            continue

        text, confidence = run_ocr(
            reader,
            image
        )

        results.append(
            (
                frame_number,
                path.name,
                text,
                confidence
            )
        )

        if text:

            print(
                f"Frame {frame_number:03d}: "
                f"{text} "
                f"(confidence={confidence:.2f})"
            )

        else:

            print(
                f"Frame {frame_number:03d}: "
                "[blank]"
            )

    # ========================================================
    # TARGET MESSAGE ANALYSIS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TARGET MESSAGE ANALYSIS"
    )

    print(
        "=" * 60
    )

    target_frames = []

    for frame, filename, text, confidence in results:

        if target_fragment(text):

            target_frames.append(
                frame
            )

    if target_frames:

        print(
            f"Target fragment detected in frames: "
            f"{min(target_frames)} - "
            f"{max(target_frames)}"
        )

    else:

        print(
            "EasyOCR did not directly detect "
            "the target text."
        )

        print(
            "This is possible because the supplied "
            "images contain a scrolling 16-segment display."
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "METHOD 3 COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()