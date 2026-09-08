import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "data" / "text"

CELL_STARTS = [
    18, 80, 141, 202, 264, 326, 387,
    448, 510, 572, 633, 694, 756
]

CELL_WIDTH = 55

MIN_ACTIVE_PIXELS = 80

# Characters present in the dataset
CHARACTERS = list("ABCDEF IJKLNOPRSTUW".replace(" ", ""))

TARGET_TEXTS = [
    "WATER IN FLUE",
    "AIR FILTER IS BLOCKED",
    "PLEASE CONTACT JCB DEALER"
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def create_display_mask(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    _, saturation, value = cv2.split(hsv)

    mask = (
        (saturation > 80) &
        (value > 80)
    ).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# ============================================================
# CELL EXTRACTION
# ============================================================

def extract_cells(mask):

    cells = []

    for start_x in CELL_STARTS:

        end_x = start_x + CELL_WIDTH

        cell = mask[:, start_x:end_x]

        cells.append(cell)

    return cells


# ============================================================
# GEOMETRIC DESCRIPTOR
# ============================================================

def calculate_features(cell):

    active_pixels = int(np.count_nonzero(cell))

    if active_pixels < MIN_ACTIVE_PIXELS:
        return None

    contours, _ = cv2.findContours(
        cell,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) >= 2
    ]

    if not contours:
        return None

    total_area = sum(
        cv2.contourArea(c)
        for c in contours
    )

    largest = max(
        contours,
        key=cv2.contourArea
    )

    points = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(points)

    aspect_ratio = w / max(h, 1)

    fill_ratio = (
        active_pixels /
        max(w * h, 1)
    )

    contour_count = len(contours)

    largest_area_ratio = (
        cv2.contourArea(largest) /
        max(total_area, 1)
    )

    moments = cv2.moments(
        cell,
        binaryImage=True
    )

    if moments["m00"] != 0:

        center_x = (
            moments["m10"] /
            moments["m00"]
        )

        center_y = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        center_x = 0
        center_y = 0

    normalized_center_x = (
        center_x / cell.shape[1]
    )

    normalized_center_y = (
        center_y / cell.shape[0]
    )

    return np.array([
        aspect_ratio,
        fill_ratio,
        contour_count,
        largest_area_ratio,
        normalized_center_x,
        normalized_center_y,
        active_pixels /
        (cell.shape[0] * cell.shape[1])
    ], dtype=np.float32)


# ============================================================
# SHAPE DESCRIPTOR
# ============================================================

def create_shape_descriptor(cell):

    h, w = cell.shape

    descriptor = []

    # Divide character into a grid.
    # This captures where illuminated pixels occur.
    rows = 4
    cols = 3

    for r in range(rows):

        y1 = int(r * h / rows)
        y2 = int((r + 1) * h / rows)

        for c in range(cols):

            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)

            region = cell[y1:y2, x1:x2]

            ratio = (
                np.count_nonzero(region) /
                max(region.size, 1)
            )

            descriptor.append(ratio)

    features = calculate_features(cell)

    if features is None:
        return None

    descriptor.extend(
        features.tolist()
    )

    return np.array(
        descriptor,
        dtype=np.float32
    )


# ============================================================
# FRAME LOADING
# ============================================================

def get_files():

    return sorted(
        TEXT_DIR.glob("*.bmp")
    )


def load_frame(frame_number):

    files = get_files()

    if (
        frame_number < 1 or
        frame_number > len(files)
    ):
        return None

    return cv2.imread(
        str(files[frame_number - 1])
    )


# ============================================================
# CHARACTER REFERENCE LOCATIONS
# ============================================================

# These are reliable examples already obtained from
# the supplied dataset during Method 2 template creation.

REFERENCE_LOCATIONS = {

    "A": [(21, 10), (22, 10), (28, 10)],

    "B": [(70, 10), (69, 10)],

    "C": [(60, 10), (61, 10)],

    "D": [(73, 9), (72, 9)],

    "E": [(7, 11), (6, 11)],

    "F": [(103, 11), (102, 11)],

    "I": [(10, 13), (9, 13)],

    "J": [(68, 10), (67, 10)],

    "K": [(39, 11), (38, 11)],

    "L": [(37, 10), (36, 10)],

    "N": [(67, 5), (66, 5)],

    "O": [(61, 10), (62, 10)],

    "P": [(58, 5), (57, 5)],

    "R": [(29, 5), (30, 5)],

    "S": [(64, 3), (63, 3)],

    "T": [(3, 13), (4, 13)],

    "U": [(102, 13), (101, 13)],

    "W": [(89, 10), (90, 10)],
}


# ============================================================
# BUILD MULTIPLE PROTOTYPES
# ============================================================

def build_prototypes():

    prototypes = {
        char: []
        for char in CHARACTERS
    }

    print(
        "Building multiple contour prototypes..."
    )

    for character, locations in REFERENCE_LOCATIONS.items():

        for frame_number, cell_number in locations:

            image = load_frame(
                frame_number
            )

            if image is None:
                continue

            mask = create_display_mask(
                image
            )

            cells = extract_cells(mask)

            index = cell_number - 1

            if (
                index < 0 or
                index >= len(cells)
            ):
                continue

            cell = cells[index]

            descriptor = create_shape_descriptor(
                cell
            )

            if descriptor is None:
                continue

            prototypes[character].append(
                descriptor
            )

    total = sum(
        len(v)
        for v in prototypes.values()
    )

    for character in CHARACTERS:

        print(
            f"  {character}: "
            f"{len(prototypes[character])} examples"
        )

    print(
        f"\nCreated {total} prototype examples "
        f"for {len(prototypes)} characters."
    )

    return prototypes


# ============================================================
# DISTANCE
# ============================================================

def descriptor_distance(a, b):

    denominator = np.maximum(
        np.abs(a) + np.abs(b),
        1e-6
    )

    distance = np.mean(
        np.abs(a - b) /
        denominator
    )

    return float(distance)


# ============================================================
# CHARACTER CLASSIFICATION
# ============================================================

def classify_character(
    cell,
    prototypes
):

    descriptor = create_shape_descriptor(
        cell
    )

    if descriptor is None:
        return "", 0.0

    character_distances = {}

    for character, examples in prototypes.items():

        if not examples:
            continue

        distances = [
            descriptor_distance(
                descriptor,
                example
            )
            for example in examples
        ]

        # Use the closest example.
        best_distance = min(
            distances
        )

        character_distances[
            character
        ] = best_distance

    if not character_distances:
        return "", 0.0

    ranked = sorted(
        character_distances.items(),
        key=lambda x: x[1]
    )

    best_character, best_distance = ranked[0]

    confidence = max(
        0.0,
        1.0 - best_distance
    )

    if confidence < 0.55:
        return "", confidence

    return (
        best_character,
        confidence
    )


# ============================================================
# FRAME DECODING
# ============================================================

def decode_frame(
    image,
    prototypes
):

    mask = create_display_mask(
        image
    )

    cells = extract_cells(
        mask
    )

    decoded = []

    for cell in cells:

        active_pixels = np.count_nonzero(
            cell
        )

        if active_pixels < MIN_ACTIVE_PIXELS:

            decoded.append("")
            continue

        character, confidence = (
            classify_character(
                cell,
                prototypes
            )
        )

        decoded.append(
            character
        )

    return "".join(decoded)


# ============================================================
# SCROLLING TEXT CLEANUP
# ============================================================

def clean_text(text):

    text = text.strip()

    while "  " in text:
        text = text.replace(
            "  ",
            " "
        )

    return text


# ============================================================
# TARGET DETECTION
# ============================================================

def normalize_text(text):

    return (
        text.upper()
        .replace(" ", "")
        .strip()
    )


def is_target_fragment(text):

    normalized = normalize_text(text)

    if not normalized:
        return False

    target = normalize_text(
        "AIR FILTER IS BLOCKED"
    )

    return normalized in target


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "METHOD 3 - MULTI-PROTOTYPE "
        "CONTOUR RECOGNITION"
    )

    print("=" * 60)

    print(
        f"\nDataset: {TEXT_DIR}"
    )

    if not TEXT_DIR.exists():

        print(
            "\nERROR: Dataset folder not found."
        )

        return

    prototypes = build_prototypes()

    valid_characters = sum(
        bool(v)
        for v in prototypes.values()
    )

    if valid_characters < 10:

        print(
            "\nERROR: Too few character prototypes."
        )

        return

    files = get_files()

    print(
        f"\nFound {len(files)} BMP frames."
    )

    print(
        "\nDecoding frames...\n"
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
            continue

        text = decode_frame(
            image,
            prototypes
        )

        text = clean_text(
            text
        )

        results.append(
            (
                frame_number,
                path.name,
                text
            )
        )

        print(
            f"Frame {frame_number:03d}: "
            f"{text}"
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

    # Target frames based on the Method 3 result.
    target_frames = [
        frame
        for frame, filename, text
        in results
        if is_target_fragment(text)
    ]

    if target_frames:

        print(
            "\nTarget message evidence:"
        )

        print(
            f"First matching frame: "
            f"{min(target_frames)}"
        )

        print(
            f"Last matching frame: "
            f"{max(target_frames)}"
        )

    return results


if __name__ == "__main__":
    main()