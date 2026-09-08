# HIL Test Automation – Needle Gauge & 16-Segment Display

## Overview

This project implements computer-vision-based solutions for two HIL (Hardware-in-the-Loop) test automation tasks:

1. **Needle Gauge Detection**
2. **16-Segment Display Text Detection**

The solutions are implemented primarily using **Python**, **OpenCV**, and **NumPy**.

The project evaluates multiple computer vision approaches for recognizing text from a scrolling 16-segment display and compares their practical performance on the supplied image dataset.

---

## Task 1 – Needle Gauge Detection

### Objective

The objective is to accurately detect the position of a gauge needle from five supplied images.

The reference values provided in the dataset are:

| Image | Expected Value |
|---|---:|
| `1.59592113294115.jpg` | 0 |
| `895.590073713866.jpg` | 900 |
| `1463.jpg` | 1500 |
| `1880.jpg` | 1900 |
| `2984.jpg` | 3000 |

### Approach

The needle is detected using image processing techniques in OpenCV.

**Processing steps:**

1. Load the gauge image.
2. Convert the image from BGR to HSV color space.
3. Create a mask for the colored needle.
4. Apply morphological processing to reduce noise.
5. Detect line candidates using the Probabilistic Hough Line Transform.
6. Identify the line corresponding to the needle using its relationship to the gauge pivot.
7. Calculate the needle angle relative to the vertical axis.
8. Convert the detected angle into the gauge value using piecewise-linear calibration.

The gauge pivot is approximately located at:

```
(505, 360)
```

The angle convention used by the implementation is:

- `0°` = 12 o'clock
- Angle increases clockwise

### Results

The implementation correctly detects all five reference values:

| Expected | Detected |
|---:|---:|
| 0 | 0 |
| 900 | 900 |
| 1500 | 1500 |
| 1900 | 1900 |
| 3000 | 3000 |

The detected results are also saved as annotated images in:

```
outputs/needle_gauge/
```

---

## Task 2 – 16-Segment Display Text Detection

### Objective

The second task contains 103 BMP images showing a scrolling 16-segment display.

The objective is to:

1. Identify three possible methods for detecting the displayed text.
2. Implement at least two methods using custom code.
3. Determine when the message **"AIR FILTER IS BLOCKED"** is displayed.

### Dataset

The supplied dataset contains **103 BMP images**.

Each image has a scrolling 16-segment display. The sequence contains three messages:

- `WATER IN FLUE`
- `AIR FILTER IS BLOCKED`
- `PLEASE CONTACT JCB DEALER`

...followed by another occurrence of:

- `WATER IN FLUE`

---

### Method 1 – 16-Segment Pattern Recognition

**Description**

The first approach directly models the structure of the 16-segment display. Each character cell is divided according to the known 16-segment geometry. The implementation determines which segments are illuminated and maps the resulting segment pattern to a character.

The decoder contains predefined segment patterns for the characters present in the supplied dataset.

**Processing Steps**

1. Convert the image to HSV color space.
2. Detect illuminated display pixels.
3. Divide the display into character cells.
4. Sample the predefined 16 segment locations.
5. Determine which segments are active.
6. Convert the segment pattern into a character.
7. Combine recognized characters from all display cells.

**Advantages**

- Uses the known physical structure of the 16-segment display
- Does not require a machine learning model
- Does not depend on general-purpose OCR
- Provides consistent results on the supplied dataset
- Uses custom computer vision code

**Implementation:** `src/segment_decoder.py`

---

### Method 2 – Character Template Matching

**Description**

The second approach uses image templates instead of explicitly decoding the individual 16-segment states.

Character templates are generated from reliable examples in the supplied dataset. The implementation creates templates for the 18 characters used by the dataset:

`A B C D E F I J K L N O P R S T U W`

Each character is normalized and resized before comparison.

**Processing Steps**

1. Detect the illuminated display region.
2. Divide the display into character cells.
3. Extract the character image.
4. Normalize the character using its bounding box.
5. Resize the normalized character to a common size.
6. Compare it against the stored character templates.
7. Select the template with the highest similarity.
8. Combine recognized characters to reconstruct the scrolling text.

**Template Generation**

Templates are generated using:

```
src/build_character_templates.py
```

The generated templates are stored in:

```
outputs/templates/
```

All 18 required character templates were successfully generated.

**Recognition**

The template-matching decoder is implemented in:

```
src/template_decoder.py
```

**Result**

The method successfully follows the scrolling target message across frames 18–40.

Examples of the decoded sequence include:

```
Frame 018: A
Frame 019: AI
Frame 020: AIR
...
Frame 028: AIR FILTER
...
Frame 031: AIR FILTER IS
...
Frame 039: ER IS BLOCKED
Frame 040: ER IS BLOCKED
Frame 041: (blank)
```

Because the message is scrolling, the entire message does not appear in one frame. Different portions of the message enter and leave the display over time.

---

### Method 3 – Contour and Shape-Based Recognition

**Description**

The third approach uses contour-based computer vision. Instead of explicitly identifying the 16 individual segments, this method extracts geometric and spatial characteristics from each character cell.

Multiple reference prototypes are created for the characters, and new character cells are compared against these prototypes.

**Features Used**

The character descriptor includes:

- Aspect ratio
- Fill ratio
- Number of contours
- Largest contour area ratio
- Character center position
- Active pixel density
- Spatial pixel distribution using a grid

**Processing Steps**

1. Create an illuminated-pixel mask.
2. Extract individual display cells.
3. Detect contours within each cell.
4. Calculate geometric features.
5. Calculate spatial pixel-density features.
6. Compare the descriptor with reference prototypes.
7. Select the closest character prototype.

**Implementation:** `src/contour_decoder.py`

**Evaluation**

The contour-based approach successfully processed all 103 frames and detected the beginning of the target message at frame 18.

However, it produced more character substitutions than the first two approaches. Examples include:

```
Frame 028: AIRFIAETA
Frame 031: SIRFIATEAIS
Frame 039: ERISBACCKTD
```

Therefore, this approach is considered less reliable for the supplied 16-segment scrolling display than the direct segment-pattern and template-matching approaches. It is included as a third evaluated computer vision method.

---

### Optional Method – EasyOCR

EasyOCR was also evaluated as a general-purpose OCR approach.

**Implementation:** `src/easyocr_decoder.py`

EasyOCR provides a useful comparison against the custom display-specific computer vision methods. However, general-purpose OCR is not ideally suited to this particular problem because:

- The display uses 16-segment characters
- Characters are partially visible while scrolling
- Character shapes differ from conventional printed text
- Characters at the edges of the display may be incomplete

Therefore, the custom display-specific approaches are more suitable for this dataset.

> EasyOCR is treated as an optional experiment and is not required for the main implementation.

---

### Target Message Timing

The target message is:

> **AIR FILTER IS BLOCKED**

| Event | Frame | Timestamp |
|---|---|---|
| First target frame | 018 | 09:59:23.806 |
| Last target frame | 040 | 09:59:33.523 |
| First blank frame after message | 041 | 09:59:33.950 |

The duration from the first visible target frame to the first blank frame after the message is:

**10.144 seconds**

### Final Result

**"AIR FILTER IS BLOCKED"** is displayed for approximately **10.14 seconds** in the supplied image sequence.

---

## Project Structure

```
HIL-Test-Automation/
│
├── data/
│   ├── needle_gauge/
│   │   ├── 1.59592113294115.jpg
│   │   ├── 895.590073713866.jpg
│   │   ├── 1463.jpg
│   │   ├── 1880.jpg
│   │   └── 2984.jpg
│   │
│   └── text/
│       └── 103 BMP display images
│
├── outputs/
│   ├── needle_gauge/
│   │   └── detected gauge images
│   │
│   ├── templates/
│   │   └── character templates
│   │
│   └── text/
│       ├── preprocessed_first_frame.png
│       └── segment_decoder_results.csv
│
├── src/
│   ├── build_character_templates.py
│   ├── calibration.py
│   ├── contour_decoder.py
│   ├── easyocr_decoder.py
│   ├── evaluate_template_decoder.py
│   ├── main.py
│   ├── needle_gauge.py
│   ├── segment_decoder.py
│   └── template_decoder.py
│
├── archive/
│   └── development and analysis helper scripts
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd HIL-Test-Automation
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Needle Gauge Detection

From the project root:

```bash
python src\main.py
```

This performs gauge calibration, detects the needle angle, calculates the gauge value, and saves annotated output images.

### Build Character Templates

From the project root:

```bash
python src\build_character_templates.py
```

This generates the 18 character templates in:

```
outputs/templates/
```

### Template Matching

```bash
python src\template_decoder.py
```

This loads the generated templates and decodes the 103 display frames.

### Contour Recognition

```bash
python src\contour_decoder.py
```

This builds multiple character prototypes and performs contour and shape-based recognition.

### EasyOCR Experiment (Optional)

```bash
python src\easyocr_decoder.py
```

---

## Dependencies

**Main implementation:**

- Python
- OpenCV
- NumPy

**Optional:**

- EasyOCR
- PyTorch

> The custom implementations for the needle gauge and 16-segment display do not require a trained machine learning model.

---

## Key Results

### Needle Gauge

All five supplied reference images were correctly detected:

| Expected | Detected |
|---:|---:|
| 0 | 0 |
| 900 | 900 |
| 1500 | 1500 |
| 1900 | 1900 |
| 3000 | 3000 |

### Text Detection

The strongest approaches for the supplied display are:

1. 16-segment pattern recognition
2. Character template matching

Both are implemented using custom computer vision code.

The contour-based approach was also implemented and evaluated, but produced more recognition errors on the scrolling display.

EasyOCR was evaluated as an additional general-purpose OCR comparison.

### Target Message

> **AIR FILTER IS BLOCKED**
> Observed display duration: **Approximately 10.14 seconds**

---

## Conclusion

This project demonstrates multiple approaches for HIL test automation using computer vision.

For the supplied datasets, display-specific approaches are more suitable than general-purpose OCR because they take advantage of the known 16-segment display structure and character appearance.

**The final recommended approaches are:**

| Task | Recommended Approach |
|---|---|
| Needle Gauge | HSV masking + Hough Line Detection + calibration |
| Text Detection – Method 1 | 16-segment pattern recognition |
| Text Detection – Method 2 | Character template matching |
| Text Detection – Method 3 | Contour and shape-based recognition |
| Optional comparison | EasyOCR |

The results demonstrate that custom computer vision techniques can provide accurate and deterministic detection for structured HIL display interfaces.