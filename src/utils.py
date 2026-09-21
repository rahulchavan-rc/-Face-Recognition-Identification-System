"""
src/utils.py
============
Shared helpers for image I/O, validation, drawing, and result formatting.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (
    COLOR_KNOWN,
    COLOR_TEXT,
    COLOR_UNKNOWN,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    MAX_IMAGE_BYTES,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Image loading and validation
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """
    Read an image file as a BGR NumPy array.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    np.ndarray
        BGR image array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported or the image cannot be decoded.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{p.suffix}'.  "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    image = cv2.imread(str(p))
    if image is None:
        raise ValueError(f"OpenCV could not decode the image: {p}")
    return image


def load_image_from_bytes(data: bytes, filename: str = "upload") -> np.ndarray:
    """
    Decode a raw bytes buffer (e.g., from an HTTP upload) into a BGR array.

    Raises
    ------
    ValueError
        If the data is empty, too large, unsupported format, or not decodable.
    """
    if not data:
        raise ValueError("Image data is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image size {len(data) / 1_048_576:.1f} MB exceeds the "
            f"{MAX_IMAGE_BYTES // 1_048_576} MB limit."
        )
    ext = Path(filename).suffix.lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'.  "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Could not decode image '{filename}'.  "
            "The file may be corrupt or not a supported image format."
        )
    return image


def validate_image(image: np.ndarray) -> None:
    """
    Raise ValueError when an image array is clearly invalid.

    Parameters
    ----------
    image:
        BGR image array to validate.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy ndarray, got None or wrong type.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image; got shape {image.shape}."
        )
    if image.size == 0:
        raise ValueError("Image is empty (zero size).")


# ---------------------------------------------------------------------------
# Safe filesystem names
# ---------------------------------------------------------------------------

def safe_name(name: str) -> str:
    """
    Sanitize a person's name so it can be used as a file/directory name.

    * Strips leading/trailing whitespace.
    * Collapses internal whitespace to a single underscore.
    * Removes characters that are invalid in cross-platform filenames.
    * Raises ValueError when the result would be empty.
    """
    cleaned = name.strip()
    cleaned = re.sub(r"[\s]+", "_", cleaned)          # spaces → _
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", cleaned)  # illegal chars
    cleaned = cleaned.strip("._")                      # remove leading/trailing dots and underscores
    if not cleaned:
        raise ValueError(f"Person name '{name}' produced an empty sanitized name.")
    return cleaned


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_face_box(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    label: str,
    similarity: float,
    is_unknown: bool,
) -> None:
    """
    Draw a bounding box and identity label on ``frame`` in-place.

    Parameters
    ----------
    frame:
        BGR image to draw on.
    bbox:
        ``(left, top, right, bottom)`` in pixel coordinates.
    label:
        Person's name or ``"Unknown"``.
    similarity:
        Cosine similarity score.
    is_unknown:
        Whether the face was rejected as unknown.
    """
    left, top, right, bottom = (int(v) for v in bbox)
    box_color = COLOR_UNKNOWN if is_unknown else COLOR_KNOWN

    # Bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, LABEL_THICKNESS)

    # Label background.
    display_name = "UNKNOWN" if is_unknown else label
    text = f"{display_name}  {similarity:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
    )
    label_y = max(top - 10, text_h + baseline + 4)
    cv2.rectangle(
        frame,
        (left, label_y - text_h - baseline - 4),
        (left + text_w + 8, label_y + 2),
        box_color,
        cv2.FILLED,
    )

    # Label text.
    cv2.putText(
        frame,
        text,
        (left + 4, label_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        COLOR_TEXT,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_result(name: str, similarity: float, threshold: float) -> str:
    """Return a human-readable one-line identification result."""
    is_unknown = similarity < threshold
    status = "UNKNOWN" if is_unknown else f"Recognized as: {name}"
    return f"{status}  |  similarity={similarity:.3f}  |  threshold={threshold:.3f}"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def collect_image_paths(directory: str | Path) -> list[Path]:
    """
    Recursively collect all supported image files under ``directory``.

    Returns a sorted list; subdirectory structure is NOT assumed.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths


def eprint(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
    """Print to stderr (useful for error messages in CLI mode)."""
    print(*args, file=sys.stderr, **kwargs)
