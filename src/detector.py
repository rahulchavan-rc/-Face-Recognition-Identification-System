"""
src/detector.py
===============
Face detection using InsightFace's SCRFD model.

SCRFD (Sample- and Computation-efficient Face Detector) is a high-accuracy,
lightweight detector that works well on CPU.  It is bundled inside the
InsightFace "buffalo_l" model pack together with ArcFace.

This module is intentionally separated from embedding so that detection and
recognition can be composed flexibly (e.g., detect all faces first, then only
embed the largest one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import DET_SIZE, EXECUTION_PROVIDER, MODEL_NAME


@dataclass
class DetectedFace:
    """A single face found by SCRFD in a BGR image frame."""

    # Bounding box: (left, top, right, bottom) in pixel coordinates.
    bbox: tuple[float, float, float, float]

    # SCRFD detection confidence in [0, 1].
    detection_score: float

    # Face area in pixels (width × height of the bounding box).
    area: float = field(init=False)

    # Raw cropped face array (BGR), set by the detector if crop=True.
    crop: np.ndarray | None = None

    # The raw InsightFace face object, kept for downstream use by FaceEmbedder.
    _raw: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        left, top, right, bottom = self.bbox
        self.area = max(0.0, (right - left) * (bottom - top))

    @property
    def bbox_int(self) -> tuple[int, int, int, int]:
        """Bounding box as integers for use with OpenCV drawing functions."""
        return tuple(int(v) for v in self.bbox)  # type: ignore[return-value]


class FaceDetector:
    """
    Detect faces in BGR images using InsightFace's SCRFD.

    Parameters
    ----------
    model_name:
        InsightFace model pack to load (default: ``buffalo_l``).
    det_size:
        Detection input resolution.  Larger values improve small-face recall.
    provider:
        ONNX Runtime execution provider (``CPUExecutionProvider`` or
        ``CUDAExecutionProvider``).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        det_size: tuple[int, int] = DET_SIZE,
        provider: str = EXECUTION_PROVIDER,
    ) -> None:
        self._app = FaceAnalysis(
            name=model_name,
            providers=[provider],
        )
        self._app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in a BGR image.

        Parameters
        ----------
        image:
            A BGR image as a NumPy array (the format returned by ``cv2.imread``
            and ``cv2.VideoCapture.read``).

        Returns
        -------
        list[DetectedFace]
            Detected faces sorted by detection score (highest first).

        Raises
        ------
        ValueError
            If the image is None or empty.
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot detect faces in an empty or None image.")

        raw_faces = self._app.get(image)
        results: list[DetectedFace] = []

        for raw in raw_faces:
            bbox = tuple(float(v) for v in raw.bbox)
            score = float(raw.det_score)
            left, top, right, bottom = (int(v) for v in raw.bbox)
            # Safely crop the face region from the image.
            h, w = image.shape[:2]
            x1, y1 = max(0, left), max(0, top)
            x2, y2 = min(w, right), min(h, bottom)
            crop = image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None
            results.append(
                DetectedFace(bbox=bbox, detection_score=score, crop=crop, _raw=raw)  # type: ignore[arg-type]
            )

        # Sort highest-confidence detections first.
        results.sort(key=lambda f: f.detection_score, reverse=True)
        return results

    def detect_file(self, image_path: str | Path) -> list[DetectedFace]:
        """
        Read an image file and detect faces.

        Parameters
        ----------
        image_path:
            Path to a supported image file (JPG, PNG, BMP).

        Raises
        ------
        FileNotFoundError
            If the file cannot be read by OpenCV.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")
        return self.detect(image)
