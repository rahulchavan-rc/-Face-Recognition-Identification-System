"""
src/recognition.py
==================
Face identification pipeline: detect → embed → match → return result.

This module provides the high-level ``identify_image`` and
``identify_frame`` functions used by both the CLI and the FastAPI backend.
It also contains the real-time webcam loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_CAMERA, MATCH_THRESHOLD, WEBCAM_WINDOW_TITLE
from database import FaceDatabase
from src.detector import DetectedFace, FaceDetector
from src.embedder import FaceEmbedder
from src.matcher import FaceMatcher, MatchResult
from src.utils import draw_face_box, load_image


@dataclass
class FaceResult:
    """The full result for a single face found in an image."""

    # Bounding box of the detected face.
    bbox: tuple[float, float, float, float]

    # SCRFD detection confidence.
    detection_score: float

    # Matching result (name + similarity + is_unknown).
    match: MatchResult


def identify_image(
    image_path: str | Path,
    database: FaceDatabase,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    threshold: float = MATCH_THRESHOLD,
) -> list[FaceResult]:
    """
    Identify all faces in an image file.

    Pipeline
    --------
    1. Load and validate the image.
    2. Run SCRFD face detection → bounding boxes.
    3. Generate ArcFace embeddings for each detected face.
    4. Compare each embedding against enrolled database embeddings.
    5. Apply threshold → recognized or UNKNOWN.

    Parameters
    ----------
    image_path:
        Path to the input image.
    database:
        Loaded ``FaceDatabase`` with enrolled embeddings.
    detector, embedder:
        Pre-initialized model instances (shared to avoid re-loading weights).
    threshold:
        Cosine similarity cutoff.

    Returns
    -------
    list[FaceResult]
        One entry per detected face.  Empty list if no faces were detected.
    """
    image = load_image(image_path)
    return identify_frame(image, database, detector, embedder, threshold)


def identify_frame(
    frame: np.ndarray,
    database: FaceDatabase,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    threshold: float = MATCH_THRESHOLD,
) -> list[FaceResult]:
    """
    Identify all faces in an in-memory BGR image array.

    This is the inner function used by both ``identify_image`` (for file
    input) and the webcam loop (for live frames).

    Parameters
    ----------
    frame:
        BGR image array (NumPy ndarray).
    database, detector, embedder, threshold:
        Same as ``identify_image``.

    Returns
    -------
    list[FaceResult]
        One entry per detected face.
    """
    enrolled = database.load_all()
    matcher = FaceMatcher(threshold=threshold)

    faces: list[DetectedFace] = detector.detect(frame)
    results: list[FaceResult] = []

    for face in faces:
        try:
            embedding = embedder.embed(face)
        except (ValueError, RuntimeError):
            # Skip faces where embedding generation fails (e.g., blurry crop).
            continue

        match = matcher.match_against(embedding, enrolled)
        results.append(
            FaceResult(
                bbox=face.bbox,
                detection_score=face.detection_score,
                match=match,
            )
        )

    return results


def run_webcam(
    database: FaceDatabase,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    threshold: float = MATCH_THRESHOLD,
    camera_index: int = DEFAULT_CAMERA,
) -> None:
    """
    Real-time face recognition from a webcam feed.

    For each frame the pipeline runs:
      1. SCRFD face detection.
      2. ArcFace embedding per face.
      3. Cosine similarity matching against enrolled database.
      4. Draw bounding box + name + similarity score on the frame.

    Recognized faces → green box + name + score.
    Unknown faces    → red box + "UNKNOWN" + score.

    Press ``q`` to quit.

    Parameters
    ----------
    database, detector, embedder, threshold, camera_index:
        Same semantics as ``identify_image``.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}.  "
            "Check that a webcam is connected and not in use by another app."
        )

    print(f"Webcam started on camera {camera_index}.  Press 'q' to quit.")
    print(f"Recognition threshold: {threshold:.3f}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Could not read frame from webcam — retrying.")
                continue

            results = identify_frame(frame, database, detector, embedder, threshold)

            for face_result in results:
                draw_face_box(
                    frame=frame,
                    bbox=face_result.bbox,
                    label=face_result.match.name,
                    similarity=face_result.match.similarity,
                    is_unknown=face_result.match.is_unknown,
                )

            # Overlay stats.
            n_known = sum(1 for r in results if not r.match.is_unknown)
            n_unknown = sum(1 for r in results if r.match.is_unknown)
            stats = (
                f"Faces: {len(results)}  |  Known: {n_known}  |  Unknown: {n_unknown}"
                f"  |  Threshold: {threshold:.2f}"
            )
            cv2.putText(
                frame, stats, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow(WEBCAM_WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")
