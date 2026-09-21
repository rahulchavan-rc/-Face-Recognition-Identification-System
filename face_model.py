"""
face_model.py  (compatibility shim)
====================================
This module is kept for backward compatibility with the FastAPI backend
(backend/api/main.py) and the evaluation script (evaluate.py) that import
from it directly.

It re-exports ``FaceModel`` — a thin wrapper that combines ``FaceDetector``
and ``FaceEmbedder`` from the new ``src/`` package into the same single-class
interface that the rest of the codebase already expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from config import DET_SIZE, EXECUTION_PROVIDER, MODEL_NAME


@dataclass
class DetectedFace:
    """Backward-compatible detected-face container."""

    bbox: tuple[float, float, float, float]
    detection_score: float
    embedding: np.ndarray


class FaceModel:
    """
    Combined face detection + embedding model (SCRFD + ArcFace).

    This class provides the same API as the original ``face_model.FaceModel``
    so that existing code requires no changes.

    Model
    -----
    - **Detector**: SCRFD (part of InsightFace ``buffalo_l``)
    - **Embedder**: ArcFace ResNet-100 (pretrained on MS1MV3)
    - **Embedding dim**: 512
    - **Runtime**: ONNX Runtime (CPU by default)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        det_size: tuple[int, int] = DET_SIZE,
    ) -> None:
        self._detector = FaceDetector(
            model_name=model_name,
            det_size=det_size,
            provider=EXECUTION_PROVIDER,
        )
        self._embedder = FaceEmbedder()

    def detect_and_embed(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces and return their ArcFace embeddings."""
        raw_faces = self._detector.detect(image)
        results: list[DetectedFace] = []
        for face in raw_faces:
            try:
                emb = self._embedder.embed(face)
            except (ValueError, RuntimeError):
                continue
            results.append(
                DetectedFace(
                    bbox=face.bbox,
                    detection_score=face.detection_score,
                    embedding=emb,
                )
            )
        return results

    def detect_and_embed_file(self, image_path: str | Path) -> list[DetectedFace]:
        """Read an image file and return detected faces with embeddings."""
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_and_embed(image)