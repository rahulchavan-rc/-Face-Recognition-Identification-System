"""ArcFace embeddings and SCRFD face detection via InsightFace."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    """A detected face and its normalized ArcFace embedding."""

    bbox: tuple[float, float, float, float]
    detection_score: float
    embedding: np.ndarray


class FaceModel:
    """Load SCRFD and ArcFace, then detect faces and generate embeddings."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_and_embed(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect every face in a BGR image and return ArcFace embeddings."""
        if image is None or image.size == 0:
            raise ValueError("The image is empty.")

        detected_faces = self.app.get(image)
        results: list[DetectedFace] = []
        for face in detected_faces:
            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue

            normalized_embedding = embedding / norm
            results.append(
                DetectedFace(
                    bbox=tuple(float(value) for value in face.bbox),
                    detection_score=float(face.det_score),
                    embedding=normalized_embedding,
                )
            )
        return results

    def detect_and_embed_file(self, image_path: str | Path) -> list[DetectedFace]:
        """Read an image file and return detected faces with embeddings."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_and_embed(image)