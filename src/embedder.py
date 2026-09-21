"""
src/embedder.py
===============
Face embedding generation using ArcFace (via InsightFace).

Model information
-----------------
- **Name**: ArcFace (included in InsightFace model pack ``buffalo_l``)
- **Architecture**: ResNet-100 trained with the ArcFace loss function
- **Embedding dimension**: 512 floats (32-bit)
- **Pretrained on**: MS1MV3 (~5.8 M identities, ~93 M images)
- **Framework**: ONNX Runtime (CPU by default, GPU optional)
- **Why ArcFace?**
    ArcFace is one of the most accurate open-source face-recognition models
    available.  It uses an additive angular margin loss that forces embeddings
    of the same person to cluster tightly in high-dimensional space, making
    cosine similarity an excellent matching metric.  The buffalo_l pack is
    easy to install, works on CPU, and produces state-of-the-art accuracy on
    standard benchmarks (99.83 % on LFW).

Normalization
-------------
Raw ArcFace output vectors are L2-normalized before storage and matching.
Normalized vectors have unit length so that their dot product equals cosine
similarity directly — no division required during matching.
"""

from __future__ import annotations

import numpy as np

from src.detector import DetectedFace


class FaceEmbedder:
    """
    Generate a 512-dimensional ArcFace embedding from a detected face.

    This class is intentionally stateless — it delegates the heavy lifting
    to the InsightFace model that was already loaded by ``FaceDetector``.
    The ``DetectedFace._raw`` attribute holds the InsightFace face object
    which already carries a pre-computed ``embedding`` field.

    Embedding dimension: **512**
    """

    EMBEDDING_DIM: int = 512

    def embed(self, face: DetectedFace) -> np.ndarray:
        """
        Extract and L2-normalize the ArcFace embedding for a detected face.

        Parameters
        ----------
        face:
            A ``DetectedFace`` returned by ``FaceDetector.detect()``.  The
            ``_raw`` attribute must contain the original InsightFace face
            object (which already has an ``embedding`` field populated by
            the ArcFace model).

        Returns
        -------
        np.ndarray
            A float32 array of shape ``(512,)`` with unit L2-norm.

        Raises
        ------
        ValueError
            If the face has no embedding or the embedding has zero norm.
        RuntimeError
            If the raw InsightFace object is missing.
        """
        raw = face._raw
        if raw is None:
            raise RuntimeError(
                "The DetectedFace has no raw InsightFace object. "
                "Ensure FaceDetector.detect() was used to create this face."
            )

        raw_embedding = getattr(raw, "embedding", None)
        if raw_embedding is None:
            raise ValueError(
                "The InsightFace face object has no embedding attribute. "
                "Check that the ArcFace recognition model is loaded inside "
                "the InsightFace FaceAnalysis app."
            )

        vector = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError(
                "ArcFace produced a zero-norm embedding — this should not "
                "happen with a valid face image.  Check that the face crop "
                "is not empty or corrupted."
            )

        return vector / norm  # L2-normalized → unit vector

    def embed_batch(self, faces: list[DetectedFace]) -> list[np.ndarray]:
        """
        Embed multiple detected faces and return their normalized vectors.

        Faces that produce zero-norm embeddings are silently skipped.
        """
        embeddings: list[np.ndarray] = []
        for face in faces:
            try:
                embeddings.append(self.embed(face))
            except (ValueError, RuntimeError):
                pass
        return embeddings
