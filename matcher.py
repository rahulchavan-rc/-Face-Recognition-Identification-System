"""Similarity-based matching for ArcFace embeddings."""

from dataclasses import dataclass

import numpy as np

from database import FaceDatabase


@dataclass
class MatchResult:
    """The best database match and its cosine similarity score."""

    name: str
    similarity: float
    is_unknown: bool


class FaceMatcher:
    """Compare a query embedding with enrolled reference embeddings."""

    def __init__(self, database: FaceDatabase, threshold: float = 0.45) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("The threshold must be between 0.0 and 1.0.")
        self.database = database
        self.threshold = threshold

    def match(self, embedding: np.ndarray) -> MatchResult:
        """Return the closest identity or Unknown when confidence is too low."""
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("The query embedding cannot have zero length.")
        query = query / query_norm

        enrolled = self.database.load_all()
        if not enrolled:
            return MatchResult("Unknown", 0.0, True)

        scores = []
        for name, reference in enrolled:
            if reference.shape != query.shape:
                raise ValueError(f"Embedding dimension mismatch for enrolled person '{name}'.")
            # Normalized vectors make the dot product equal to cosine similarity.
            scores.append((name, float(np.dot(query, reference))))

        best_name, best_score = max(scores, key=lambda item: item[1])
        is_unknown = best_score < self.threshold
        return MatchResult(
            name="Unknown" if is_unknown else best_name,
            similarity=best_score,
            is_unknown=is_unknown,
        )