"""
matcher.py  (compatibility shim)
=================================
Kept for backward compatibility with the FastAPI backend and existing CLI code
that imports ``FaceMatcher`` and ``MatchResult`` from this module directly.

The real implementation now lives in ``src/matcher.py``.
"""

from __future__ import annotations

import numpy as np

from src.matcher import FaceMatcher, MatchResult
from database import FaceDatabase


class FaceMatcher(FaceMatcher):  # type: ignore[no-redef]
    """
    Backward-compatible FaceMatcher that also accepts a ``FaceDatabase``
    instance (legacy interface used by the FastAPI backend).

    If ``database`` is provided, calling ``match(embedding)`` loads all
    enrolled embeddings and delegates to ``match_against``.
    """

    def __init__(
        self,
        database: FaceDatabase | None = None,
        threshold: float = 0.45,
    ) -> None:
        super().__init__(threshold=threshold)
        self._database = database

    def match(self, embedding: np.ndarray) -> MatchResult:
        """Legacy method: load enrolled embeddings from the database then match."""
        if self._database is None:
            raise RuntimeError(
                "No database was provided to this FaceMatcher instance.  "
                "Use match_against(embedding, enrolled) instead."
            )
        enrolled = self._database.load_all()
        return self.match_against(embedding, enrolled)


__all__ = ["FaceMatcher", "MatchResult"]