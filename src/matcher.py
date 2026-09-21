"""
src/matcher.py
==============
Cosine-similarity matching with configurable unknown-rejection threshold.

How cosine similarity works
----------------------------
Two face embeddings ``a`` and ``b`` (both L2-normalized to unit length) are
compared by computing their dot product:

    similarity = a · b  (equivalent to cosine similarity when both are unit vectors)

A similarity of 1.0 means the vectors point in exactly the same direction
(identical faces in theory), and 0.0 means they are orthogonal (completely
unrelated).  Values below 0.0 are theoretically possible but rare in practice
for ArcFace embeddings.

Threshold decision
------------------
    if similarity >= MATCH_THRESHOLD:
        identity = matched_person      # recognized
    else:
        identity = "Unknown"           # rejected

The threshold is NOT universally optimal.  Use the evaluation sweep in
``evaluation/evaluate.py`` to choose the value that best balances False
Acceptance Rate and False Rejection Rate on your validation dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import MATCH_THRESHOLD


@dataclass(frozen=True)
class MatchResult:
    """The outcome of a single matching attempt against the enrolled database."""

    # Person's name when recognized, or "Unknown" when rejected.
    name: str

    # Best cosine similarity found (between the query and the closest enrolled
    # reference).  Always in range [-1, 1].
    similarity: float

    # True when the similarity was below the threshold and the face was
    # classified as unknown.
    is_unknown: bool

    def __str__(self) -> str:
        status = "UNKNOWN" if self.is_unknown else self.name
        return f"{status} (similarity={self.similarity:.3f})"


class FaceMatcher:
    """
    Compare a query embedding against all enrolled reference embeddings and
    return the best match or "Unknown".

    Parameters
    ----------
    threshold:
        Minimum cosine similarity required to accept a match.
        Defaults to ``config.MATCH_THRESHOLD`` (0.45).

    Examples
    --------
    >>> matcher = FaceMatcher(threshold=0.50)
    >>> result = matcher.match_against(query_embedding, enrolled_dict)
    >>> print(result)
    Alice (similarity=0.812)
    """

    def __init__(self, threshold: float = MATCH_THRESHOLD) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold must be in [0.0, 1.0]; got {threshold!r}."
            )
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Primary interface — works directly with a plain dict or list
    # ------------------------------------------------------------------

    def match_against(
        self,
        query: np.ndarray,
        enrolled: dict[str, np.ndarray] | list[tuple[str, np.ndarray]],
    ) -> MatchResult:
        """
        Find the best match for ``query`` among ``enrolled`` embeddings.

        Parameters
        ----------
        query:
            A 1-D float32 embedding vector.  It does not have to be
            pre-normalized; this method normalizes it internally.
        enrolled:
            Either a ``dict`` mapping person name → reference embedding, or a
            list of ``(name, embedding)`` tuples (the format returned by
            ``FaceDatabase.load_all()``).

        Returns
        -------
        MatchResult
            The closest person and their similarity score.  If the score is
            below the threshold, ``is_unknown`` is True and ``name`` is
            ``"Unknown"``.

        Raises
        ------
        ValueError
            If ``query`` has zero norm (meaningless embedding).
        """
        # Normalize the query vector.
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            raise ValueError("Query embedding has zero norm — invalid input.")
        q = q / norm

        # Flatten enrolled to a list of (name, vector) pairs.
        if isinstance(enrolled, dict):
            pairs: list[tuple[str, np.ndarray]] = list(enrolled.items())
        else:
            pairs = list(enrolled)

        if not pairs:
            # Empty database → always unknown.
            return MatchResult(name="Unknown", similarity=0.0, is_unknown=True)

        # Compute cosine similarities in bulk.
        scores: list[tuple[str, float]] = []
        for name, ref in pairs:
            r = np.asarray(ref, dtype=np.float32).reshape(-1)
            if r.shape != q.shape:
                continue  # Skip mismatched dimensions gracefully.
            # Dot product of two unit vectors = cosine similarity.
            scores.append((name, float(np.dot(q, r))))

        if not scores:
            return MatchResult(name="Unknown", similarity=0.0, is_unknown=True)

        best_name, best_score = max(scores, key=lambda item: item[1])
        is_unknown = best_score < self.threshold
        return MatchResult(
            name="Unknown" if is_unknown else best_name,
            similarity=best_score,
            is_unknown=is_unknown,
        )

    def all_scores(
        self,
        query: np.ndarray,
        enrolled: dict[str, np.ndarray] | list[tuple[str, np.ndarray]],
    ) -> list[tuple[str, float]]:
        """
        Return cosine similarities against every enrolled person, sorted
        descending.  Useful for debugging and threshold calibration.
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            raise ValueError("Query embedding has zero norm.")
        q = q / norm

        if isinstance(enrolled, dict):
            pairs: list[tuple[str, np.ndarray]] = list(enrolled.items())
        else:
            pairs = list(enrolled)

        scores = []
        for name, ref in pairs:
            r = np.asarray(ref, dtype=np.float32).reshape(-1)
            if r.shape == q.shape:
                scores.append((name, float(np.dot(q, r))))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores
