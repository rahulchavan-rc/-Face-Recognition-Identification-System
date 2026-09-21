"""
tests/test_matcher.py
=====================
Unit tests for the FaceMatcher in src/matcher.py.

Tests covered
-------------
1. Same person's embeddings → recognized (similarity ≥ threshold)
2. Different person's embeddings → rejected (orthogonal vectors)
3. Similarity below threshold → UNKNOWN
4. Empty database → UNKNOWN with similarity 0.0
5. Zero embedding → ValueError
6. Invalid threshold → ValueError
7. Multiple enrolled persons → correct best-match selected
8. Embedding dimension mismatch → graceful UNKNOWN
9. All scores API → returns sorted list
10. Frozen MatchResult → is immutable dataclass

These tests use only NumPy arrays — no model loading required.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make sure project root is on sys.path when running from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matcher import FaceMatcher, MatchResult


def _unit(v: list[float]) -> np.ndarray:
    """Return a L2-normalized float32 array from a list of floats."""
    arr = np.array(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


class TestFaceMatcher(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Same person's embeddings → recognized
    # ------------------------------------------------------------------
    def test_same_embedding_is_recognized(self) -> None:
        """Identical embeddings should always match above any reasonable threshold."""
        ref = _unit([1.0, 0.0, 0.0, 0.0])
        enrolled = {"Alice": ref}
        matcher = FaceMatcher(threshold=0.80)

        result = matcher.match_against(ref, enrolled)

        self.assertEqual(result.name, "Alice")
        self.assertAlmostEqual(result.similarity, 1.0, places=4)
        self.assertFalse(result.is_unknown)

    # ------------------------------------------------------------------
    # 2. Different person's embeddings → rejected (orthogonal vectors)
    # ------------------------------------------------------------------
    def test_orthogonal_embedding_is_rejected(self) -> None:
        """Orthogonal unit vectors have cosine similarity 0 — well below any useful threshold."""
        enrolled = {"Alice": _unit([1.0, 0.0, 0.0])}
        query = _unit([0.0, 1.0, 0.0])  # perpendicular to Alice
        matcher = FaceMatcher(threshold=0.45)

        result = matcher.match_against(query, enrolled)

        self.assertEqual(result.name, "Unknown")
        self.assertTrue(result.is_unknown)
        self.assertAlmostEqual(result.similarity, 0.0, places=4)

    # ------------------------------------------------------------------
    # 3. Similarity below threshold → UNKNOWN
    # ------------------------------------------------------------------
    def test_below_threshold_returns_unknown(self) -> None:
        """A face that is a close but not close-enough match should be rejected."""
        alice = _unit([1.0, 0.1, 0.0])
        query = _unit([0.8, 0.6, 0.0])  # similar but not identical
        enrolled = {"Alice": alice}

        # Use a high threshold to force rejection.
        matcher = FaceMatcher(threshold=0.99)
        result = matcher.match_against(query, enrolled)

        self.assertEqual(result.name, "Unknown")
        self.assertTrue(result.is_unknown)

    # ------------------------------------------------------------------
    # 4. Empty database → UNKNOWN with similarity 0.0
    # ------------------------------------------------------------------
    def test_empty_database_returns_unknown(self) -> None:
        """With no enrolled persons the result must always be Unknown."""
        matcher = FaceMatcher(threshold=0.45)
        result = matcher.match_against(_unit([1.0, 0.0, 0.0]), {})

        self.assertEqual(result.name, "Unknown")
        self.assertTrue(result.is_unknown)
        self.assertEqual(result.similarity, 0.0)

    # ------------------------------------------------------------------
    # 5. Zero embedding → ValueError
    # ------------------------------------------------------------------
    def test_zero_embedding_raises_value_error(self) -> None:
        """A zero-norm embedding is meaningless and should raise ValueError."""
        matcher = FaceMatcher(threshold=0.45)
        with self.assertRaises(ValueError):
            matcher.match_against(np.zeros(3, dtype=np.float32), {"Alice": _unit([1.0, 0.0, 0.0])})

    # ------------------------------------------------------------------
    # 6. Invalid threshold → ValueError
    # ------------------------------------------------------------------
    def test_threshold_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            FaceMatcher(threshold=1.1)

    def test_threshold_below_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            FaceMatcher(threshold=-0.01)

    def test_boundary_thresholds_are_valid(self) -> None:
        """0.0 and 1.0 are valid boundary values."""
        FaceMatcher(threshold=0.0)
        FaceMatcher(threshold=1.0)

    # ------------------------------------------------------------------
    # 7. Multiple enrolled persons → correct best-match selected
    # ------------------------------------------------------------------
    def test_selects_closest_enrolled_person(self) -> None:
        """With multiple enrollees the person with the highest similarity wins."""
        alice = _unit([1.0, 0.0, 0.0, 0.0])
        bob   = _unit([0.0, 1.0, 0.0, 0.0])
        carol = _unit([0.0, 0.0, 1.0, 0.0])

        enrolled = {"Alice": alice, "Bob": bob, "Carol": carol}
        # Query close to Alice.
        query = _unit([0.9, 0.1, 0.0, 0.0])
        matcher = FaceMatcher(threshold=0.50)

        result = matcher.match_against(query, enrolled)

        self.assertEqual(result.name, "Alice")
        self.assertFalse(result.is_unknown)

    # ------------------------------------------------------------------
    # 8. Embedding dimension mismatch → graceful UNKNOWN
    # ------------------------------------------------------------------
    def test_dimension_mismatch_returns_unknown(self) -> None:
        """If the enrolled embedding has a different shape, skip and return Unknown."""
        enrolled = {"Alice": np.array([1.0, 0.0], dtype=np.float32)}  # 2-D
        query = _unit([1.0, 0.0, 0.0])  # 3-D
        matcher = FaceMatcher(threshold=0.45)

        result = matcher.match_against(query, enrolled)

        self.assertEqual(result.name, "Unknown")
        self.assertTrue(result.is_unknown)

    # ------------------------------------------------------------------
    # 9. all_scores API → returns sorted list
    # ------------------------------------------------------------------
    def test_all_scores_sorted_descending(self) -> None:
        enrolled = {
            "Alice": _unit([1.0, 0.0, 0.0]),
            "Bob":   _unit([0.0, 1.0, 0.0]),
            "Carol": _unit([0.7, 0.7, 0.0]),
        }
        query = _unit([1.0, 0.0, 0.0])
        matcher = FaceMatcher(threshold=0.45)

        scores = matcher.all_scores(query, enrolled)

        self.assertEqual(len(scores), 3)
        # Must be descending.
        for (_, s1), (_, s2) in zip(scores, scores[1:]):
            self.assertGreaterEqual(s1, s2)
        # Alice should be on top.
        self.assertEqual(scores[0][0], "Alice")

    # ------------------------------------------------------------------
    # 10. MatchResult is an immutable frozen dataclass
    # ------------------------------------------------------------------
    def test_match_result_is_frozen(self) -> None:
        result = MatchResult(name="Alice", similarity=0.87, is_unknown=False)
        with self.assertRaises((AttributeError, TypeError)):
            result.name = "Bob"  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 11. list-of-tuples enrolled format also works
    # ------------------------------------------------------------------
    def test_list_of_tuples_enrolled_format(self) -> None:
        """FaceDatabase.load_all() returns list[tuple[str, ndarray]] — must work."""
        ref = _unit([1.0, 0.0, 0.0])
        enrolled = [("Alice", ref)]
        matcher = FaceMatcher(threshold=0.80)

        result = matcher.match_against(ref, enrolled)

        self.assertEqual(result.name, "Alice")
        self.assertFalse(result.is_unknown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
