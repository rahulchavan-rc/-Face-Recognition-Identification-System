"""
src/enrollment.py
=================
Enrollment pipeline: detect → embed → average → store.

When a person is enrolled with multiple images, their embeddings are averaged
into a single reference vector.  Averaging over several clear photos of a
person smooths out pose and lighting variation, producing a more robust
reference than any single image would.

The enrollment result is stored in:
  1. SQLite database (``faces.db``) — fast runtime lookup.
  2. ``database/embeddings/<name>.npy`` — portable NumPy export.
  3. ``database/metadata.json`` — human-readable registry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import EMBEDDINGS_DIR, METADATA_PATH, SUPPORTED_EXTENSIONS
from database import FaceDatabase
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.utils import safe_name


class EnrollmentError(Exception):
    """Raised when an enrollment image cannot be processed."""


def _load_metadata() -> dict[str, object]:
    """Load the metadata.json file, returning an empty dict if it doesn't exist."""
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_metadata(metadata: dict[str, object]) -> None:
    """Write the metadata dict to disk, creating parent directories as needed."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _export_npy(name: str, embedding: np.ndarray) -> Path:
    """Save a reference embedding as a .npy file and return its path."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = EMBEDDINGS_DIR / f"{safe_name(name)}.npy"
    np.save(str(npy_path), embedding)
    return npy_path


class FaceEnroller:
    """
    Enroll one or more face images for a person.

    Parameters
    ----------
    database:
        The ``FaceDatabase`` instance to write the reference embedding into.
    detector:
        ``FaceDetector`` used to locate faces in enrollment images.
    embedder:
        ``FaceEmbedder`` used to compute ArcFace vectors.
    """

    def __init__(
        self,
        database: FaceDatabase,
        detector: FaceDetector,
        embedder: FaceEmbedder,
    ) -> None:
        self._db = database
        self._detector = detector
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll_from_paths(
        self, name: str, image_paths: list[str | Path]
    ) -> dict[str, object]:
        """
        Enroll a person from a list of image file paths.

        Each image must contain **exactly one** clearly visible face.
        Images with zero or multiple faces are rejected with a warning
        but do not abort the enrollment if at least one valid image remains.

        Parameters
        ----------
        name:
            Display name stored in the database.
        image_paths:
            One or more paths to face images (JPG, PNG, BMP).

        Returns
        -------
        dict
            Summary containing ``person``, ``accepted``, ``rejected``,
            ``embedding_dim``, and ``npy_path``.

        Raises
        ------
        EnrollmentError
            If no valid face images are found.
        ValueError
            If ``name`` is empty.
        """
        if not name.strip():
            raise ValueError("Person name cannot be empty.")
        if not image_paths:
            raise EnrollmentError("No image paths provided for enrollment.")

        valid_embeddings: list[np.ndarray] = []
        rejected: list[str] = []

        for path in image_paths:
            path = Path(path)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected.append(f"{path.name}: unsupported format")
                continue
            try:
                faces = self._detector.detect_file(path)
            except FileNotFoundError as exc:
                rejected.append(f"{path.name}: {exc}")
                continue

            if len(faces) == 0:
                rejected.append(f"{path.name}: no face detected")
                continue
            if len(faces) > 1:
                rejected.append(f"{path.name}: {len(faces)} faces detected (need exactly 1)")
                continue

            try:
                embedding = self._embedder.embed(faces[0])
            except (ValueError, RuntimeError) as exc:
                rejected.append(f"{path.name}: embedding failed — {exc}")
                continue

            valid_embeddings.append(embedding)

        if not valid_embeddings:
            detail = "; ".join(rejected) if rejected else "no images provided"
            raise EnrollmentError(
                f"Could not enroll '{name}': no valid face images were accepted. "
                f"Rejection details: {detail}"
            )

        return self._finalize(name, valid_embeddings, rejected)

    def enroll_from_arrays(
        self, name: str, images: list[np.ndarray]
    ) -> dict[str, object]:
        """
        Enroll a person from in-memory BGR image arrays (e.g., from API uploads).

        Parameters
        ----------
        name:
            Display name.
        images:
            List of BGR image arrays.
        """
        if not name.strip():
            raise ValueError("Person name cannot be empty.")
        if not images:
            raise EnrollmentError("No images provided for enrollment.")

        valid_embeddings: list[np.ndarray] = []
        rejected: list[str] = []

        for i, image in enumerate(images):
            label = f"image[{i}]"
            try:
                faces = self._detector.detect(image)
            except ValueError as exc:
                rejected.append(f"{label}: {exc}")
                continue

            if len(faces) == 0:
                rejected.append(f"{label}: no face detected")
                continue
            if len(faces) > 1:
                rejected.append(f"{label}: {len(faces)} faces detected (need exactly 1)")
                continue

            try:
                embedding = self._embedder.embed(faces[0])
            except (ValueError, RuntimeError) as exc:
                rejected.append(f"{label}: embedding failed — {exc}")
                continue

            valid_embeddings.append(embedding)

        if not valid_embeddings:
            detail = "; ".join(rejected) if rejected else "unknown"
            raise EnrollmentError(
                f"Could not enroll '{name}': no valid face images. Details: {detail}"
            )

        return self._finalize(name, valid_embeddings, rejected)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalize(
        self,
        name: str,
        valid_embeddings: list[np.ndarray],
        rejected: list[str],
    ) -> dict[str, object]:
        """Average embeddings, persist to SQLite + .npy + metadata.json."""
        # Average multiple embeddings for a more stable reference.
        matrix = np.vstack(valid_embeddings)           # shape: (N, 512)
        reference = np.mean(matrix, axis=0)            # shape: (512,)
        norm = float(np.linalg.norm(reference))
        if norm > 0:
            reference = reference / norm               # re-normalize after averaging

        # 1. SQLite storage (primary runtime store).
        self._db.save(name, reference, sample_count=len(valid_embeddings))

        # 2. Export .npy file.
        npy_path = _export_npy(name, reference)

        # 3. Update metadata.json.
        metadata = _load_metadata()
        metadata[name] = {
            "name": name,
            "sample_count": len(valid_embeddings),
            "embedding_dim": int(reference.shape[0]),
            "npy_file": str(npy_path.relative_to(npy_path.parents[1])),
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_metadata(metadata)

        return {
            "person": name,
            "accepted": len(valid_embeddings),
            "rejected": rejected,
            "embedding_dim": int(reference.shape[0]),
            "npy_path": str(npy_path),
        }


def enroll_dataset_folder(
    dataset_dir: str | Path,
    database: FaceDatabase,
    detector: FaceDetector,
    embedder: FaceEmbedder,
) -> list[dict[str, object]]:
    """
    Bulk-enroll every person folder found in ``dataset_dir``.

    Expected layout::

        dataset_dir/
            Alice/
                alice1.jpg
                alice2.jpg
            Bob/
                bob1.jpg

    Parameters
    ----------
    dataset_dir:
        Root directory containing one sub-folder per person.
    database, detector, embedder:
        Shared instances passed to ``FaceEnroller``.

    Returns
    -------
    list[dict]
        One summary dict per person (same format as ``enroll_from_paths``).
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    person_folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not person_folders:
        raise ValueError(f"No person sub-folders found in: {root}")

    enroller = FaceEnroller(database, detector, embedder)
    results: list[dict[str, object]] = []

    for folder in person_folders:
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_paths:
            print(f"  Skipping '{folder.name}': no supported images found.")
            continue

        try:
            summary = enroller.enroll_from_paths(folder.name, image_paths)
            accepted = summary["accepted"]
            total = len(image_paths)
            print(f"  Enrolled '{folder.name}' — {accepted}/{total} images accepted.")
            if summary["rejected"]:
                for msg in summary["rejected"]:
                    print(f"    Rejected: {msg}")
            results.append(summary)
        except EnrollmentError as exc:
            print(f"  ERROR enrolling '{folder.name}': {exc}")

    return results
