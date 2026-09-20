"""SQLite storage for enrolled ArcFace embeddings."""

import sqlite3
from pathlib import Path

import numpy as np


class FaceDatabase:
    """Store one normalized reference embedding for each enrolled person."""

    def __init__(self, database_path: str | Path = "faces.db") -> None:
        self.database_path = str(database_path)
        self._create_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _create_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    name TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(self, name: str, embedding: np.ndarray) -> None:
        """Insert or replace a person's reference embedding."""
        if not name.strip():
            raise ValueError("The person's name cannot be empty.")

        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(normalized)
        if norm == 0:
            raise ValueError("The embedding cannot have zero length.")
        normalized = normalized / norm

        # SQLite stores the NumPy vector as bytes; the dimension allows safe reconstruction.
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO people (name, embedding, dimension)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    created_at = CURRENT_TIMESTAMP
                """,
                (name.strip(), normalized.tobytes(), normalized.size),
            )

    def load_all(self) -> list[tuple[str, np.ndarray]]:
        """Load all enrolled names and normalized embeddings."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, embedding, dimension FROM people ORDER BY name"
            ).fetchall()

        return [
            (name, np.frombuffer(blob, dtype=np.float32, count=dimension).copy())
            for name, blob, dimension in rows
        ]