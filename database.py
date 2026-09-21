"""SQLite storage for enrolled ArcFace embeddings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


class FaceDatabase:
    """Store one normalized reference embedding for each enrolled person."""

    def __init__(self, database_path: str | Path = "faces.db") -> None:
        self.database_path = str(database_path)
        self._create_table()
        self._ensure_schema()
        self._create_config_table()
        self._create_recognition_log_table()

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
                    sample_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            columns = connection.execute("PRAGMA table_info(people)").fetchall()
            existing = {column[1] for column in columns}
            if "sample_count" not in existing:
                connection.execute(
                    "ALTER TABLE people ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 1"
                )
            if "created_at" not in existing:
                connection.execute(
                    "ALTER TABLE people ADD COLUMN created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
                )

    def _create_config_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES ('threshold', '0.45')"
            )

    def _create_recognition_log_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    threshold REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'upload',
                    message TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save(self, name: str, embedding: np.ndarray, sample_count: int = 1) -> None:
        """Insert or replace a person's reference embedding."""
        if not name.strip():
            raise ValueError("The person's name cannot be empty.")

        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(normalized)
        if norm == 0:
            raise ValueError("The embedding cannot have zero length.")
        normalized = normalized / norm

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO people (name, embedding, dimension, sample_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    sample_count = excluded.sample_count,
                    created_at = CURRENT_TIMESTAMP
                """,
                (name.strip(), normalized.tobytes(), normalized.size, max(1, int(sample_count))),
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

    def list_people(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, sample_count FROM people ORDER BY name"
            ).fetchall()
        return [
            {"name": name, "sample_count": sample_count, "status": "Active"}
            for name, sample_count in rows
        ]

    def count_people(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM people").fetchone()
        return int(row[0]) if row else 0

    def get_threshold(self) -> float:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM config WHERE key = 'threshold'").fetchone()
        return float(row[0]) if row else 0.45

    def set_threshold(self, value: float) -> None:
        threshold = float(value)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO config (key, value) VALUES ('threshold', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(threshold),),
            )

    def delete_person(self, name: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM people WHERE name = ?", (name,))
        return cursor.rowcount > 0

    def add_recognition_log(
        self,
        person: str,
        similarity: float,
        threshold: float,
        status: str,
        source: str = "upload",
        message: str = "",
    ) -> None:
        normalized_source = str(source).strip().lower()
        if normalized_source in {"upload", "image upload", "uploaded image", "image_upload", "uploaded", "file upload", "manual upload"}:
            normalized_source = "upload"
        elif normalized_source in {"webcam", "camera", "webcam recognition", "live webcam", "web source", "web-source", "web_source", "webcam source"}:
            normalized_source = "webcam"
        elif "web" in normalized_source or "camera" in normalized_source or "live" in normalized_source:
            normalized_source = "webcam"
        elif "upload" in normalized_source or "file" in normalized_source or "manual" in normalized_source:
            normalized_source = "upload"
        else:
            normalized_source = "upload" if normalized_source.startswith("upload") else "webcam"

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recognition_logs
                    (person, similarity, threshold, status, source, message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (person, float(similarity), float(threshold), status, normalized_source, message),
            )

    def list_recognition_logs(self, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, person, similarity, threshold, status, source, message, created_at
                FROM recognition_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [
            {
                "id": row[0],
                "person": row[1],
                "similarity": row[2],
                "threshold": row[3],
                "status": row[4],
                "source": row[5],
                "message": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]
