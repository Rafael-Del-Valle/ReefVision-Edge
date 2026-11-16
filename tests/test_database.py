from __future__ import annotations

import sqlite3
from pathlib import Path

from reefvision_edge.db import Database


def test_initialize_creates_versioned_field_schema(tmp_path: Path) -> None:
    database = Database(tmp_path / "edge.sqlite3")

    database.initialize()

    with sqlite3.connect(database.path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        schema_version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]

    assert {"metadata", "models", "sources", "sessions", "frames", "detections"} <= table_names
    assert schema_version == "1"


def test_restart_marks_stale_running_sessions_interrupted(tmp_path: Path) -> None:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                id, model_id, source_id, status, settings_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "session-1",
                "model-1",
                "source-1",
                "running",
                "{}",
                "2026-11-16T12:00:00Z",
                "2026-11-16T12:00:00Z",
            ),
        )

    recovered = database.recover_interrupted_sessions()

    with database.connect() as connection:
        row = connection.execute(
            "SELECT status, error_message FROM sessions WHERE id = ?", ("session-1",)
        ).fetchone()
    assert recovered == 1
    assert row == ("interrupted", "Service restarted before the session completed.")
