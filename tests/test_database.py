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
    assert schema_version == "2"


def test_initialize_migrates_v1_session_and_frame_columns_idempotently(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "edge.sqlite3")
    with sqlite3.connect(database.path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '1');
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, model_id TEXT NOT NULL, source_id TEXT NOT NULL,
                status TEXT NOT NULL, settings_json TEXT NOT NULL, provider TEXT,
                frame_count INTEGER NOT NULL DEFAULT 0,
                detection_count INTEGER NOT NULL DEFAULT 0, inference_fps REAL,
                average_latency_ms REAL, dropped_frames INTEGER NOT NULL DEFAULT 0,
                error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE frames (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, frame_index INTEGER NOT NULL,
                timestamp_ms REAL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                latency_ms REAL NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(session_id, frame_index)
            );
            """
        )

    database.initialize()
    database.initialize()

    with sqlite3.connect(database.path) as connection:
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        frame_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(frames)")
        }
        version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    assert {
        "requested_compute",
        "actual_device",
        "provider_reason",
        "input_fps",
        "median_latency_ms",
        "p95_latency_ms",
        "peak_vram_mb",
        "operator_profile_json",
        "operator_profile_path",
        "monitor_video_path",
    } <= session_columns
    assert "inference_status" in frame_columns
    assert version == "2"


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
