from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    provider TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT,
    sha256 TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    provider TEXT,
    frame_count INTEGER NOT NULL DEFAULT 0,
    detection_count INTEGER NOT NULL DEFAULT 0,
    inference_fps REAL,
    average_latency_ms REAL,
    dropped_frames INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_ms REAL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, frame_index)
);

CREATE TABLE IF NOT EXISTS detections (
    id TEXT PRIMARY KEY,
    frame_id TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    score REAL NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                """
                INSERT INTO metadata (key, value) VALUES ('schema_version', '1')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )

    def recover_interrupted_sessions(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET status = 'interrupted',
                    error_message = 'Service restarted before the session completed.',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'running'
                """
            )
            return cursor.rowcount
