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
    requested_compute TEXT NOT NULL DEFAULT 'auto',
    actual_device TEXT,
    provider_reason TEXT,
    frame_count INTEGER NOT NULL DEFAULT 0,
    detection_count INTEGER NOT NULL DEFAULT 0,
    input_fps REAL,
    inference_fps REAL,
    average_latency_ms REAL,
    median_latency_ms REAL,
    p95_latency_ms REAL,
    peak_vram_mb REAL,
    dropped_frames INTEGER NOT NULL DEFAULT 0,
    operator_profile_json TEXT,
    operator_profile_path TEXT,
    monitor_video_path TEXT,
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
    inference_status TEXT NOT NULL DEFAULT 'inferred',
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

SESSION_MIGRATIONS = {
    "requested_compute": "TEXT NOT NULL DEFAULT 'auto'",
    "actual_device": "TEXT",
    "provider_reason": "TEXT",
    "input_fps": "REAL",
    "median_latency_ms": "REAL",
    "p95_latency_ms": "REAL",
    "peak_vram_mb": "REAL",
    "operator_profile_json": "TEXT",
    "operator_profile_path": "TEXT",
    "monitor_video_path": "TEXT",
}

FRAME_MIGRATIONS = {"inference_status": "TEXT NOT NULL DEFAULT 'inferred'"}


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
            self._add_missing_columns(connection, "sessions", SESSION_MIGRATIONS)
            self._add_missing_columns(connection, "frames", FRAME_MIGRATIONS)
            connection.execute(
                """
                INSERT INTO metadata (key, value) VALUES ('schema_version', '2')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )
        reserve = self.path.parent / ".reefvision-edge-emergency-reserve"
        if not reserve.exists():
            with reserve.open("wb") as stream:
                stream.write(b"\0" * 65536)

    @staticmethod
    def _add_missing_columns(
        connection: sqlite3.Connection, table: str, migrations: dict[str, str]
    ) -> None:
        existing = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for name, declaration in migrations.items():
            if name not in existing:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                )

    def release_emergency_reserve(self) -> None:
        (self.path.parent / ".reefvision-edge-emergency-reserve").unlink(
            missing_ok=True
        )

    def recover_interrupted_sessions(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET status = 'interrupted',
                    frame_count = (
                        SELECT COUNT(*) FROM frames WHERE frames.session_id = sessions.id
                    ),
                    detection_count = (
                        SELECT COUNT(*) FROM detections
                        JOIN frames ON frames.id = detections.frame_id
                        WHERE frames.session_id = sessions.id
                    ),
                    average_latency_ms = (
                        SELECT AVG(latency_ms) FROM frames
                        WHERE frames.session_id = sessions.id
                    ),
                    error_message = 'Service restarted before the session completed.',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'running'
                """
            )
            return cursor.rowcount
