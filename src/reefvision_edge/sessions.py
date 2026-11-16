from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from reefvision_edge.db import Database
from reefvision_edge.domain import RegisteredModel
from reefvision_edge.inference import YoloOnnxDetector
from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_registry import ModelRegistrationError, ModelRegistry
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sources import RecordedVideoSource, SourceError, V4L2CameraSource


@dataclass(frozen=True)
class SessionRecord:
    id: str
    model_id: str
    source_id: str
    status: str
    sample_every: int
    provider: str | None
    frame_count: int
    detection_count: int
    inference_fps: float | None
    average_latency_ms: float | None
    dropped_frames: int
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


DetectorFactory = Callable[[RegisteredModel], YoloOnnxDetector]


class SessionRunner:
    def __init__(
        self,
        database: Database,
        model_registry: ModelRegistry,
        media_store: MediaStore,
        previews: PreviewStore,
        *,
        detector_factory: DetectorFactory = YoloOnnxDetector,
        max_workers: int = 1,
    ) -> None:
        self.database = database
        self.model_registry = model_registry
        self.media_store = media_store
        self.previews = previews
        self.detector_factory = detector_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="reefvision-edge-session"
        )
        self._lock = threading.Lock()
        self._cancel: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[None]] = {}

    def start(self, session_id: str) -> None:
        with self._lock:
            existing = self._futures.get(session_id)
            if existing is not None and not existing.done():
                raise RuntimeError(f"Session {session_id} is already running.")
            event = threading.Event()
            self._cancel[session_id] = event
            self._futures[session_id] = self._executor.submit(self.run, session_id, event)

    def stop(self, session_id: str) -> bool:
        with self._lock:
            event = self._cancel.get(session_id)
        if event is None:
            return False
        event.set()
        return True

    def wait(self, session_id: str, timeout: float | None = None) -> None:
        with self._lock:
            future = self._futures.get(session_id)
        if future is not None:
            future.result(timeout=timeout)

    def run(
        self, session_id: str, cancel_event: threading.Event | None = None
    ) -> None:
        canceled = cancel_event or threading.Event()
        started = time.perf_counter()
        processed = 0
        detection_count = 0
        latency_total = 0.0
        try:
            record = _get_session(self.database, session_id)
            model = self.model_registry.get(record.model_id)
            detector = self.detector_factory(model)
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE sessions SET status = 'running', provider = ?,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        error_message = NULL
                    WHERE id = ? AND status = 'queued'
                    """,
                    (detector.model.provider, session_id),
                )
            source_record = self.media_store.get_record(record.source_id)
            source_path = Path(str(source_record["path"]))
            camera: V4L2CameraSource | None = None
            if source_record["kind"] == "video":
                if not source_path.exists():
                    raise SourceError(f"Source media {source_path.name} is unavailable.")
                source_frames = RecordedVideoSource(source_path).frames(record.sample_every)
            elif source_record["kind"] == "camera":
                camera = V4L2CameraSource(source_path)
                source_frames = camera.frames(queue_size=1)
            else:
                raise SourceError(f"Unsupported source kind {source_record['kind']}.")
            for source_frame in source_frames:
                if canceled.is_set():
                    self._finish(
                        session_id,
                        "canceled",
                        processed,
                        detection_count,
                        started,
                        latency_total,
                    )
                    return
                result = detector.predict(source_frame.frame)
                frame_id = str(uuid.uuid4())
                with self.database.connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO frames (
                            id, session_id, frame_index, timestamp_ms, width, height,
                            latency_ms, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                        """,
                        (
                            frame_id,
                            session_id,
                            source_frame.frame_index,
                            source_frame.timestamp_ms,
                            source_frame.frame.shape[1],
                            source_frame.frame.shape[0],
                            result.latency_ms,
                        ),
                    )
                    connection.executemany(
                        """
                        INSERT INTO detections (
                            id, frame_id, class_id, class_name, score,
                            x_min, y_min, x_max, y_max
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                str(uuid.uuid4()),
                                frame_id,
                                detection.class_id,
                                detection.class_name,
                                detection.score,
                                *detection.box,
                            )
                            for detection in result.detections
                        ],
                    )
                processed += 1
                detection_count += len(result.detections)
                latency_total += result.latency_ms
                self.previews.put(
                    session_id,
                    source_frame.frame_index,
                    source_frame.timestamp_ms,
                    source_frame.frame,
                    result.detections,
                )
                self._update_progress(
                    session_id,
                    processed,
                    detection_count,
                    started,
                    latency_total,
                    camera.dropped_frames if camera else 0,
                )
            terminal = "canceled" if canceled.is_set() else "completed"
            self._finish(
                session_id, terminal, processed, detection_count, started, latency_total
            )
        except (SourceError, ModelRegistrationError, OSError, sqlite3.Error, RuntimeError) as exc:
            self._fail(session_id, str(exc), processed, detection_count, started, latency_total)

    def _update_progress(
        self,
        session_id: str,
        processed: int,
        detections: int,
        started: float,
        latency_total: float,
        dropped_frames: int = 0,
    ) -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE sessions SET frame_count = ?, detection_count = ?,
                    inference_fps = ?, average_latency_ms = ?,
                    dropped_frames = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (
                    processed,
                    detections,
                    processed / elapsed,
                    latency_total / processed,
                    dropped_frames,
                    session_id,
                ),
            )

    def _finish(
        self,
        session_id: str,
        status: str,
        processed: int,
        detections: int,
        started: float,
        latency_total: float,
    ) -> None:
        if processed:
            self._update_progress(
                session_id, processed, detections, started, latency_total
            )
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE sessions SET status = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (status, session_id),
            )

    def _fail(
        self,
        session_id: str,
        message: str,
        processed: int,
        detections: int,
        started: float,
        latency_total: float,
    ) -> None:
        if processed:
            self._update_progress(session_id, processed, detections, started, latency_total)
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE sessions SET status = 'failed', error_message = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (message, session_id),
            )


class SessionService:
    def __init__(self, database: Database, runner: SessionRunner) -> None:
        self.database = database
        self.runner = runner

    def create(self, model_id: str, source_id: str, *, sample_every: int = 1) -> SessionRecord:
        if sample_every < 1:
            raise ValueError("sample_every must be at least one")
        self.runner.model_registry.get(model_id)
        self.runner.media_store.get_record(source_id)
        session_id = str(uuid.uuid4())
        settings = json.dumps({"sample_every": sample_every}, separators=(",", ":"))
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, model_id, source_id, status, settings_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (session_id, model_id, source_id, settings),
            )
        return self.get(session_id)

    def start(self, session_id: str) -> SessionRecord:
        record = self.get(session_id)
        if record.status != "queued":
            raise ValueError(f"Session {session_id} cannot start from {record.status}.")
        self.runner.start(session_id)
        return self.get(session_id)

    def stop(self, session_id: str) -> SessionRecord:
        record = self.get(session_id)
        if record.status == "queued" and not self.runner.stop(session_id):
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE sessions SET status = 'canceled',
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ?
                    """,
                    (session_id,),
                )
        elif record.status == "running":
            self.runner.stop(session_id)
        return self.get(session_id)

    def get(self, session_id: str) -> SessionRecord:
        return _get_session(self.database, session_id)

    def list(self, limit: int = 100) -> list[SessionRecord]:
        with self.database.connect() as connection:
            identifiers = connection.execute(
                "SELECT id FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get(row[0]) for row in identifiers]


def _get_session(database: Database, session_id: str) -> SessionRecord:
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT id, model_id, source_id, status, settings_json, provider,
                frame_count, detection_count, inference_fps, average_latency_ms,
                dropped_frames, error_message, created_at, updated_at, completed_at
            FROM sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        raise KeyError(session_id)
    settings = json.loads(row[4])
    return SessionRecord(
        id=row[0],
        model_id=row[1],
        source_id=row[2],
        status=row[3],
        sample_every=int(settings["sample_every"]),
        provider=row[5],
        frame_count=row[6],
        detection_count=row[7],
        inference_fps=row[8],
        average_latency_ms=row[9],
        dropped_frames=row[10],
        error_message=row[11],
        created_at=row[12],
        updated_at=row[13],
        completed_at=row[14],
    )
