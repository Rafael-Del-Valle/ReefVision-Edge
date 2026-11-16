from __future__ import annotations

import json
import math
import shutil
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

from reefvision_edge.compute import ComputeChoice, sample_process_vram_mb
from reefvision_edge.db import Database
from reefvision_edge.domain import LATENCY_STATISTICS_WINDOW_FRAMES, OverlaySettings
from reefvision_edge.inference import YoloOnnxDetector
from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_registry import ModelRegistry
from reefvision_edge.monitor_video import MonitorVideoWriter, render_monitor_frame
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sources import (
    RecordedVideoSource,
    SourceError,
    V4L2CameraSource,
    WindowsCameraSource,
    camera_source,
)


@dataclass(frozen=True)
class SessionRecord:
    id: str
    model_id: str
    source_id: str
    status: str
    sample_every: int
    compute_device: ComputeChoice
    record_data: bool
    record_monitor: bool
    preview_overlays: OverlaySettings
    recording_overlays: OverlaySettings
    enable_profiling: bool
    provider: str | None
    actual_device: str | None
    provider_reason: str | None
    frame_count: int
    detection_count: int
    input_fps: float | None
    inference_fps: float | None
    average_latency_ms: float | None
    median_latency_ms: float | None
    p95_latency_ms: float | None
    peak_vram_mb: float | None
    dropped_frames: int
    operator_profile: dict[str, object] | None
    operator_profile_path: str | None
    monitor_video_path: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    completed_at: str | None


DetectorFactory = Callable[..., YoloOnnxDetector]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


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
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._cancel: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[None]] = {}

    def start(self, session_id: str) -> None:
        with self._lock:
            self._futures = {
                key: future for key, future in self._futures.items() if not future.done()
            }
            self._cancel = {
                key: event for key, event in self._cancel.items() if key in self._futures
            }
            existing = self._futures.get(session_id)
            if existing is not None and not existing.done():
                raise RuntimeError(f"Session {session_id} is already running.")
            if len(self._futures) >= self._max_workers:
                raise RuntimeError("Session worker capacity is currently full.")
            with self.database.connect() as connection:
                claimed = connection.execute(
                    """
                    UPDATE sessions SET status = 'running', error_message = NULL,
                        completed_at = NULL,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ? AND status = 'queued'
                    """,
                    (session_id,),
                ).rowcount
            if claimed != 1:
                raise ValueError(f"Session {session_id} is not queued.")
            event = threading.Event()
            self._cancel[session_id] = event
            try:
                self._futures[session_id] = self._executor.submit(
                    self.run, session_id, event
                )
            except Exception:
                with self.database.connect() as connection:
                    connection.execute(
                        "UPDATE sessions SET status = 'queued' WHERE id = ?",
                        (session_id,),
                    )
                self._cancel.pop(session_id, None)
                raise

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

    def shutdown(self, timeout: float = 5) -> None:
        with self._lock:
            events = tuple(self._cancel.values())
            futures = tuple(self._futures.values())
        for event in events:
            event.set()
        _, pending = wait(futures, timeout=timeout) if futures else (set(), set())
        self._executor.shutdown(wait=not pending, cancel_futures=True)

    def run(
        self, session_id: str, cancel_event: threading.Event | None = None
    ) -> None:
        canceled = cancel_event or threading.Event()
        started = time.perf_counter()
        processed = 0
        detection_count = 0
        latency_total = 0.0
        latencies: deque[float] = deque(maxlen=LATENCY_STATISTICS_WINDOW_FRAMES)
        median_latency_ms: float | None = None
        p95_latency_ms: float | None = None
        peak_vram_mb: float | None = None
        next_vram_sample_at = started
        detector: YoloOnnxDetector | None = None
        monitor_writer: MonitorVideoWriter | None = None
        try:
            record = _get_session(self.database, session_id)
            if record.status == "queued":
                with self.database.connect() as connection:
                    connection.execute(
                        """
                        UPDATE sessions SET status = 'running', error_message = NULL,
                            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE id = ? AND status = 'queued'
                        """,
                        (session_id,),
                    )
            elif record.status != "running":
                raise RuntimeError(
                    f"Session {session_id} cannot run from {record.status}."
                )
            model = self.model_registry.get(record.model_id)
            detector = self.detector_factory(
                model, record.compute_device, record.enable_profiling
            )
            with self.database.connect() as connection:
                connection.execute(
                    """
                    UPDATE sessions SET provider = ?, actual_device = ?,
                        provider_reason = ?,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                        error_message = NULL
                    WHERE id = ? AND status = 'running'
                    """,
                    (
                        detector.resolution.active_provider,
                        detector.resolution.actual_device,
                        detector.resolution.explanation,
                        session_id,
                    ),
                )
            source_record = self.media_store.get_record(record.source_id)
            source_path = Path(str(source_record["path"]))
            source_metadata = source_record["metadata"]
            assert isinstance(source_metadata, dict)
            if record.record_monitor:
                reported_fps = float(source_metadata.get("fps") or 30.0)
                recording_fps = max(reported_fps / record.sample_every, 0.1)
                monitor_writer = MonitorVideoWriter(
                    self.database.path.parent / "recordings" / f"{session_id}.mp4",
                    fps=recording_fps,
                )
            camera: V4L2CameraSource | WindowsCameraSource | None = None
            if source_record["kind"] == "video":
                if not source_path.exists():
                    raise SourceError(f"Source media {source_path.name} is unavailable.")
                source_frames = RecordedVideoSource(source_path).frames(record.sample_every)
            elif source_record["kind"] == "camera":
                camera = camera_source(str(source_record["path"]))
                source_frames = camera.frames(queue_size=1, stop_event=canceled)
            else:
                raise SourceError(f"Unsupported source kind {source_record['kind']}.")
            for source_frame in source_frames:
                if canceled.is_set():
                    self._persist_terminal_percentiles(session_id, latencies)
                    self._finalize_artifacts(session_id, detector, monitor_writer)
                    self._finish(session_id, "canceled")
                    return
                result = detector.predict(source_frame.frame)
                frame_id = str(uuid.uuid4())
                next_processed = processed + 1
                next_detection_count = detection_count + len(result.detections)
                next_latency_total = latency_total + result.latency_ms
                latencies.append(result.latency_ms)
                if next_processed == 1 or next_processed % 30 == 0:
                    window = list(latencies)
                    median_latency_ms = _percentile(window, 50)
                    p95_latency_ms = _percentile(window, 95)
                elapsed = max(time.perf_counter() - started, 1e-9)
                now = time.perf_counter()
                current_vram = None
                if (
                    detector.resolution.active_provider == "CUDAExecutionProvider"
                    and now >= next_vram_sample_at
                ):
                    current_vram = sample_process_vram_mb()
                    next_vram_sample_at = now + 2.0
                if current_vram is not None:
                    peak_vram_mb = max(peak_vram_mb or 0, current_vram)
                with self.database.connect() as connection:
                    if record.record_data:
                        connection.execute(
                            """
                            INSERT INTO frames (
                                id, session_id, frame_index, timestamp_ms, width, height,
                                latency_ms, inference_status, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'inferred',
                                strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
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
                    connection.execute(
                        """
                        UPDATE sessions SET frame_count = ?, detection_count = ?,
                            input_fps = ?, inference_fps = ?, average_latency_ms = ?,
                            median_latency_ms = ?, p95_latency_ms = ?, provider = ?,
                            actual_device = ?, provider_reason = ?, peak_vram_mb = ?,
                            dropped_frames = ?,
                            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE id = ? AND status = 'running'
                        """,
                        (
                            next_processed,
                            next_detection_count,
                            next_processed / elapsed,
                            1000 * next_processed / max(next_latency_total, 1e-9),
                            next_latency_total / next_processed,
                            median_latency_ms,
                            p95_latency_ms,
                            detector.resolution.active_provider,
                            detector.resolution.actual_device,
                            detector.resolution.explanation,
                            peak_vram_mb,
                            camera.dropped_frames if camera else 0,
                            session_id,
                        ),
                    )
                processed = next_processed
                detection_count = next_detection_count
                latency_total = next_latency_total
                if monitor_writer is not None:
                    monitor_writer.write(
                        render_monitor_frame(
                            source_frame.frame,
                            result.detections,
                            source_frame.timestamp_ms,
                            record.recording_overlays,
                        )
                    )
                self.previews.put(
                    session_id,
                    source_frame.frame_index,
                    source_frame.timestamp_ms,
                    source_frame.frame,
                    result.detections,
                    record.preview_overlays,
                )
            self._persist_terminal_percentiles(session_id, latencies)
            self._finalize_artifacts(session_id, detector, monitor_writer)
            terminal = "canceled" if canceled.is_set() else "completed"
            self._finish(session_id, terminal)
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            if detector is not None:
                try:
                    self._persist_terminal_percentiles(session_id, latencies)
                    self._finalize_artifacts(session_id, detector, monitor_writer)
                except Exception as finalization_error:
                    message += f" Artifact finalization also failed: {finalization_error}"
            elif monitor_writer is not None:
                monitor_writer.close()
            self._fail_safely(session_id, message)

    def _persist_terminal_percentiles(
        self, session_id: str, latencies: deque[float]
    ) -> None:
        window = list(latencies)
        if not window:
            return
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE sessions SET median_latency_ms = ?, p95_latency_ms = ?
                WHERE id = ?
                """,
                (_percentile(window, 50), _percentile(window, 95), session_id),
            )

    def _finalize_artifacts(
        self,
        session_id: str,
        detector: YoloOnnxDetector,
        monitor_writer: MonitorVideoWriter | None,
    ) -> None:
        if monitor_writer is not None:
            monitor_writer.close()
            if monitor_writer.frames_written and monitor_writer.path.is_file():
                with self.database.connect() as connection:
                    connection.execute(
                        "UPDATE sessions SET monitor_video_path = ? WHERE id = ?",
                        (str(monitor_writer.path), session_id),
                    )
        self._store_profile(session_id, detector)

    def _store_profile(self, session_id: str, detector: YoloOnnxDetector) -> None:
        profile = detector.finish_profile()
        if profile is None or detector.profile_path is None:
            return
        profile_dir = self.database.path.parent / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        target = profile_dir / f"{session_id}.json"
        shutil.move(str(detector.profile_path), target)
        value: dict[str, object] = {
            "provider_counts": profile.provider_counts,
            "cuda_operators": list(profile.cuda_operators),
            "cpu_operators": list(profile.cpu_operators),
            "cuda_executed": profile.cuda_executed,
        }
        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE sessions SET operator_profile_json = ?, operator_profile_path = ?
                WHERE id = ?
                """,
                (json.dumps(value, separators=(",", ":")), str(target), session_id),
            )

    def _finish(
        self,
        session_id: str,
        status: str,
    ) -> None:
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
    ) -> None:
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

    def _fail_safely(self, session_id: str, message: str) -> None:
        try:
            self._fail(session_id, message)
        except sqlite3.Error:
            self.database.release_emergency_reserve()
            with suppress(sqlite3.Error):
                self._fail(session_id, message)


class SessionService:
    def __init__(self, database: Database, runner: SessionRunner) -> None:
        self.database = database
        self.runner = runner

    def create(
        self,
        model_id: str,
        source_id: str,
        *,
        sample_every: int = 1,
        compute_device: ComputeChoice | str = ComputeChoice.AUTO,
        record_data: bool = True,
        record_monitor: bool = False,
        preview_overlays: OverlaySettings | None = None,
        recording_overlays: OverlaySettings | None = None,
        enable_profiling: bool = False,
    ) -> SessionRecord:
        if sample_every < 1:
            raise ValueError("sample_every must be at least one")
        self.runner.model_registry.get(model_id)
        self.runner.media_store.get_record(source_id)
        session_id = str(uuid.uuid4())
        compute = ComputeChoice(compute_device)
        preview = preview_overlays or OverlaySettings()
        recording = recording_overlays or OverlaySettings()
        for name, overlays in (("preview", preview), ("recording", recording)):
            unsupported = [
                overlay
                for overlay in ("trails", "masks")
                if getattr(overlays, overlay)
            ]
            if unsupported:
                raise ValueError(
                    f"{name} overlays are not supported by the active pipeline: "
                    + ", ".join(unsupported)
                )
        settings = json.dumps(
            {
                "sample_every": sample_every,
                "record_data": record_data,
                "record_monitor": record_monitor,
                "preview_overlays": asdict(preview),
                "recording_overlays": asdict(recording),
                "enable_profiling": enable_profiling,
            },
            separators=(",", ":"),
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    id, model_id, source_id, status, settings_json, requested_compute,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (session_id, model_id, source_id, settings, compute.value),
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

    def shutdown(self, timeout: float = 5) -> None:
        self.runner.shutdown(timeout)

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
                requested_compute, actual_device, provider_reason, frame_count,
                detection_count, input_fps, inference_fps, average_latency_ms,
                median_latency_ms, p95_latency_ms, peak_vram_mb, dropped_frames,
                operator_profile_json, operator_profile_path, monitor_video_path,
                error_message, created_at, updated_at, completed_at
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
        compute_device=ComputeChoice(row[6]),
        record_data=bool(settings.get("record_data", True)),
        record_monitor=bool(settings.get("record_monitor", False)),
        preview_overlays=OverlaySettings(**settings.get("preview_overlays", {})),
        recording_overlays=OverlaySettings(**settings.get("recording_overlays", {})),
        enable_profiling=bool(settings.get("enable_profiling", False)),
        provider=row[5],
        actual_device=row[7],
        provider_reason=row[8],
        frame_count=row[9],
        detection_count=row[10],
        input_fps=row[11],
        inference_fps=row[12],
        average_latency_ms=row[13],
        median_latency_ms=row[14],
        p95_latency_ms=row[15],
        peak_vram_mb=row[16],
        dropped_frames=row[17],
        operator_profile=json.loads(row[18]) if row[18] else None,
        operator_profile_path=row[19],
        monitor_video_path=row[20],
        error_message=row[21],
        created_at=row[22],
        updated_at=row[23],
        completed_at=row[24],
    )
