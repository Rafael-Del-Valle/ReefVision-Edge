from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.fixtures.build_video import build_mjpeg_video
from tests.test_model_registry import valid_metadata

from reefvision_edge.db import Database
from reefvision_edge.inference import YoloOnnxDetector
from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_registry import ModelRegistry
from reefvision_edge.preview import PreviewStore
from reefvision_edge.sessions import SessionRunner, SessionService


def prepared_services(tmp_path: Path, *, frame_count: int = 5):
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    model_registry = ModelRegistry(database, tmp_path / "models")
    model = model_registry.register(
        io.BytesIO(model_path.read_bytes()), "tiny.onnx", valid_metadata()
    )
    video_path = build_mjpeg_video(tmp_path / "source.avi", frame_count=frame_count)
    media_store = MediaStore(database, tmp_path / "media")
    source = media_store.import_video(io.BytesIO(video_path.read_bytes()), "source.avi")
    previews = PreviewStore(tmp_path / "previews")
    runner = SessionRunner(database, model_registry, media_store, previews)
    service = SessionService(database, runner)
    return database, model, source, previews, runner, service


def test_recorded_session_persists_metrics_detections_and_preview(tmp_path: Path) -> None:
    database, model, source, previews, runner, service = prepared_services(tmp_path)
    session = service.create(model.id, source.id, sample_every=2)

    runner.run(session.id)

    completed = service.get(session.id)
    assert completed.status == "completed"
    assert completed.frame_count == 3
    assert completed.detection_count == 6
    assert completed.provider == model.provider
    assert completed.inference_fps is not None and completed.inference_fps > 0
    assert completed.average_latency_ms is not None and completed.average_latency_ms >= 0
    with database.connect() as connection:
        frames = connection.execute(
            """
            SELECT frame_index, timestamp_ms
            FROM frames WHERE session_id = ? ORDER BY frame_index
            """,
            (session.id,),
        ).fetchall()
        first_box = connection.execute(
            """
            SELECT d.x_min, d.y_min, d.x_max, d.y_max
            FROM detections d JOIN frames f ON d.frame_id = f.id
            WHERE f.session_id = ? AND f.frame_index = 0 AND d.class_id = 0
            """,
            (session.id,),
        ).fetchone()
    assert frames == [(0, 0.0), (2, 200.0), (4, 400.0)]
    assert first_box == pytest.approx((60.0, 35.0, 100.0, 55.0))
    preview = previews.get(session.id)
    assert preview.frame_index == 4
    assert preview.detection_count == 2
    decoded = cv2.imdecode(np.frombuffer(preview.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (90, 160, 3)

    restored = PreviewStore(tmp_path / "previews").get(session.id)
    assert restored.frame_index == 4
    assert restored.detection_count == 2
    assert restored.jpeg == preview.jpeg


def test_running_session_can_be_canceled_with_committed_progress(tmp_path: Path) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path, frame_count=40)

    class SlowDetector(YoloOnnxDetector):
        def predict(self, frame):  # type: ignore[no-untyped-def]
            time.sleep(0.01)
            return super().predict(frame)

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=SlowDetector,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id, sample_every=1)

    service.start(session.id)
    deadline = time.monotonic() + 3
    while service.get(session.id).status != "running" and time.monotonic() < deadline:
        time.sleep(0.005)
    service.stop(session.id)
    runner.wait(session.id, timeout=3)

    canceled = service.get(session.id)
    assert canceled.status == "canceled"
    assert 0 <= canceled.frame_count < 40
    assert canceled.completed_at is not None


def test_preview_overlay_draws_box_at_original_coordinates() -> None:
    from reefvision_edge.domain import Detection
    from reefvision_edge.preview import draw_overlay

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    rendered = draw_overlay(
        frame,
        (Detection(0, "coral", 0.9, (20.0, 30.0, 80.0, 70.0)),),
    )

    assert tuple(rendered[30, 20]) != (0, 0, 0)
    assert tuple(frame[30, 20]) == (0, 0, 0)


def test_runner_bounds_concurrent_work_and_shutdown_cancels_active_job(
    tmp_path: Path,
) -> None:
    database, model, source, previews, _, _ = prepared_services(
        tmp_path, frame_count=40
    )

    class SlowDetector(YoloOnnxDetector):
        def predict(self, frame):  # type: ignore[no-untyped-def]
            time.sleep(0.02)
            return super().predict(frame)

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=SlowDetector,
    )
    service = SessionService(database, runner)
    first = service.create(model.id, source.id)
    second = service.create(model.id, source.id)

    service.start(first.id)
    with pytest.raises(RuntimeError, match="capacity"):
        service.start(second.id)
    assert service.get(first.id).status == "running"
    assert service.get(second.id).status == "queued"

    service.shutdown(timeout=3)

    assert service.get(first.id).status == "canceled"


def test_unexpected_detector_error_finishes_failed(tmp_path: Path) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path)

    class BrokenDetector(YoloOnnxDetector):
        def predict(self, frame):  # type: ignore[no-untyped-def]
            raise ValueError("unexpected detector failure")

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=BrokenDetector,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id)

    runner.run(session.id)

    failed = service.get(session.id)
    assert failed.status == "failed"
    assert failed.error_message == "unexpected detector failure"
