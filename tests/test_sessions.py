from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.fixtures.build_video import build_mjpeg_video
from tests.test_model_registry import valid_metadata

from reefvision_edge.compute import ComputeChoice, cpu_fallback
from reefvision_edge.db import Database
from reefvision_edge.domain import OverlaySettings
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
    assert completed.input_fps is not None and completed.input_fps > 0
    assert completed.average_latency_ms is not None and completed.average_latency_ms >= 0
    assert completed.median_latency_ms is not None and completed.median_latency_ms >= 0
    assert completed.p95_latency_ms is not None
    assert completed.p95_latency_ms >= completed.median_latency_ms
    assert completed.compute_device is ComputeChoice.AUTO
    assert completed.actual_device == "CPU"
    assert completed.provider_reason is not None
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


def test_terminal_latency_percentiles_include_every_processed_frame_within_window(
    tmp_path: Path,
) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path)
    expected_latencies = iter((1.0, 2.0, 3.0, 4.0, 100.0))

    class ControlledLatencyDetector(YoloOnnxDetector):
        def predict(self, frame):  # type: ignore[no-untyped-def]
            return replace(
                super().predict(frame), latency_ms=next(expected_latencies)
            )

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=ControlledLatencyDetector,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id)

    runner.run(session.id)

    completed = service.get(session.id)
    assert completed.median_latency_ms == pytest.approx(3.0)
    assert completed.p95_latency_ms == pytest.approx(80.8)


def test_session_settings_keep_recording_and_overlay_controls_independent(
    tmp_path: Path,
) -> None:
    database, model, source, _, runner, service = prepared_services(tmp_path)
    preview = OverlaySettings(timer=False, boxes=True, labels=False)
    recording = OverlaySettings(timer=True, boxes=False, labels=True)
    session = service.create(
        model.id,
        source.id,
        compute_device=ComputeChoice.CPU,
        record_data=False,
        record_monitor=True,
        preview_overlays=preview,
        recording_overlays=recording,
    )

    runner.run(session.id)

    completed = service.get(session.id)
    assert completed.compute_device is ComputeChoice.CPU
    assert completed.record_data is False
    assert completed.record_monitor is True
    assert completed.preview_overlays == preview
    assert completed.recording_overlays == recording
    assert completed.frame_count == 5
    assert completed.detection_count == 10
    with database.connect() as connection:
        frame_rows = connection.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id = ?", (session.id,)
        ).fetchone()[0]
        detection_rows = connection.execute(
            """
            SELECT COUNT(*) FROM detections d
            JOIN frames f ON f.id = d.frame_id WHERE f.session_id = ?
            """,
            (session.id,),
        ).fetchone()[0]
    assert frame_rows == 0
    assert detection_rows == 0


def test_session_preserves_overlay_preferences_the_active_pipeline_cannot_render(
    tmp_path: Path,
) -> None:
    _, model, source, _, runner, service = prepared_services(tmp_path)
    preview = OverlaySettings(trails=True)
    recording = OverlaySettings(masks=True)

    session = service.create(
        model.id,
        source.id,
        preview_overlays=preview,
        recording_overlays=recording,
    )
    runner.run(session.id)

    completed = service.get(session.id)
    assert completed.status == "completed"
    assert completed.preview_overlays == preview
    assert completed.recording_overlays == recording


@pytest.mark.parametrize(
    ("record_data", "record_monitor", "expect_rows", "expect_video"),
    [
        (True, False, True, False),
        (False, True, False, True),
        (True, True, True, True),
    ],
)
def test_data_and_monitor_recording_modes_are_independent(
    tmp_path: Path,
    record_data: bool,
    record_monitor: bool,
    expect_rows: bool,
    expect_video: bool,
) -> None:
    database, model, source, _, runner, service = prepared_services(tmp_path)
    session = service.create(
        model.id,
        source.id,
        record_data=record_data,
        record_monitor=record_monitor,
    )

    runner.run(session.id)

    completed = service.get(session.id)
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM frames WHERE session_id = ?", (session.id,)
        ).fetchone()[0]
    assert (rows > 0) is expect_rows
    assert (completed.monitor_video_path is not None) is expect_video
    if completed.monitor_video_path:
        capture = cv2.VideoCapture(completed.monitor_video_path)
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == completed.frame_count
        finally:
            capture.release()


def test_cancel_finalizes_partial_monitor_video(tmp_path: Path) -> None:
    database, model, source, previews, _, _ = prepared_services(
        tmp_path, frame_count=80
    )

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
    session = service.create(model.id, source.id, record_monitor=True)
    service.start(session.id)
    deadline = time.monotonic() + 3
    while service.get(session.id).frame_count < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    service.stop(session.id)
    runner.wait(session.id, timeout=3)

    canceled = service.get(session.id)
    assert canceled.status == "canceled"
    assert canceled.monitor_video_path is not None
    capture = cv2.VideoCapture(canceled.monitor_video_path)
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == canceled.frame_count
    finally:
        capture.release()


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


def test_preview_store_scales_image_and_overlay_together_for_browser_delivery(
    tmp_path: Path,
) -> None:
    from reefvision_edge.domain import Detection

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    store = PreviewStore(tmp_path / "previews")

    preview = store.put(
        "wide-session",
        1,
        10.0,
        frame,
        (Detection(0, "coral", 0.9, (300.0, 300.0, 600.0, 600.0)),),
        OverlaySettings(timer=False, boxes=True, labels=False),
    )
    decoded = cv2.imdecode(np.frombuffer(preview.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded.shape == (720, 1280, 3)
    assert np.any(decoded[198:203, 198:203] != 0)
    assert not np.any(frame)


def test_latest_preview_publisher_limits_rate_and_flushes_newest_frame(
    tmp_path: Path,
) -> None:
    from reefvision_edge.preview import LatestPreviewPublisher

    clock = [100.0]
    store = PreviewStore(tmp_path / "previews")
    publisher = LatestPreviewPublisher(
        store,
        "camera-session",
        OverlaySettings(timer=False, boxes=False, labels=False),
        max_fps=30.0,
        clock=lambda: clock[0],
    )

    assert publisher.offer(0, 0.0, np.full((12, 18, 3), 10, dtype=np.uint8), ())
    clock[0] += 0.010
    assert not publisher.offer(
        1, 10.0, np.full((12, 18, 3), 20, dtype=np.uint8), ()
    )
    clock[0] += 0.010
    assert not publisher.offer(
        2, 20.0, np.full((12, 18, 3), 30, dtype=np.uint8), ()
    )
    assert store.get("camera-session").frame_index == 0

    assert publisher.flush()
    latest = store.get("camera-session")
    restored = PreviewStore(tmp_path / "previews").get("camera-session")
    assert latest.frame_index == 2
    assert restored.frame_index == 2
    assert latest.published_at > 0
    assert not publisher.flush()


def test_latest_preview_publisher_does_not_halve_a_30_fps_camera() -> None:
    from reefvision_edge.preview import LatestPreviewPublisher

    class RecordingStore:
        def __init__(self) -> None:
            self.frame_indexes: list[int] = []

        def put(self, _session_id, frame_index, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.frame_indexes.append(frame_index)

    clock = [100.0]
    store = RecordingStore()
    publisher = LatestPreviewPublisher(
        store,  # type: ignore[arg-type]
        "camera-session",
        OverlaySettings(timer=False, boxes=False, labels=False),
        max_fps=30.0,
        clock=lambda: clock[0],
    )
    frame = np.zeros((12, 18, 3), dtype=np.uint8)

    for frame_index in range(30):
        publisher.offer(frame_index, frame_index * 33.0, frame, ())
        clock[0] += 0.033

    assert store.frame_indexes == list(range(30))


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


def test_detector_failure_finalizes_partial_monitor_video(tmp_path: Path) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path)

    class BrokenAfterTwoDetector(YoloOnnxDetector):
        calls = 0

        def predict(self, frame):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls > 2:
                raise ValueError("failure after two frames")
            return super().predict(frame)

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=BrokenAfterTwoDetector,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id, record_monitor=True)

    runner.run(session.id)

    failed = service.get(session.id)
    assert failed.status == "failed"
    assert failed.monitor_video_path is not None
    capture = cv2.VideoCapture(failed.monitor_video_path)
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    finally:
        capture.release()


def test_vram_sampling_is_rate_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path, frame_count=40)
    samples = 0

    def sample() -> float:
        nonlocal samples
        samples += 1
        return 256.0

    monkeypatch.setattr("reefvision_edge.sessions.sample_process_vram_mb", sample)

    class CudaDetector(YoloOnnxDetector):
        @property
        def resolution(self):  # type: ignore[no-untyped-def]
            return replace(super().resolution, active_provider="CUDAExecutionProvider")

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=CudaDetector,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id, compute_device=ComputeChoice.CPU)

    runner.run(session.id)

    assert service.get(session.id).frame_count == 40
    assert samples <= 2


def test_realized_auto_fallback_device_and_reason_are_persisted(tmp_path: Path) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path)

    class AutoFallbackDetector(YoloOnnxDetector):
        def predict(self, frame):  # type: ignore[no-untyped-def]
            result = super().predict(frame)
            self.compute = cpu_fallback(
                ComputeChoice.AUTO, "CUDA executed zero supported operators."
            )
            return result

    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
        detector_factory=AutoFallbackDetector,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id, compute_device=ComputeChoice.AUTO)

    runner.run(session.id)

    completed = service.get(session.id)
    assert completed.provider == "CPUExecutionProvider"
    assert completed.actual_device == "CPU"
    assert completed.provider_reason is not None
    assert "using CPU" in completed.provider_reason
