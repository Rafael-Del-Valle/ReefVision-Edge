from __future__ import annotations

import hashlib
import io
import platform
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.fixtures.build_video import build_mjpeg_video

from reefvision_edge.db import Database
from reefvision_edge.media_store import MediaStore
from reefvision_edge.sources import (
    NewestFrameQueue,
    RecordedVideoSource,
    SourceError,
    V4L2CameraSource,
    VideoProbe,
    WindowsCameraSource,
    configure_windows_capture,
    probe_video,
)


def test_windows_capture_negotiates_a_verified_sixteen_by_nine_mode() -> None:
    class NegotiatingCapture:
        def __init__(self) -> None:
            self.width = 640
            self.height = 480
            self.requested_widths: list[int] = []

        def set(self, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = int(value)
                self.requested_widths.append(self.width)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = int(value)
            return True

        def read(self):  # type: ignore[no-untyped-def]
            if (self.width, self.height) == (1920, 1080):
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    capture = NegotiatingCapture()

    first_frame = configure_windows_capture(capture)

    assert capture.requested_widths == [1920, 1280]
    assert first_frame is not None
    assert first_frame.shape == (720, 1280, 3)


def test_windows_capture_keeps_actual_four_by_three_frame_when_modes_are_rejected() -> None:
    class FourByThreeCapture:
        def set(self, _prop: int, _value: float) -> bool:
            return False

        def read(self):  # type: ignore[no-untyped-def]
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

    first_frame = configure_windows_capture(FourByThreeCapture())

    assert first_frame is not None
    assert first_frame.shape == (480, 640, 3)


def test_windows_capture_negotiation_failure_releases_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCapture:
        released = False

        def isOpened(self) -> bool:
            return True

        def set(self, _prop: int, _value: float) -> bool:
            raise RuntimeError("driver rejected property")

        def release(self) -> None:
            self.released = True

    capture = FailingCapture()
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        "reefvision_edge.sources.cv2.VideoCapture", lambda *_args: capture
    )
    frames = WindowsCameraSource(0, backend="dshow").frames()

    with pytest.raises(SourceError, match="could not negotiate a capture mode"):
        next(frames)
    assert capture.released is True


def media_store(tmp_path: Path) -> MediaStore:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    return MediaStore(database, tmp_path / "media")


def test_import_copies_video_without_changing_supplied_bytes(tmp_path: Path) -> None:
    original = build_mjpeg_video(tmp_path / "source.avi").read_bytes()
    stream = io.BytesIO(original)

    source = media_store(tmp_path).import_video(stream, "DJI_0001.avi")

    assert stream.getvalue() == original
    assert source.path.read_bytes() == original
    assert source.sha256 == hashlib.sha256(original).hexdigest()
    assert source.path.parent == tmp_path / "media" / source.id
    assert source.probe.width == 160
    assert source.probe.height == 90
    assert source.probe.frame_count == 5
    assert source.probe.fps == pytest.approx(10.0)
    assert source.probe.codec.strip() == "MJPG"


def test_video_upload_limit_removes_partial_copy(tmp_path: Path) -> None:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    store = MediaStore(database, tmp_path / "media", max_video_bytes=4)

    with pytest.raises(SourceError, match="upload limit"):
        store.import_video(io.BytesIO(b"12345"), "large.avi")

    assert list((tmp_path / "media").glob("*")) == []


def test_recorded_sampling_starts_at_zero_and_is_deterministic(tmp_path: Path) -> None:
    path = build_mjpeg_video(tmp_path / "source.avi")
    source = RecordedVideoSource(path)

    first = list(source.frames(sample_every=2))
    second = list(source.frames(sample_every=2))

    assert [item.frame_index for item in first] == [0, 2, 4]
    assert [item.timestamp_ms for item in first] == pytest.approx([0.0, 200.0, 400.0])
    assert [item.frame_index for item in second] == [0, 2, 4]
    assert all(item.frame.shape == (90, 160, 3) for item in first)


def test_probe_rejects_unreadable_video(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(SourceError, match="decode"):
        probe_video(broken)


def test_camera_reports_host_and_device_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if platform.system() != "Linux":
        with pytest.raises(SourceError, match="Linux"):
            V4L2CameraSource(Path("/dev/video0"))

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(SourceError, match="does not exist"):
        V4L2CameraSource(tmp_path / "missing-video-device")


def test_newest_frame_queue_discards_oldest_when_full() -> None:
    queue = NewestFrameQueue(max_size=1)
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)

    queue.put(first)
    queue.put(second)

    assert queue.dropped == 1
    assert np.array_equal(queue.get(timeout=0.01), second)


def test_recorded_source_reports_early_decoder_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenCapture:
        def __init__(self) -> None:
            self.reads = 0

        def isOpened(self) -> bool:
            return True

        def read(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            if self.reads == 1:
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            return None

    source = object.__new__(RecordedVideoSource)
    source.path = tmp_path / "truncated.avi"
    source.probe = VideoProbe(2, 2, 10.0, 3, "MJPG", "avi")
    monkeypatch.setattr("reefvision_edge.sources.cv2.VideoCapture", lambda _: BrokenCapture())

    with pytest.raises(SourceError, match="ended early"):
        list(source.frames())


def test_windows_camera_disconnect_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DisconnectingCapture:
        released = False
        reads = 0

        def isOpened(self) -> bool:
            return True

        def read(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            if self.reads == 1:
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            self.released = True

    capture = DisconnectingCapture()
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        "reefvision_edge.sources.cv2.VideoCapture", lambda *_args: capture
    )
    source = WindowsCameraSource(0, backend="msmf")
    frames = source.frames()

    assert next(frames).frame_index == 0
    with pytest.raises(SourceError, match="stopped delivering"):
        next(frames)
    assert capture.released is True


def test_windows_camera_frames_keep_capture_index_and_elapsed_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimedCapture:
        reads = 0

        def isOpened(self) -> bool:
            return True

        def read(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            if self.reads <= 2:
                return True, np.full((2, 2, 3), self.reads, dtype=np.uint8)
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        "reefvision_edge.sources.cv2.VideoCapture", lambda *_args: TimedCapture()
    )
    frames = WindowsCameraSource(0, backend="dshow").frames(queue_size=2)

    first = next(frames)
    second = next(frames)

    assert first.frame_index == 0
    assert first.timestamp_ms is not None and first.timestamp_ms >= 0
    assert second.frame_index == 1
    assert second.timestamp_ms is not None
    assert second.timestamp_ms > first.timestamp_ms
