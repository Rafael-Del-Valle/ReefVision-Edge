from __future__ import annotations

import hashlib
import io
import platform
from pathlib import Path

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
    probe_video,
)


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
