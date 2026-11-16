from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.fixtures.build_video import build_mjpeg_video

from reefvision_edge.db import Database
from reefvision_edge.media_store import MediaStore
from reefvision_edge.monitor import MonitorPreviewService
from reefvision_edge.sources import SourceError


def _store(tmp_path: Path) -> tuple[MediaStore, str]:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    store = MediaStore(database, tmp_path / "media")
    video = build_mjpeg_video(tmp_path / "source.avi", frame_count=4, fps=20)
    source = store.import_video(io.BytesIO(video.read_bytes()), "source.avi")
    return store, source.id


def test_recorded_monitor_preview_is_paced_loops_and_changes_frames(tmp_path: Path) -> None:
    store, source_id = _store(tmp_path)
    monitor = MonitorPreviewService(store)
    try:
        monitor.select(source_id)
        deadline = time.monotonic() + 2
        while monitor.status().frame_index < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        first = monitor.jpeg()
        first_index = monitor.status().frame_index
        while monitor.status().frame_index <= first_index and time.monotonic() < deadline:
            time.sleep(0.01)
        second = monitor.jpeg()

        assert monitor.status().state == "replaying"
        assert monitor.status().source_id == source_id
        assert first != second
        assert cv2.imdecode(np.frombuffer(second, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (
            90,
            160,
            3,
        )
    finally:
        monitor.shutdown()


def test_monitor_reports_missing_selection_and_releases_for_inference(tmp_path: Path) -> None:
    store, source_id = _store(tmp_path)
    monitor = MonitorPreviewService(store)
    with pytest.raises(SourceError, match="not found"):
        monitor.select("missing")

    monitor.select(source_id)
    monitor.release_for_inference(source_id)

    assert monitor.status().state == "idle"
    assert monitor.status().source_id is None
    with pytest.raises(KeyError):
        monitor.jpeg()
    monitor.shutdown()
