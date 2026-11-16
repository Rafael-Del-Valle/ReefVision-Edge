from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reefvision_edge.domain import Detection, OverlaySettings
from reefvision_edge.monitor_video import (
    MonitorVideoWriter,
    overlay_capabilities,
    render_monitor_frame,
)


def test_preview_and_recording_overlay_settings_render_independently() -> None:
    frame = np.zeros((96, 160, 3), dtype=np.uint8)
    detections = (Detection(0, "coral", 0.9, (30.0, 40.0, 90.0, 80.0)),)

    preview = render_monitor_frame(
        frame,
        detections,
        1250.0,
        OverlaySettings(timer=False, boxes=True, labels=False),
    )
    recording = render_monitor_frame(
        frame,
        detections,
        1250.0,
        OverlaySettings(timer=True, boxes=False, labels=True),
    )

    assert tuple(preview[40, 30]) != (0, 0, 0)
    assert tuple(recording[40, 30]) == (0, 0, 0)
    assert np.any(recording[2:24, 2:100] != 0)
    assert not np.any(preview[2:24, 2:25] != 0)


def test_overlay_capabilities_explain_unsupported_masks_and_trails() -> None:
    capabilities = overlay_capabilities()

    assert capabilities["timer"]["supported"] is True
    assert capabilities["boxes"]["supported"] is True
    assert capabilities["labels"]["supported"] is True
    assert capabilities["masks"] == {
        "supported": False,
        "reason": "The active detector does not output segmentation masks.",
    }
    assert capabilities["trails"] == {
        "supported": False,
        "reason": "Tracking IDs are not available for the active detector.",
    }


def test_monitor_video_writer_creates_playable_mp4(tmp_path: Path) -> None:
    path = tmp_path / "monitor.mp4"
    writer = MonitorVideoWriter(path, fps=12.0)
    for index in range(6):
        frame = np.full((64, 96, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.close()

    capture = cv2.VideoCapture(str(path))
    try:
        decoded = 0
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    assert decoded == 6


def test_monitor_video_writer_reports_encoder_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ClosedWriter:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        "reefvision_edge.monitor_video.cv2.VideoWriter", lambda *_a, **_k: ClosedWriter()
    )
    writer = MonitorVideoWriter(tmp_path / "failed.mp4", fps=10)

    with pytest.raises(RuntimeError, match="software MP4 encoder"):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
