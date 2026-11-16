from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from reefvision_edge.domain import Detection, OverlaySettings

OVERLAY_COLOR = (201, 219, 114)
OVERLAY_TEXT_COLOR = (6, 23, 28)


def overlay_capabilities() -> dict[str, dict[str, object]]:
    return {
        "timer": {"supported": True, "reason": None},
        "trails": {
            "supported": False,
            "reason": "Tracking IDs are not available for the active detector.",
        },
        "boxes": {"supported": True, "reason": None},
        "masks": {
            "supported": False,
            "reason": "The active detector does not output segmentation masks.",
        },
        "labels": {"supported": True, "reason": None},
    }


def _timer_text(timestamp_ms: float | None) -> str:
    milliseconds = max(0, round(timestamp_ms or 0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def render_monitor_frame(
    frame: NDArray[np.uint8],
    detections: tuple[Detection, ...],
    timestamp_ms: float | None,
    settings: OverlaySettings,
) -> NDArray[np.uint8]:
    rendered = frame.copy()
    if settings.timer:
        value = _timer_text(timestamp_ms)
        (width, height), _ = cv2.getTextSize(
            value, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(rendered, (2, 2), (width + 10, height + 10), (6, 23, 28), -1)
        cv2.putText(
            rendered,
            value,
            (6, height + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            OVERLAY_COLOR,
            1,
            cv2.LINE_AA,
        )
    for detection in detections:
        x_min, y_min, x_max, y_max = (round(value) for value in detection.box)
        if settings.boxes:
            cv2.rectangle(rendered, (x_min, y_min), (x_max, y_max), OVERLAY_COLOR, 2)
        if settings.labels:
            label = f"{detection.class_name} {detection.score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            label_bottom = max(text_height + 4, y_min - 2)
            label_top = max(0, label_bottom - text_height - 8)
            cv2.rectangle(
                rendered,
                (x_min, label_top),
                (x_min + text_width + 8, label_bottom),
                OVERLAY_COLOR,
                -1,
            )
            cv2.putText(
                rendered,
                label,
                (x_min + 4, label_bottom - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                OVERLAY_TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
    return rendered


class MonitorVideoWriter:
    def __init__(self, path: Path, *, fps: float) -> None:
        if fps <= 0:
            raise ValueError("Monitor recording FPS must be positive.")
        self.path = path
        self.fps = fps
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self._closed = False
        self._timeline_origin_ms: float | None = None
        self._next_output_ms = 0.0

    def write(
        self, frame: NDArray[np.uint8], *, timestamp_ms: float | None = None
    ) -> bool:
        if self._closed:
            raise RuntimeError("Monitor video writer is already closed.")
        if timestamp_ms is not None:
            if self._timeline_origin_ms is None:
                self._timeline_origin_ms = timestamp_ms
            relative_ms = max(0.0, timestamp_ms - self._timeline_origin_ms)
            if relative_ms + 1e-6 < self._next_output_ms:
                return False
            frame_interval_ms = 1000.0 / self.fps
            while self._next_output_ms <= relative_ms + 1e-6:
                self._next_output_ms += frame_interval_ms
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            self._writer = cv2.VideoWriter(
                str(self.path), fourcc, self.fps, size
            )
            self._size = size
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                self.path.unlink(missing_ok=True)
                raise RuntimeError(
                    "The software MP4 encoder could not open the monitor recording."
                )
        if size != self._size:
            raise RuntimeError("Monitor recording frame dimensions changed during the run.")
        self._writer.write(frame)
        self.frames_written += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None
