from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from reefvision_edge.domain import Detection


@dataclass(frozen=True)
class PreviewFrame:
    frame_index: int
    timestamp_ms: float | None
    detection_count: int
    jpeg: bytes


def draw_overlay(
    frame: NDArray[np.uint8], detections: tuple[Detection, ...]
) -> NDArray[np.uint8]:
    rendered = frame.copy()
    color = (201, 219, 114)
    for detection in detections:
        x_min, y_min, x_max, y_max = (round(value) for value in detection.box)
        cv2.rectangle(rendered, (x_min, y_min), (x_max, y_max), color, 2)
        label = f"{detection.class_name} {detection.score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_top = max(0, y_min - text_height - 8)
        cv2.rectangle(
            rendered,
            (x_min, label_top),
            (x_min + text_width + 8, y_min),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            label,
            (x_min + 4, max(text_height + 1, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (6, 23, 28),
            1,
            cv2.LINE_AA,
        )
    return rendered


class PreviewStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._previews: dict[str, PreviewFrame] = {}

    def put(
        self,
        session_id: str,
        frame_index: int,
        timestamp_ms: float | None,
        frame: NDArray[np.uint8],
        detections: tuple[Detection, ...],
    ) -> PreviewFrame:
        rendered = draw_overlay(frame, detections)
        encoded, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not encoded:
            raise RuntimeError("Preview JPEG encoding failed.")
        preview = PreviewFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detection_count=len(detections),
            jpeg=buffer.tobytes(),
        )
        with self._lock:
            self._previews[session_id] = preview
        return preview

    def get(self, session_id: str) -> PreviewFrame:
        with self._lock:
            preview = self._previews.get(session_id)
        if preview is None:
            raise KeyError(session_id)
        return preview
