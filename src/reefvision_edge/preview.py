from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

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
    def __init__(self, storage_dir: Path | None = None, *, max_cached: int = 32) -> None:
        if max_cached < 1:
            raise ValueError("max_cached must be at least one")
        self.storage_dir = storage_dir
        self.max_cached = max_cached
        self._lock = threading.Lock()
        self._previews: OrderedDict[str, PreviewFrame] = OrderedDict()

    def _path(self, session_id: str) -> Path | None:
        if self.storage_dir is None:
            return None
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.storage_dir / f"{digest}.preview"

    def _remember(self, session_id: str, preview: PreviewFrame) -> None:
        self._previews[session_id] = preview
        self._previews.move_to_end(session_id)
        while len(self._previews) > self.max_cached:
            self._previews.popitem(last=False)

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
        target = self._path(session_id)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = json.dumps(
                {
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "detection_count": len(detections),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                stream.write(metadata + b"\n" + preview.jpeg)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        with self._lock:
            self._remember(session_id, preview)
        return preview

    def get(self, session_id: str) -> PreviewFrame:
        with self._lock:
            preview = self._previews.get(session_id)
            if preview is not None:
                self._previews.move_to_end(session_id)
                return preview
        target = self._path(session_id)
        if target is None or not target.is_file():
            raise KeyError(session_id)
        try:
            metadata_bytes, jpeg = target.read_bytes().split(b"\n", 1)
            metadata = json.loads(metadata_bytes)
            if metadata["session_id"] != session_id or not jpeg:
                raise ValueError("preview metadata mismatch")
            preview = PreviewFrame(
                frame_index=int(metadata["frame_index"]),
                timestamp_ms=(
                    None
                    if metadata["timestamp_ms"] is None
                    else float(metadata["timestamp_ms"])
                ),
                detection_count=int(metadata["detection_count"]),
                jpeg=jpeg,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise KeyError(session_id) from exc
        with self._lock:
            self._remember(session_id, preview)
        return preview
