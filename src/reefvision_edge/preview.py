from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from reefvision_edge.domain import Detection, OverlaySettings
from reefvision_edge.monitor_video import render_monitor_frame

PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720


@dataclass(frozen=True)
class PreviewFrame:
    frame_index: int
    timestamp_ms: float | None
    detection_count: int
    published_at: float
    jpeg: bytes


@dataclass(frozen=True)
class _PendingPreview:
    frame_index: int
    timestamp_ms: float | None
    frame: NDArray[np.uint8]
    detections: tuple[Detection, ...]


def draw_overlay(
    frame: NDArray[np.uint8], detections: tuple[Detection, ...]
) -> NDArray[np.uint8]:
    return render_monitor_frame(
        frame,
        detections,
        None,
        OverlaySettings(timer=False, boxes=True, labels=True),
    )


def fit_browser_preview(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Bound browser decode work without changing the source aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return frame
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(
        cv2.resize(frame, size, interpolation=cv2.INTER_AREA), dtype=np.uint8
    )


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
        settings: OverlaySettings | None = None,
    ) -> PreviewFrame:
        rendered = render_monitor_frame(
            frame, detections, timestamp_ms, settings or OverlaySettings()
        )
        rendered = fit_browser_preview(rendered)
        encoded, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not encoded:
            raise RuntimeError("Preview JPEG encoding failed.")
        preview = PreviewFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detection_count=len(detections),
            published_at=time.time(),
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
                    "published_at": preview.published_at,
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
                published_at=float(metadata.get("published_at", 0.0)),
                jpeg=jpeg,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise KeyError(session_id) from exc
        with self._lock:
            self._remember(session_id, preview)
        return preview


class LatestPreviewPublisher:
    def __init__(
        self,
        store: PreviewStore,
        session_id: str,
        settings: OverlaySettings,
        *,
        max_fps: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_fps <= 0:
            raise ValueError("Preview publication FPS must be positive.")
        self.store = store
        self.session_id = session_id
        self.settings = settings
        self.interval = 1.0 / max_fps
        self.clock = clock
        self._next_publish_at = 0.0
        self._pending: _PendingPreview | None = None
        self._published_frame_index: int | None = None

    def offer(
        self,
        frame_index: int,
        timestamp_ms: float | None,
        frame: NDArray[np.uint8],
        detections: tuple[Detection, ...],
    ) -> bool:
        self._pending = _PendingPreview(
            frame_index,
            timestamp_ms,
            frame.copy(),
            detections,
        )
        now = self.clock()
        if self._published_frame_index is None or now >= self._next_publish_at:
            return self._publish(now)
        return False

    def flush(self) -> bool:
        return self._publish(self.clock())

    def _publish(self, now: float) -> bool:
        pending = self._pending
        if pending is None or pending.frame_index == self._published_frame_index:
            return False
        self.store.put(
            self.session_id,
            pending.frame_index,
            pending.timestamp_ms,
            pending.frame,
            pending.detections,
            self.settings,
        )
        self._published_frame_index = pending.frame_index
        self._next_publish_at = now + self.interval
        return True
