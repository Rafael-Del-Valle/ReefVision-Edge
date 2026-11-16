from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from reefvision_edge.media_store import MediaStore
from reefvision_edge.sources import RecordedVideoSource, SourceError, camera_source

MonitorState = Literal[
    "idle", "connecting", "receiving", "replaying", "stalled", "disconnected", "error"
]


@dataclass(frozen=True)
class MonitorStatus:
    state: MonitorState
    source_id: str | None
    frame_index: int
    detail: str | None
    updated_at: float


class MonitorPreviewService:
    def __init__(self, media_store: MediaStore) -> None:
        self.media_store = media_store
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = MonitorStatus("idle", None, -1, None, time.time())
        self._frame: NDArray[np.uint8] | None = None

    def select(self, source_id: str) -> MonitorStatus:
        record = self.media_store.get_record(source_id)
        self.stop()
        with self._lock:
            self._stop = threading.Event()
            self._status = MonitorStatus(
                "connecting", source_id, -1, "Opening source.", time.time()
            )
            self._frame = None
            self._thread = threading.Thread(
                target=self._run,
                args=(record, self._stop),
                name="reefvision-edge-monitor-preview",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            stop = self._stop
        stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self._lock:
            self._thread = None
            self._frame = None
            self._status = MonitorStatus("idle", None, -1, None, time.time())

    def shutdown(self) -> None:
        self.stop()

    def release_for_inference(self, source_id: str) -> None:
        if self.status().source_id == source_id:
            self.stop()

    def status(self) -> MonitorStatus:
        with self._lock:
            return self._status

    def jpeg(self) -> bytes:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            raise KeyError("No monitor preview frame is available.")
        encoded, value = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not encoded:
            raise SourceError("Monitor preview frame could not be encoded.")
        return value.tobytes()

    def _put_frame(
        self, source_id: str, state: MonitorState, frame_index: int, frame: NDArray[np.uint8]
    ) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._status = MonitorStatus(state, source_id, frame_index, None, time.time())

    def _set_terminal(self, source_id: str, state: MonitorState, detail: str) -> None:
        with self._lock:
            if self._status.source_id == source_id:
                self._status = MonitorStatus(
                    state, source_id, self._status.frame_index, detail, time.time()
                )

    def _run(self, record: dict[str, object], stopped: threading.Event) -> None:
        source_id = str(record["id"])
        try:
            if record["kind"] == "video":
                video_source = RecordedVideoSource(Path(str(record["path"])))
                delay = (
                    1 / video_source.probe.fps if video_source.probe.fps > 0 else 0.033
                )
                output_index = 0
                while not stopped.is_set():
                    for item in video_source.frames():
                        if stopped.is_set():
                            return
                        self._put_frame(source_id, "replaying", output_index, item.frame)
                        output_index += 1
                        stopped.wait(delay)
            elif record["kind"] == "camera":
                live_source = camera_source(str(record["path"]))
                for item in live_source.frames(queue_size=1, stop_event=stopped):
                    if stopped.is_set():
                        return
                    self._put_frame(source_id, "receiving", item.frame_index, item.frame)
            else:
                raise SourceError(f"Unsupported source kind {record['kind']}.")
        except SourceError as exc:
            state: MonitorState = (
                "disconnected" if record.get("kind") == "camera" else "error"
            )
            self._set_terminal(source_id, state, str(exc))
        except Exception as exc:
            self._set_terminal(source_id, "error", str(exc) or type(exc).__name__)
