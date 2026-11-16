from __future__ import annotations

import platform
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    container: str
    timestamp_basis: str = "frame_index / reported_fps"


@dataclass(frozen=True)
class SourceFrame:
    frame_index: int
    timestamp_ms: float | None
    frame: NDArray[np.uint8]


@dataclass(frozen=True)
class VideoSource:
    id: str
    name: str
    path: Path
    sha256: str
    probe: VideoProbe


def _fourcc_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4)).replace(
        "\x00", " "
    )


def probe_video(path: Path) -> VideoProbe:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SourceError(f"Video decoder could not open {path.name}.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, _ = capture.read()
        if not readable or width <= 0 or height <= 0:
            raise SourceError(f"Video decoder could not decode {path.name}.")
        return VideoProbe(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            codec=_fourcc_text(capture.get(cv2.CAP_PROP_FOURCC)),
            container=path.suffix.casefold().lstrip(".") or "unknown",
        )
    finally:
        capture.release()


class RecordedVideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.probe = probe_video(path)

    def frames(self, sample_every: int = 1) -> Iterator[SourceFrame]:
        if sample_every < 1:
            raise ValueError("sample_every must be at least one")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise SourceError(f"Video decoder could not open {self.path.name}.")
        frame_index = 0
        try:
            while True:
                readable, frame = capture.read()
                if not readable:
                    if (
                        self.probe.frame_count > 1
                        and frame_index < self.probe.frame_count - 1
                    ):
                        raise SourceError(
                            f"Video decoder ended early at frame {frame_index} of "
                            f"{self.probe.frame_count} for {self.path.name}."
                        )
                    break
                if frame_index % sample_every == 0:
                    timestamp_ms = (
                        frame_index / self.probe.fps * 1000
                        if self.probe.fps > 0
                        else None
                    )
                    yield SourceFrame(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        frame=cast(NDArray[np.uint8], frame),
                    )
                frame_index += 1
        finally:
            capture.release()


class NewestFrameQueue:
    def __init__(self, max_size: int = 1) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least one")
        self._queue: queue.Queue[NDArray[np.uint8]] = queue.Queue(maxsize=max_size)
        self.dropped = 0

    def put(self, frame: NDArray[np.uint8]) -> None:
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped += 1
        except queue.Empty:
            pass
        self._queue.put_nowait(frame)

    def get(self, timeout: float | None = None) -> NDArray[np.uint8]:
        return self._queue.get(timeout=timeout)


class V4L2CameraSource:
    def __init__(self, device: Path) -> None:
        if platform.system() != "Linux":
            raise SourceError("V4L2 camera capture is available only on Linux.")
        if not device.exists():
            raise SourceError(f"Camera device {device} does not exist.")
        self.device = device
        self.dropped_frames = 0

    def frames(
        self,
        queue_size: int = 1,
        stop_event: threading.Event | None = None,
    ) -> Iterator[SourceFrame]:
        capture = cv2.VideoCapture(str(self.device), cv2.CAP_V4L2)
        if not capture.isOpened():
            raise SourceError(f"Camera device {self.device} could not be opened.")
        latest = NewestFrameQueue(queue_size)
        stopped = threading.Event()
        producer_done = threading.Event()
        failure: list[str] = []

        def produce() -> None:
            try:
                while not stopped.is_set():
                    readable, frame = capture.read()
                    if not readable:
                        failure.append(f"Camera device {self.device} stopped delivering frames.")
                        return
                    latest.put(cast(NDArray[np.uint8], frame))
            finally:
                producer_done.set()

        thread = threading.Thread(target=produce, name="reefvision-edge-camera", daemon=True)
        thread.start()
        frame_index = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    frame = latest.get(timeout=0.25)
                except queue.Empty:
                    if producer_done.is_set():
                        raise SourceError(
                            failure[0] if failure else "Camera capture ended."
                        ) from None
                    continue
                self.dropped_frames = latest.dropped
                yield SourceFrame(frame_index=frame_index, timestamp_ms=None, frame=frame)
                frame_index += 1
        finally:
            stopped.set()
            capture.release()
            thread.join(timeout=2)
