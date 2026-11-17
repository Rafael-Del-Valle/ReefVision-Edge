from __future__ import annotations

import platform
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

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


T = TypeVar("T")


class NewestFrameQueue(Generic[T]):
    def __init__(self, max_size: int = 1) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least one")
        self._queue: queue.Queue[T] = queue.Queue(maxsize=max_size)
        self.dropped = 0

    def put(self, frame: T) -> None:
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

    def get(self, timeout: float | None = None) -> T:
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
        latest: NewestFrameQueue[SourceFrame] = NewestFrameQueue(queue_size)
        stopped = threading.Event()
        producer_done = threading.Event()
        failure: list[str] = []
        started = time.perf_counter()

        def produce() -> None:
            frame_index = 0
            try:
                while not stopped.is_set():
                    readable, frame = capture.read()
                    if not readable:
                        failure.append(f"Camera device {self.device} stopped delivering frames.")
                        return
                    latest.put(
                        SourceFrame(
                            frame_index=frame_index,
                            timestamp_ms=(time.perf_counter() - started) * 1000,
                            frame=cast(NDArray[np.uint8], frame),
                        )
                    )
                    frame_index += 1
            finally:
                producer_done.set()

        thread = threading.Thread(target=produce, name="reefvision-edge-camera", daemon=True)
        thread.start()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    source_frame = latest.get(timeout=0.25)
                except queue.Empty:
                    if producer_done.is_set():
                        raise SourceError(
                            failure[0] if failure else "Camera capture ended."
                        ) from None
                    continue
                self.dropped_frames = latest.dropped
                yield source_frame
        finally:
            stopped.set()
            capture.release()
            thread.join(timeout=2)


class WindowsCameraSource:
    def __init__(self, index: int, *, backend: str = "msmf") -> None:
        if platform.system() != "Windows":
            raise SourceError("Windows camera capture is available only on Windows.")
        if index < 0:
            raise SourceError("Camera index must not be negative.")
        normalized = backend.casefold()
        if normalized not in {"msmf", "dshow"}:
            raise SourceError(f"Unsupported Windows camera backend {backend}.")
        self.index = index
        self.backend = normalized
        self.dropped_frames = 0

    @property
    def identifier(self) -> str:
        return f"camera:{self.backend}:{self.index}"

    def frames(
        self,
        queue_size: int = 1,
        stop_event: threading.Event | None = None,
    ) -> Iterator[SourceFrame]:
        backend_value = cv2.CAP_MSMF if self.backend == "msmf" else cv2.CAP_DSHOW
        capture = cv2.VideoCapture(self.index, backend_value)
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"Camera {self.identifier} could not be opened.")
        first_frame = configure_windows_capture(capture)
        latest: NewestFrameQueue[SourceFrame] = NewestFrameQueue(queue_size)
        stopped = threading.Event()
        producer_done = threading.Event()
        failure: list[str] = []
        started = time.perf_counter()

        def produce() -> None:
            frame_index = 0
            buffered_frame = first_frame
            try:
                while not stopped.is_set():
                    frame: NDArray[np.uint8] | None = None
                    if buffered_frame is not None:
                        readable, frame = True, buffered_frame
                        buffered_frame = None
                    else:
                        readable, raw_frame = capture.read()
                        if readable:
                            frame = cast(NDArray[np.uint8], raw_frame)
                    if not readable or frame is None:
                        failure.append(
                            f"Camera {self.identifier} stopped delivering frames."
                        )
                        return
                    latest.put(
                        SourceFrame(
                            frame_index=frame_index,
                            timestamp_ms=(time.perf_counter() - started) * 1000,
                            frame=frame,
                        )
                    )
                    frame_index += 1
            finally:
                producer_done.set()

        thread = threading.Thread(
            target=produce, name="reefvision-edge-windows-camera", daemon=True
        )
        thread.start()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    source_frame = latest.get(timeout=0.25)
                except queue.Empty:
                    if producer_done.is_set():
                        raise SourceError(
                            failure[0] if failure else "Camera capture ended."
                        ) from None
                    continue
                self.dropped_frames = latest.dropped
                yield source_frame
        finally:
            stopped.set()
            capture.release()
            thread.join(timeout=2)


WINDOWS_CAPTURE_MODES = ((1920, 1080, 30.0), (1280, 720, 30.0))


def configure_windows_capture(
    capture: cv2.VideoCapture,
) -> NDArray[np.uint8] | None:
    """Negotiate a responsive 16:9 mode and return the verified first frame."""
    if not hasattr(capture, "set"):
        return None
    last_frame: NDArray[np.uint8] | None = None
    for width, height, fps in WINDOWS_CAPTURE_MODES:
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore[attr-defined]
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        readable, raw_frame = capture.read()
        if not readable or raw_frame is None:
            continue
        frame = cast(NDArray[np.uint8], raw_frame)
        last_frame = frame
        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
    return last_frame


def camera_source(identifier: str) -> V4L2CameraSource | WindowsCameraSource:
    if identifier.startswith("camera:"):
        fields = identifier.split(":")
        if len(fields) != 3:
            raise SourceError(f"Invalid Windows camera identifier {identifier}.")
        try:
            index = int(fields[2])
        except ValueError as exc:
            raise SourceError(f"Invalid Windows camera identifier {identifier}.") from exc
        return WindowsCameraSource(index, backend=fields[1])
    return V4L2CameraSource(Path(identifier))
