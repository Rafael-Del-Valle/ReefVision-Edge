from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CaptureDevice:
    id: str
    label: str
    index: int
    backend: str
    available: bool = True


def discover_capture_devices(max_devices: int = 8) -> list[CaptureDevice]:
    if platform.system() != "Windows":
        return []
    if max_devices < 1:
        return []
    backends = (
        ("msmf", "Media Foundation", cv2.CAP_MSMF),
        ("dshow", "DirectShow", cv2.CAP_DSHOW),
    )
    devices: list[CaptureDevice] = []
    for index in range(max_devices):
        for backend_id, backend_name, backend_value in backends:
            capture = cv2.VideoCapture()
            try:
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 500)
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 500)
                capture.open(index, backend_value)
                readable, _ = capture.read() if capture.isOpened() else (False, None)
                if not readable:
                    continue
                devices.append(
                    CaptureDevice(
                        id=f"camera:{backend_id}:{index}",
                        label=f"Camera {index + 1} ({backend_name})",
                        index=index,
                        backend=backend_name,
                    )
                )
                break
            finally:
                capture.release()
    return devices
