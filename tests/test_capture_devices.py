from __future__ import annotations

import pytest

from reefvision_edge.capture_devices import discover_capture_devices


def test_windows_discovery_prefers_media_foundation_and_releases_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[FakeCapture] = []

    class FakeCapture:
        def __init__(self, index: int, backend: int) -> None:
            self.index = index
            self.backend = backend
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            return self.index == 0

        def read(self):  # type: ignore[no-untyped-def]
            return self.index == 0, object()

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr("reefvision_edge.capture_devices.platform.system", lambda: "Windows")
    monkeypatch.setattr("reefvision_edge.capture_devices.cv2.VideoCapture", FakeCapture)

    devices = discover_capture_devices(max_devices=2)

    assert [(item.id, item.index, item.backend) for item in devices] == [
        ("camera:msmf:0", 0, "Media Foundation")
    ]
    assert devices[0].label == "Camera 1 (Media Foundation)"
    assert all(capture.released for capture in captures)


def test_capture_discovery_is_empty_on_unsupported_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reefvision_edge.capture_devices.platform.system", lambda: "Darwin")

    assert discover_capture_devices() == []
