from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.test_model_registry import valid_metadata

from reefvision_edge.compute import ComputeChoice, ComputeDeviceError
from reefvision_edge.db import Database
from reefvision_edge.geometry import letterbox, restore_box
from reefvision_edge.inference import YoloOnnxDetector, provider_assignment_summary
from reefvision_edge.model_registry import ModelRegistry


def test_letterbox_round_trip_restores_original_coordinates() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    transformed, mapping = letterbox(frame, width=640, height=640)
    restored = restore_box((50.0, 165.0, 150.0, 265.0), mapping)

    assert transformed.shape == (640, 640, 3)
    assert mapping.scale == 0.5
    assert mapping.pad_x == 0
    assert mapping.pad_y == 140
    assert restored == (100.0, 50.0, 300.0, 250.0)


def test_restore_box_clips_padding_and_frame_edges() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, mapping = letterbox(frame, width=640, height=640)

    assert restore_box((-10.0, 100.0, 660.0, 600.0), mapping) == (
        0.0,
        0.0,
        1280.0,
        720.0,
    )


def test_detector_returns_classified_boxes_in_original_frame_pixels(tmp_path: Path) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    registered = ModelRegistry(database, tmp_path / "models").register(
        io.BytesIO(model_path.read_bytes()), "tiny.onnx", valid_metadata()
    )
    detector = YoloOnnxDetector(registered)

    result = detector.predict(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert result.provider == registered.provider
    assert result.latency_ms >= 0
    assert [(item.class_id, item.class_name) for item in result.detections] == [
        (0, "coral"),
        (1, "fish"),
    ]
    first = result.detections[0]
    assert first.score == pytest.approx(0.9, abs=1e-5)
    assert first.box == pytest.approx((480.0, 280.0, 800.0, 440.0))


def test_detector_reselects_provider_when_registration_snapshot_is_stale(
    tmp_path: Path,
) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    registered = ModelRegistry(database, tmp_path / "models").register(
        io.BytesIO(model_path.read_bytes()), "tiny.onnx", valid_metadata()
    )

    detector = YoloOnnxDetector(
        replace(registered, provider="UnavailableExecutionProvider")
    )

    assert "CPUExecutionProvider" in detector.provider


def test_provider_assignment_summary_makes_cpu_fallback_explicit() -> None:
    class Assignment:
        def __init__(self, name: str, nodes: int) -> None:
            self.ep_name = name
            self._nodes = [object()] * nodes

        def get_nodes(self) -> list[object]:
            return self._nodes

    class Session:
        def get_provider_graph_assignment_info(self) -> list[Assignment]:
            return [
                Assignment("TensorrtExecutionProvider", 42),
                Assignment("CPUExecutionProvider", 3),
            ]

        def get_providers(self) -> list[str]:
            return ["TensorrtExecutionProvider", "CPUExecutionProvider"]

    assert provider_assignment_summary(Session()) == (
        "TensorrtExecutionProvider (42 nodes) -> "
        "CPUExecutionProvider fallback (3 nodes)"
    )

    class CpuSession(Session):
        def get_provider_graph_assignment_info(self) -> list[Assignment]:
            return [Assignment("CPUExecutionProvider", 45)]

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    assert provider_assignment_summary(CpuSession()) == "CPUExecutionProvider (45 nodes)"


def test_explicit_nvidia_detector_does_not_fall_back_when_session_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    registered = ModelRegistry(database, tmp_path / "models").register(
        io.BytesIO(model_path.read_bytes()), "tiny.onnx", valid_metadata()
    )
    monkeypatch.setattr(
        "reefvision_edge.inference.resolve_compute",
        lambda _choice: __import__(
            "reefvision_edge.compute", fromlist=["ComputeResolution"]
        ).ComputeResolution(
            requested=ComputeChoice.NVIDIA_GPU,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            active_provider="CUDAExecutionProvider",
            actual_device="NVIDIA GeForce RTX 3090",
            explanation="NVIDIA GPU was selected explicitly.",
            fell_back=False,
        ),
    )

    def fail_cuda(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("cudnn64_9.dll was not found")

    monkeypatch.setattr("reefvision_edge.inference.ort.InferenceSession", fail_cuda)

    with pytest.raises(ComputeDeviceError, match=r"cudnn64_9\.dll"):
        YoloOnnxDetector(registered, ComputeChoice.NVIDIA_GPU)


def test_explicit_nvidia_rejects_a_session_with_zero_cuda_operators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    registered = ModelRegistry(database, tmp_path / "models").register(
        io.BytesIO(model_path.read_bytes()), "tiny.onnx", valid_metadata()
    )
    resolution_type = __import__(
        "reefvision_edge.compute", fromlist=["ComputeResolution"]
    ).ComputeResolution
    monkeypatch.setattr(
        "reefvision_edge.inference.resolve_compute",
        lambda _choice: resolution_type(
            requested=ComputeChoice.NVIDIA_GPU,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            active_provider="CUDAExecutionProvider",
            actual_device="NVIDIA GeForce RTX 3090",
            explanation="NVIDIA GPU was selected explicitly.",
            fell_back=False,
        ),
    )

    class Input:
        name = "images"

    class CpuOnlySession:
        def __init__(self, *_args: object, sess_options, **_kwargs: object) -> None:
            self.options = sess_options

        def get_inputs(self) -> list[Input]:
            return [Input()]

        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_session_options(self):  # type: ignore[no-untyped-def]
            return self.options

        def run(self, *_args: object, **_kwargs: object) -> list[np.ndarray]:
            return [np.zeros((1, 6, 1), dtype=np.float32)]

        def end_profiling(self) -> str:
            target = tmp_path / "cpu-only-profile.json"
            target.write_text(
                json.dumps(
                    [
                        {
                            "cat": "Node",
                            "args": {
                                "provider": "CPUExecutionProvider",
                                "op_name": "Gemm",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            return str(target)

    monkeypatch.setattr("reefvision_edge.inference.ort.InferenceSession", CpuOnlySession)
    detector = YoloOnnxDetector(registered, ComputeChoice.NVIDIA_GPU)

    with pytest.raises(ComputeDeviceError, match="executed zero operators"):
        detector.predict(np.zeros((90, 160, 3), dtype=np.uint8))


def test_auto_falls_back_to_cpu_when_cuda_executes_zero_operators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    registered = ModelRegistry(database, tmp_path / "models").register(
        io.BytesIO(model_path.read_bytes()), "tiny.onnx", valid_metadata()
    )
    resolution_type = __import__(
        "reefvision_edge.compute", fromlist=["ComputeResolution"]
    ).ComputeResolution
    monkeypatch.setattr(
        "reefvision_edge.inference.resolve_compute",
        lambda _choice: resolution_type(
            requested=ComputeChoice.AUTO,
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            active_provider="CUDAExecutionProvider",
            actual_device="NVIDIA GeForce RTX 3090",
            explanation="Auto selected CUDA.",
            fell_back=False,
        ),
    )

    class Input:
        name = "images"

    class Session:
        instances = 0

        def __init__(self, *_args: object, sess_options, providers, **_kwargs: object) -> None:
            self.options = sess_options
            self.providers = providers
            Session.instances += 1

        def get_inputs(self) -> list[Input]:
            return [Input()]

        def get_providers(self) -> list[str]:
            return list(self.providers)

        def get_session_options(self):  # type: ignore[no-untyped-def]
            return self.options

        def run(self, *_args: object, **_kwargs: object) -> list[np.ndarray]:
            return [np.zeros((1, 6, 1), dtype=np.float32)]

        def end_profiling(self) -> str:
            target = tmp_path / "auto-cpu-profile.json"
            target.write_text(
                json.dumps(
                    [
                        {
                            "cat": "Node",
                            "args": {
                                "provider": "CPUExecutionProvider",
                                "op_name": "Gemm",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            return str(target)

    monkeypatch.setattr("reefvision_edge.inference.ort.InferenceSession", Session)
    detector = YoloOnnxDetector(registered, ComputeChoice.AUTO)

    detector.predict(np.zeros((90, 160, 3), dtype=np.uint8))

    assert Session.instances == 2
    assert detector.resolution.active_provider == "CPUExecutionProvider"
    assert detector.resolution.actual_device == "CPU"
    assert detector.resolution.fell_back is True
    assert "executed zero operators" in detector.resolution.explanation
