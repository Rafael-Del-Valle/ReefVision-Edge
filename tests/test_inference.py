from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from reefvision_edge.db import Database
from reefvision_edge.geometry import letterbox, restore_box
from reefvision_edge.inference import YoloOnnxDetector
from reefvision_edge.model_registry import ModelRegistry
from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.test_model_registry import valid_metadata


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
