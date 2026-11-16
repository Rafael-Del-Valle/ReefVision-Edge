from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from reefvision_edge.db import Database
from reefvision_edge.domain import ModelMetadata
from reefvision_edge.model_registry import ModelRegistrationError, ModelRegistry
from tests.fixtures.build_tiny_model import build_tiny_yolo_model


def valid_metadata() -> dict[str, object]:
    return {
        "name": "Tiny Coral Detector",
        "format": "ultralytics_yolo_detect_v1",
        "input_width": 640,
        "input_height": 640,
        "classes": {"0": "coral", "1": "fish"},
        "preprocessing": {
            "color_order": "rgb",
            "scale": "0_1",
            "letterbox": "centered",
        },
        "output": {
            "layout": "1x(4+classes)xN",
            "box_format": "xywh",
            "score_format": "class_probabilities",
        },
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
    }


def registry(tmp_path: Path) -> ModelRegistry:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    return ModelRegistry(database, tmp_path / "models")


def test_registers_model_with_canonical_hashes_and_actual_provider(tmp_path: Path) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    model_bytes = model_path.read_bytes()
    metadata = ModelMetadata.model_validate(valid_metadata())
    expected_metadata_bytes = json.dumps(
        metadata.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    registered = registry(tmp_path).register(
        io.BytesIO(model_bytes), "tiny.onnx", valid_metadata()
    )

    assert registered.model_sha256 == hashlib.sha256(model_bytes).hexdigest()
    assert registered.metadata_sha256 == hashlib.sha256(expected_metadata_bytes).hexdigest()
    assert registered.path.read_bytes() == model_bytes
    assert registered.provider.endswith("ExecutionProvider")
    assert "CPUExecutionProvider" in registered.available_providers


def test_rejects_checkpoint_extension_before_storage(tmp_path: Path) -> None:
    with pytest.raises(ModelRegistrationError, match="ONNX"):
        registry(tmp_path).register(io.BytesIO(b"checkpoint"), "best.pt", valid_metadata())

    assert list((tmp_path / "models").glob("**/*")) == []


def test_rejects_noncontiguous_class_ids(tmp_path: Path) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "tiny.onnx")
    metadata = valid_metadata()
    metadata["classes"] = {"0": "coral", "2": "fish"}

    with pytest.raises(ModelRegistrationError, match="contiguous"):
        registry(tmp_path).register(
            io.BytesIO(model_path.read_bytes()), "tiny.onnx", metadata
        )


def test_rejects_output_dimension_that_disagrees_with_class_map(tmp_path: Path) -> None:
    model_path = build_tiny_yolo_model(tmp_path / "three-class.onnx", class_count=3)

    with pytest.raises(ModelRegistrationError, match="output"):
        registry(tmp_path).register(
            io.BytesIO(model_path.read_bytes()), "three-class.onnx", valid_metadata()
        )
