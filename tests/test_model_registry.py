from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from tests.fixtures.build_tiny_model import build_tiny_yolo_model

from reefvision_edge.db import Database
from reefvision_edge.domain import ConversionProvenance, InferenceProfile, ModelMetadata
from reefvision_edge.model_registry import ModelRegistrationError, ModelRegistry


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


def conversion_provenance(
    checkpoint: bytes,
    onnx: bytes,
    class_names: tuple[str, ...] = ("coral-open", "coral-closed"),
) -> ConversionProvenance:
    return ConversionProvenance(
        source_format="ultralytics_pt",
        source_filename="best.pt",
        source_sha256=hashlib.sha256(checkpoint).hexdigest(),
        derived_sha256=hashlib.sha256(onnx).hexdigest(),
        checkpoint_task="detect",
        architecture_family="ultralytics_yolov8",
        exporter_name="ultralytics",
        exporter_version="8.4.215",
        runtime_versions={"python": "3.12.10", "torch": "2.15.0", "onnx": "1.23.2"},
        export_settings={
            "batch": 1,
            "dynamic": False,
            "fp16": False,
            "nms": False,
            "opset": 17,
            "simplify": False,
        },
        class_names=class_names,
        input_width=640,
        input_height=640,
        preprocessing={
            "color_order": "rgb",
            "scale": "0_1",
            "letterbox": "centered",
        },
        output={
            "layout": "1x(4+classes)xN",
            "box_format": "xywh",
            "score_format": "class_probabilities",
        },
        confidence_threshold=0.25,
        iou_threshold=0.45,
        converted_at="2026-11-16T21:00:00.000Z",
    )


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


def test_model_upload_limit_is_enforced_before_runtime_load(tmp_path: Path) -> None:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    limited = ModelRegistry(database, tmp_path / "models", max_model_bytes=4)

    with pytest.raises(ModelRegistrationError, match="upload limit"):
        limited.register(io.BytesIO(b"12345"), "large.onnx", valid_metadata())


def test_register_converted_preserves_source_config_and_conversion_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = b"trusted synthetic checkpoint"
    config = b"names: [coral-open, coral-closed]\n"
    checkpoint_path = tmp_path / "best.pt"
    config_path = tmp_path / "args.yaml"
    checkpoint_path.write_bytes(checkpoint)
    config_path.write_bytes(config)
    onnx_path = build_tiny_yolo_model(tmp_path / "converted.onnx")
    onnx_bytes = onnx_path.read_bytes()
    metadata = valid_metadata()
    metadata["name"] = "Synthetic thesis detector"
    metadata["classes"] = {"0": "coral-open", "1": "coral-closed"}
    provenance = conversion_provenance(checkpoint, onnx_bytes)

    registered = registry(tmp_path).register_converted(
        checkpoint_path,
        onnx_path,
        metadata,
        provenance,
        source_config_path=config_path,
    )

    model_dir = registered.path.parent
    assert (model_dir / "source.pt").read_bytes() == checkpoint
    assert (model_dir / "source-config.yaml").read_bytes() == config
    assert (model_dir / "model.onnx").read_bytes() == onnx_bytes
    assert json.loads((model_dir / "conversion.json").read_text(encoding="utf-8")) == (
        provenance.model_dump(mode="json")
    )
    assert registered.metadata.conversion == provenance
    assert registered.metadata.supported_profiles == (
        InferenceProfile.FULL_FRAME,
        InferenceProfile.THESIS_TILING,
    )
    assert registered.model_sha256 == provenance.derived_sha256
    assert registered.metadata.classes == {0: "coral-open", 1: "coral-closed"}
    assert checkpoint_path.read_bytes() == checkpoint
    assert config_path.read_bytes() == config


def test_register_converted_rejects_derived_hash_mismatch_without_moving_source(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"trusted checkpoint")
    onnx_path = build_tiny_yolo_model(tmp_path / "converted.onnx")
    provenance = conversion_provenance(checkpoint_path.read_bytes(), onnx_path.read_bytes())
    bad_provenance = provenance.model_copy(update={"derived_sha256": "0" * 64})
    metadata = valid_metadata()
    metadata["classes"] = {"0": "coral-open", "1": "coral-closed"}

    with pytest.raises(ModelRegistrationError, match="derived SHA-256"):
        registry(tmp_path).register_converted(
            checkpoint_path,
            onnx_path,
            metadata,
            bad_provenance,
        )

    assert checkpoint_path.read_bytes() == b"trusted checkpoint"
    assert not (tmp_path / "models").exists()
