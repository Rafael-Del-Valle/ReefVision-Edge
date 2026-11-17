from __future__ import annotations

import pytest

from reefvision_edge.converter_worker import (
    ConverterCompatibilityError,
    validate_checkpoint_descriptor,
)


def descriptor(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task": "detect",
        "architecture_family": "ultralytics_yolov8",
        "class_names": {0: "coral-open", 1: "coral-closed"},
        "input_width": 640,
        "input_height": 640,
    }
    value.update(updates)
    return value


def test_descriptor_preserves_contiguous_class_order() -> None:
    result = validate_checkpoint_descriptor(descriptor())

    assert result["class_names"] == ["coral-open", "coral-closed"]
    assert result["task"] == "detect"
    assert result["architecture_family"] == "ultralytics_yolov8"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"task": "segment"}, "detection"),
        ({"architecture_family": "ultralytics_yolo11"}, "YOLOv8"),
        ({"class_names": {}}, "class labels"),
        ({"class_names": {0: "open", 2: "closed"}}, "contiguous"),
        ({"input_width": 641}, "multiple of 32"),
    ],
)
def test_descriptor_rejects_unsupported_or_ambiguous_checkpoints(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ConverterCompatibilityError, match=message):
        validate_checkpoint_descriptor(descriptor(**updates))
