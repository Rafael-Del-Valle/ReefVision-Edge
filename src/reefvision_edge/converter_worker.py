from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


class ConverterCompatibilityError(ValueError):
    pass


def _ordered_class_names(value: object) -> list[str]:
    if isinstance(value, Mapping):
        try:
            keys = sorted(int(key) for key in value)
            if keys != list(range(len(keys))):
                raise ConverterCompatibilityError(
                    "Checkpoint class IDs must be contiguous integers starting at zero."
                )
            names = [str(value[key] if key in value else value[str(key)]) for key in keys]
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ConverterCompatibilityError):
                raise
            raise ConverterCompatibilityError(
                "Checkpoint class labels could not be interpreted."
            ) from exc
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        names = [str(item) for item in value]
    else:
        names = []
    if not names or any(not name.strip() for name in names):
        raise ConverterCompatibilityError(
            "Checkpoint class labels are missing. Supply the matching args.yaml or data.yaml."
        )
    return names


def validate_checkpoint_descriptor(value: Mapping[str, object]) -> dict[str, object]:
    task = str(value.get("task", "unknown"))
    family = str(value.get("architecture_family", "unknown"))
    if task != "detect":
        raise ConverterCompatibilityError(
            f"Detected task '{task}'. ReefVision-Edge currently supports detection "
            "checkpoints only."
        )
    if family != "ultralytics_yolov8":
        raise ConverterCompatibilityError(
            f"Detected architecture '{family}'. ReefVision-Edge currently supports "
            "Ultralytics YOLOv8 detection checkpoints only."
        )
    names = _ordered_class_names(value.get("class_names"))
    try:
        width = int(cast(Any, value.get("input_width", 0)))
        height = int(cast(Any, value.get("input_height", 0)))
    except (TypeError, ValueError) as exc:
        raise ConverterCompatibilityError("Checkpoint input shape is invalid.") from exc
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ConverterCompatibilityError(
            "Checkpoint input width and height must each be a multiple of 32."
        )
    return {
        "task": task,
        "architecture_family": family,
        "class_names": names,
        "input_width": width,
        "input_height": height,
    }


def _config_values(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml  # type: ignore[import-untyped]

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _input_shape(value: object) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        dimensions = [int(item) for item in value]
        if len(dimensions) == 1:
            return dimensions[0], dimensions[0]
        if len(dimensions) == 2:
            return dimensions[1], dimensions[0]
    return 640, 640


def _architecture_family(model: object, checkpoint: Path) -> str:
    candidates: list[str] = [checkpoint.name]
    checkpoint_value = getattr(model, "ckpt", None)
    if isinstance(checkpoint_value, Mapping):
        train_args = checkpoint_value.get("train_args")
        if isinstance(train_args, Mapping):
            candidates.append(str(train_args.get("model", "")))
    loaded = getattr(model, "model", None)
    if loaded is not None:
        candidates.append(str(getattr(loaded, "yaml_file", "")))
        yaml_value = getattr(loaded, "yaml", None)
        if isinstance(yaml_value, Mapping):
            candidates.extend(
                str(yaml_value.get(key, "")) for key in ("yaml_file", "model")
            )
    return (
        "ultralytics_yolov8"
        if any("yolov8" in candidate.casefold() for candidate in candidates)
        else "unknown_ultralytics"
    )


def inspect_checkpoint(checkpoint: Path, config_path: Path | None) -> dict[str, object]:
    import onnx
    import torch  # type: ignore[import-not-found]
    from ultralytics import YOLO  # type: ignore[import-not-found]
    from ultralytics import __version__ as ultralytics_version

    model = YOLO(str(checkpoint))
    config = _config_values(config_path)
    checkpoint_value = getattr(model, "ckpt", None)
    train_args: Mapping[str, object] = {}
    if isinstance(checkpoint_value, Mapping) and isinstance(
        checkpoint_value.get("train_args"), Mapping
    ):
        train_args = checkpoint_value["train_args"]
    names = getattr(model, "names", None) or config.get("names")
    width, height = _input_shape(train_args.get("imgsz") or config.get("imgsz") or 640)
    descriptor = validate_checkpoint_descriptor(
        {
            "task": getattr(model, "task", None)
            or getattr(getattr(model, "model", None), "task", "unknown"),
            "architecture_family": _architecture_family(model, checkpoint),
            "class_names": names,
            "input_width": width,
            "input_height": height,
        }
    )
    descriptor.update(
        {
            "supported": True,
            "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "versions": {
                "python": platform.python_version(),
                "ultralytics": ultralytics_version,
                "torch": torch.__version__,
                "onnx": onnx.__version__,
            },
        }
    )
    return descriptor


def convert_checkpoint(request: Mapping[str, object]) -> dict[str, object]:
    from ultralytics import YOLO

    checkpoint = Path(str(request["checkpoint_path"]))
    config_path_value = request.get("config_path")
    config_path = Path(str(config_path_value)) if config_path_value else None
    inspection = inspect_checkpoint(checkpoint, config_path)
    model = YOLO(str(checkpoint))
    width = int(cast(Any, inspection["input_width"]))
    height = int(cast(Any, inspection["input_height"]))
    export_size: int | tuple[int, int] = width if width == height else (height, width)
    exported = Path(
        str(
            model.export(
                format="onnx",
                imgsz=export_size,
                batch=1,
                dynamic=False,
                simplify=False,
                opset=17,
                nms=False,
                half=False,
                device="cpu",
            )
        )
    )
    output_dir = Path(str(request["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "model.onnx"
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    if not target.is_file():
        raise ConverterCompatibilityError("Ultralytics export did not produce an ONNX file.")
    onnx_bytes = target.read_bytes()
    confidence = float(cast(Any, request["confidence_threshold"]))
    iou = float(cast(Any, request["iou_threshold"]))
    class_names = [str(item) for item in cast(Sequence[object], inspection["class_names"])]
    versions = cast(Mapping[str, object], inspection["versions"])
    preprocessing = {
        "color_order": "rgb",
        "scale": "0_1",
        "letterbox": "centered",
    }
    output = {
        "layout": "1x(4+classes)xN",
        "box_format": "xywh",
        "score_format": "class_probabilities",
    }
    provenance = {
        "source_format": "ultralytics_pt",
        "source_filename": str(request["original_filename"]),
        "source_sha256": inspection["source_sha256"],
        "derived_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "checkpoint_task": inspection["task"],
        "architecture_family": inspection["architecture_family"],
        "exporter_name": "ultralytics",
        "exporter_version": versions["ultralytics"],
        "runtime_versions": versions,
        "export_settings": {
            "batch": 1,
            "dynamic": False,
            "fp16": False,
            "nms": False,
            "opset": 17,
            "simplify": False,
        },
        "class_names": class_names,
        "input_width": width,
        "input_height": height,
        "preprocessing": preprocessing,
        "output": output,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "converted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    metadata = {
        "name": str(request["model_name"]),
        "format": "ultralytics_yolo_detect_v1",
        "input_width": width,
        "input_height": height,
        "classes": {str(index): name for index, name in enumerate(class_names)},
        "preprocessing": preprocessing,
        "output": output,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "supported_profiles": ["full_frame", "thesis_tiling"],
    }
    return {
        "ok": True,
        "operation": "convert",
        "inspection": inspection,
        "onnx_path": str(target),
        "metadata": metadata,
        "provenance": provenance,
    }


def _write_result(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="reefvision-edge-model-converter")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        checkpoint = Path(str(request["checkpoint_path"]))
        config_value = request.get("config_path")
        config_path = Path(str(config_value)) if config_value else None
        if request["operation"] == "inspect":
            result: dict[str, object] = {
                "ok": True,
                "operation": "inspect",
                "inspection": inspect_checkpoint(checkpoint, config_path),
            }
        elif request["operation"] == "convert":
            result = convert_checkpoint(request)
        else:
            raise ValueError(f"Unknown converter operation {request['operation']!r}.")
        _write_result(args.result, result)
    except Exception as exc:
        _write_result(
            args.result,
            {"ok": False, "error": str(exc) or type(exc).__name__},
        )
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
