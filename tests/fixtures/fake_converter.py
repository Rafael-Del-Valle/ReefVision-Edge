from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fixtures.build_tiny_model import build_tiny_yolo_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    checkpoint = Path(request["checkpoint_path"])
    checkpoint_bytes = checkpoint.read_bytes()
    if checkpoint_bytes == b"slow checkpoint":
        time.sleep(0.5)
    if checkpoint_bytes == b"malformed checkpoint":
        write_result(args.result, {"ok": False, "error": "Checkpoint could not be loaded."})
        raise SystemExit(2)

    inspection = {
        "task": "detect",
        "architecture_family": "ultralytics_yolov8",
        "supported": True,
        "class_names": ["coral-open", "coral-closed"],
        "input_width": 640,
        "input_height": 640,
        "source_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "versions": {
            "python": "3.12.10",
            "ultralytics": "8.4.215",
            "torch": "2.15.0",
            "onnx": "1.23.2",
        },
    }
    if request["operation"] == "inspect":
        write_result(args.result, {"ok": True, "operation": "inspect", "inspection": inspection})
        return

    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = build_tiny_yolo_model(output_dir / "model.onnx")
    onnx_bytes = onnx_path.read_bytes()
    confidence = float(request["confidence_threshold"])
    iou = float(request["iou_threshold"])
    provenance = {
        "source_format": "ultralytics_pt",
        "source_filename": checkpoint.name,
        "source_sha256": inspection["source_sha256"],
        "derived_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "checkpoint_task": "detect",
        "architecture_family": "ultralytics_yolov8",
        "exporter_name": "ultralytics",
        "exporter_version": "8.4.215",
        "runtime_versions": inspection["versions"],
        "export_settings": {
            "batch": 1,
            "dynamic": False,
            "fp16": False,
            "nms": False,
            "opset": 17,
            "simplify": False,
        },
        "class_names": inspection["class_names"],
        "input_width": 640,
        "input_height": 640,
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
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "converted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    metadata = {
        "name": request["model_name"],
        "format": "ultralytics_yolo_detect_v1",
        "input_width": 640,
        "input_height": 640,
        "classes": {"0": "coral-open", "1": "coral-closed"},
        "preprocessing": provenance["preprocessing"],
        "output": provenance["output"],
        "confidence_threshold": confidence,
        "iou_threshold": iou,
    }
    write_result(
        args.result,
        {
            "ok": True,
            "operation": "convert",
            "inspection": inspection,
            "onnx_path": str(onnx_path),
            "metadata": metadata,
            "provenance": provenance,
        },
    )


def write_result(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
