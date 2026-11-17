from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import httpx


def _sorted_by_class(
    values: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[(int(value.get("frame_index", 0)), int(value["class_id"]))].append(value)
    for detections in grouped.values():
        detections.sort(key=lambda item: tuple(float(value) for value in item["box"]))
    return grouped


def compare_detection_sets(
    pytorch: list[dict[str, Any]],
    onnx: list[dict[str, Any]],
    *,
    confidence_tolerance: float,
    box_tolerance_px: float,
) -> dict[str, object]:
    pytorch_counts = Counter(int(item["class_id"]) for item in pytorch)
    onnx_counts = Counter(int(item["class_id"]) for item in onnx)
    class_counts_match = pytorch_counts == onnx_counts
    pytorch_by_class = _sorted_by_class(pytorch)
    onnx_by_class = _sorted_by_class(onnx)
    frame_class_counts_match = {key: len(pytorch_by_class[key]) for key in pytorch_by_class} == {
        key: len(onnx_by_class[key]) for key in onnx_by_class
    }
    confidence_deltas: list[float] = []
    box_deltas: list[float] = []
    for key in sorted(set(pytorch_by_class) | set(onnx_by_class)):
        for left, right in zip(pytorch_by_class[key], onnx_by_class[key], strict=False):
            confidence_deltas.append(abs(float(left["confidence"]) - float(right["confidence"])))
            box_deltas.extend(
                abs(float(first) - float(second))
                for first, second in zip(left["box"], right["box"], strict=False)
            )
    max_confidence_delta = max(confidence_deltas, default=0.0)
    max_box_delta = max(box_deltas, default=0.0)
    return {
        "passed": class_counts_match
        and frame_class_counts_match
        and max_confidence_delta <= confidence_tolerance
        and max_box_delta <= box_tolerance_px,
        "class_counts_match": class_counts_match,
        "frame_class_counts_match": frame_class_counts_match,
        "pytorch_counts": dict(sorted(pytorch_counts.items())),
        "onnx_counts": dict(sorted(onnx_counts.items())),
        "max_confidence_delta": max_confidence_delta,
        "max_box_delta_px": max_box_delta,
        "confidence_tolerance": confidence_tolerance,
        "box_tolerance_px": box_tolerance_px,
    }


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.content


def _wait_import(
    client: httpx.Client, import_id: str, expected: set[str], timeout: float = 600
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _request(client, "GET", f"/api/model-imports/{import_id}")
        if value["status"] in expected:
            return value
        if value["status"] in {"failed", "interrupted"}:
            raise AssertionError(f"Model import failed: {value}")
        time.sleep(0.25)
    raise AssertionError(f"Model import {import_id} did not reach {expected}.")


def _wait_session(client: httpx.Client, session_id: str, timeout: float = 600) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _request(client, "GET", f"/api/sessions/{session_id}")
        if value["status"] not in {"queued", "running"}:
            if value["status"] != "completed":
                raise AssertionError(f"Session failed: {value}")
            return value
        time.sleep(0.1)
    raise AssertionError(f"Session {session_id} did not finish.")


def _upload_source(client: httpx.Client, video: Path) -> dict[str, Any]:
    with video.open("rb") as stream:
        return _request(
            client,
            "POST",
            "/api/sources/video",
            files={"video": (video.name, stream, "video/x-msvideo")},
        )


def _run_session(
    client: httpx.Client,
    *,
    model_id: str,
    source_id: str,
    profile: str,
) -> dict[str, Any]:
    created = _request(
        client,
        "POST",
        "/api/sessions",
        json={
            "model_id": model_id,
            "source_id": source_id,
            "sample_every": 1,
            "inference_profile": profile,
            "compute_device": "nvidia",
            "record_data": True,
            "record_monitor": True,
            "preview_overlays": {
                "timer": False,
                "trails": False,
                "boxes": True,
                "masks": False,
                "labels": True,
            },
            "recording_overlays": {
                "timer": True,
                "trails": False,
                "boxes": True,
                "masks": False,
                "labels": True,
            },
            "enable_profiling": True,
        },
    )
    _request(client, "POST", f"/api/sessions/{created['id']}/start")
    return _wait_session(client, created["id"])


def _export_session(client: httpx.Client, session_id: str, output: Path) -> dict[str, Any]:
    output.write_bytes(_request(client, "GET", f"/api/sessions/{session_id}/export"))
    extracted = output.with_suffix("")
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output) as bundle:
        bundle.extractall(extracted)
        manifest = json.loads(bundle.read("manifest.json"))
        frames = list(csv.DictReader(io.StringIO(bundle.read("frames.csv").decode("utf-8"))))
        detections = list(
            csv.DictReader(io.StringIO(bundle.read("detections.csv").decode("utf-8")))
        )
    video = extracted / "monitor.mp4"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise AssertionError(f"Could not decode {video}.")
    decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return {
        "zip": str(output.resolve()),
        "manifest": manifest,
        "frames": frames,
        "detections": detections,
        "csv": str((extracted / "detections.csv").resolve()),
        "video": str(video.resolve()),
        "decoded": {
            "frames": decoded_frames,
            "width": decoded_width,
            "height": decoded_height,
        },
    }


def _onnx_predictions(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "frame_index": int(row["frame_index"]),
            "class_id": int(row["class_id"]),
            "confidence": float(row["score"]),
            "box": [float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")],
        }
        for row in rows
    ]


def _cuda_summary(export: dict[str, Any]) -> dict[str, Any]:
    profile = export["manifest"]["compute"]["operator_profile"]
    providers = profile["provider_counts"]
    cuda_events = int(providers.get("CUDAExecutionProvider", 0))
    if cuda_events <= 0:
        raise AssertionError(f"ONNX Runtime recorded no CUDA operators: {profile}")
    return {
        "operators": cuda_events,
        "cpu_operators": int(providers.get("CPUExecutionProvider", 0)),
        "cuda_operator_names": profile.get("cuda_operators", []),
        "cpu_operator_names": profile.get("cpu_operators", []),
    }


def _artifact_summary(export: dict[str, Any]) -> dict[str, Any]:
    manifest = export["manifest"]
    return {
        "zip": export["zip"],
        "csv": export["csv"],
        "video": export["video"],
        "decoded": export["decoded"],
        "frame_rows": len(export["frames"]),
        "detection_rows": len(export["detections"]),
        "class_map": manifest["model"]["class_map"],
        "recording": manifest["recording"],
        "inference_profile": manifest["inference_profile"],
    }


def _driver() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    name, driver, memory = (value.strip() for value in completed.stdout.splitlines()[0].split(","))
    return {"name": name, "driver": driver, "memory_total_mb": memory}


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture_dir.resolve()
    checkpoint = fixture / "synthetic-yolov8n-six-class.pt"
    configuration = fixture / "args.yaml"
    regular_video = fixture / "synthetic-coral.avi"
    tiled_video = fixture / "synthetic-coral-tiled.avi"
    expected_classes = [
        "Acropora_open",
        "Acropora_closed",
        "Montipora_open",
        "Montipora_closed",
        "Pocillopora_open",
        "Pocillopora_closed",
    ]
    with httpx.Client(base_url=args.base_url, timeout=600) as client:
        with checkpoint.open("rb") as checkpoint_stream, configuration.open("rb") as config_stream:
            started = _request(
                client,
                "POST",
                "/api/model-imports",
                files={
                    "checkpoint": (
                        checkpoint.name,
                        checkpoint_stream,
                        "application/octet-stream",
                    ),
                    "configuration": (
                        configuration.name,
                        config_stream,
                        "application/yaml",
                    ),
                },
                data={"trusted": "true"},
            )
        inspected = _wait_import(client, started["id"], {"awaiting_confirmation"})
        if inspected["inspection"]["task"] != "detect":
            raise AssertionError(f"Unexpected task: {inspected}")
        if inspected["inspection"]["class_names"] != expected_classes:
            raise AssertionError(f"Class order changed: {inspected}")
        _request(
            client,
            "POST",
            f"/api/model-imports/{started['id']}/convert",
            json={
                "model_name": "Synthetic thesis YOLOv8n",
                "confidence_threshold": 0.999,
                "iou_threshold": 1.0,
            },
        )
        converted = _wait_import(client, started["id"], {"completed"})
        models = _request(client, "GET", "/api/models")
        model = next(item for item in models if item["id"] == converted["model_id"])
        if list(model["classes"].values()) != expected_classes:
            raise AssertionError(f"Registered classes changed: {model}")

        full_source = _upload_source(client, regular_video)
        full_session = _run_session(
            client,
            model_id=model["id"],
            source_id=full_source["id"],
            profile="full_frame",
        )
        full_export = _export_session(client, full_session["id"], output / "full-frame.zip")
        tiled_source = _upload_source(client, tiled_video)
        tiled_session = _run_session(
            client,
            model_id=model["id"],
            source_id=tiled_source["id"],
            profile="thesis_tiling",
        )
        tiled_export = _export_session(client, tiled_session["id"], output / "thesis-tiling.zip")

    pytorch_predictions = json.loads(args.reference_json.read_text(encoding="utf-8"))
    parity = compare_detection_sets(
        pytorch_predictions,
        _onnx_predictions(full_export["detections"]),
        confidence_tolerance=0.01,
        box_tolerance_px=1.0,
    )
    if not parity["passed"]:
        raise AssertionError(f"PyTorch/ONNX parity failed: {parity}")
    for exported in (full_export, tiled_export):
        if not exported["frames"] or not exported["detections"]:
            raise AssertionError("Imported model did not persist frame and detection rows.")
        names = {row["class_name"] for row in exported["detections"]}
        if not names <= set(expected_classes):
            raise AssertionError(f"Unexpected CSV class names: {names}")
    full_cuda = _cuda_summary(full_export)
    tiled_cuda = _cuda_summary(tiled_export)
    if full_session["provider"] != "CUDAExecutionProvider":
        raise AssertionError(f"Full-frame session did not use CUDA: {full_session}")
    if tiled_session["provider"] != "CUDAExecutionProvider":
        raise AssertionError(f"Tiled session did not use CUDA: {tiled_session}")

    metadata = json.loads((args.bundle_path / "BUNDLE-METADATA.json").read_text(encoding="utf-8"))
    result = {
        "tested_commit": metadata["commit"],
        "bundle_path": str(args.bundle_path.resolve()),
        "fixture": json.loads((fixture / "fixture-manifest.json").read_text(encoding="utf-8")),
        "converter": {
            "status": "passed",
            "inspection": inspected["inspection"],
            "source_sha256": converted["source_sha256"],
            "model_id": converted["model_id"],
            "registered_model": model,
        },
        "driver": _driver(),
        "cuda": full_cuda,
        "parity": parity,
        "full_frame": {
            "status": full_session["status"],
            "session": full_session,
            "cuda": full_cuda,
            "artifacts": _artifact_summary(full_export),
        },
        "thesis_tiling": {
            "status": tiled_session["status"],
            "session": tiled_session,
            "cuda": tiled_cuda,
            "artifacts": _artifact_summary(tiled_export),
        },
        "remaining_acceptance": [
            "real best.pt checkpoint",
            "matching args.yaml or data.yaml when available",
            "representative coral images or a short clip",
            "real-model RTX 3090 PyTorch/ONNX parity and full-frame/tiled measurements",
            "original thesis overlap-merging settings if they differ from class-aware NMS",
        ],
    }
    report = output / "result.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence-tolerance", type=float, default=0.01)
    parser.add_argument("--box-tolerance-px", type=float, default=1.0)
    parser.add_argument("--base-url")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--reference-json", type=Path)
    args = parser.parse_args()
    if args.base_url:
        required = (
            args.output_dir,
            args.bundle_path,
            args.fixture_dir,
            args.reference_json,
        )
        if any(value is None for value in required):
            parser.error("Acceptance mode requires output, bundle, fixture, and reference paths.")
        print(json.dumps(run_acceptance(args), sort_keys=True))
        return
    if args.pytorch is None or args.onnx is None or args.output is None:
        parser.error("Comparison mode requires --pytorch, --onnx, and --output.")
    result = compare_detection_sets(
        json.loads(args.pytorch.read_text(encoding="utf-8")),
        json.loads(args.onnx.read_text(encoding="utf-8")),
        confidence_tolerance=args.confidence_tolerance,
        box_tolerance_px=args.box_tolerance_px,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
