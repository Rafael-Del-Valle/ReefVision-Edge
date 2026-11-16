from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixtures.build_gpu_acceptance_model import (  # noqa: E402
    build_gpu_acceptance_model,
)
from tests.fixtures.build_video import build_mjpeg_video  # noqa: E402

from reefvision_edge.domain import Detection  # noqa: E402
from reefvision_edge.gpu_acceptance import (  # noqa: E402
    compare_predictions,
    decode_video,
    require_cuda_profile,
    require_sustained_duration,
    verify_overlay_manifest,
)


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.content


def _metadata() -> dict[str, object]:
    return {
        "name": "RTX 3090 acceptance detector",
        "format": "ultralytics_yolo_detect_v1",
        "input_width": 64,
        "input_height": 64,
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


def _upload_model(client: httpx.Client, path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return _request(
            client,
            "POST",
            "/api/models",
            files={"model": (path.name, stream, "application/octet-stream")},
            data={"metadata_json": json.dumps(_metadata())},
        )


def _upload_video(client: httpx.Client, path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return _request(
            client,
            "POST",
            "/api/sources/video",
            files={"video": (path.name, stream, "video/x-msvideo")},
        )


def _wait_terminal(
    client: httpx.Client, session_id: str, timeout: float = 120
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = _request(client, "GET", f"/api/sessions/{session_id}")
        if session["status"] not in {"queued", "running"}:
            return session
        time.sleep(0.1)
    raise AssertionError(f"Session {session_id} did not finish within {timeout} seconds.")


def _run_session(
    client: httpx.Client,
    model_id: str,
    source_id: str,
    *,
    compute: str,
    record_data: bool,
    record_monitor: bool,
    preview_overlays: dict[str, bool] | None = None,
    recording_overlays: dict[str, bool] | None = None,
    duration_seconds: float | None = None,
) -> tuple[dict[str, Any], float]:
    default_overlays = {
        "timer": True,
        "trails": False,
        "boxes": True,
        "masks": False,
        "labels": True,
    }
    created = _request(
        client,
        "POST",
        "/api/sessions",
        json={
            "model_id": model_id,
            "source_id": source_id,
            "sample_every": 1,
            "compute_device": compute,
            "record_data": record_data,
            "record_monitor": record_monitor,
            "preview_overlays": preview_overlays or default_overlays,
            "recording_overlays": recording_overlays or default_overlays,
            "enable_profiling": True,
        },
    )
    started = time.monotonic()
    _request(client, "POST", f"/api/sessions/{created['id']}/start")
    if duration_seconds is None:
        session = _wait_terminal(client, created["id"])
    else:
        deadline = started + duration_seconds
        while time.monotonic() < deadline:
            session = _request(client, "GET", f"/api/sessions/{created['id']}")
            if session["status"] not in {"queued", "running"}:
                raise AssertionError(
                    "Sustained source ended before the acceptance duration: "
                    f"{session['status']} {session.get('error_message')}"
                )
            time.sleep(0.2)
        _request(client, "POST", f"/api/sessions/{created['id']}/stop")
        session = _wait_terminal(client, created["id"], timeout=30)
    elapsed = time.monotonic() - started
    if session["status"] not in {"completed", "canceled"}:
        raise AssertionError(
            f"Session {created['id']} failed: {session['status']} "
            f"{session.get('error_message')}"
        )
    return session, elapsed


def _export(client: httpx.Client, session_id: str, target: Path) -> dict[str, Any]:
    target.write_bytes(
        _request(client, "GET", f"/api/sessions/{session_id}/export")
    )
    with zipfile.ZipFile(target) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read("manifest.json"))
        frame_rows = list(
            csv.DictReader(io.StringIO(bundle.read("frames.csv").decode("utf-8")))
        )
        detection_rows = list(
            csv.DictReader(
                io.StringIO(bundle.read("detections.csv").decode("utf-8"))
            )
        )
        video_path: Path | None = None
        if "monitor.mp4" in names:
            video_path = target.with_suffix(".monitor.mp4")
            video_path.write_bytes(bundle.read("monitor.mp4"))
    return {
        "names": names,
        "manifest": manifest,
        "frame_rows": frame_rows,
        "detection_rows": detection_rows,
        "video_path": video_path,
    }


def _detections(rows: list[dict[str, str]]) -> tuple[Detection, ...]:
    return tuple(
        Detection(
            class_id=int(row["class_id"]),
            class_name=row["class_name"],
            score=float(row["score"]),
            box=tuple(
                float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")
            ),
        )
        for row in rows
    )


def _driver_info() -> dict[str, str]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = build_gpu_acceptance_model(output / "gpu-acceptance.onnx")
    short_video = build_mjpeg_video(
        output / "parity.avi", frame_count=12, width=160, height=90, fps=10
    )
    sustained_video = build_mjpeg_video(
        output / "sustained.avi", frame_count=20_000, width=160, height=90, fps=10
    )
    recording_overlays = {
        "timer": True,
        "trails": False,
        "boxes": False,
        "masks": False,
        "labels": True,
    }
    preview_overlays = {
        "timer": False,
        "trails": False,
        "boxes": True,
        "masks": False,
        "labels": True,
    }

    with httpx.Client(base_url=arguments.base_url, timeout=120) as client:
        model = _upload_model(client, model_path)
        short_source = _upload_video(client, short_video)
        cpu, _ = _run_session(
            client,
            model["id"],
            short_source["id"],
            compute="cpu",
            record_data=True,
            record_monitor=False,
        )
        gpu, _ = _run_session(
            client,
            model["id"],
            short_source["id"],
            compute="nvidia",
            record_data=True,
            record_monitor=True,
            preview_overlays=preview_overlays,
            recording_overlays=recording_overlays,
        )
        video_only, _ = _run_session(
            client,
            model["id"],
            short_source["id"],
            compute="nvidia",
            record_data=False,
            record_monitor=True,
        )
        cpu_export = _export(client, cpu["id"], output / "cpu.zip")
        gpu_export = _export(client, gpu["id"], output / "gpu.zip")
        video_export = _export(client, video_only["id"], output / "video-only.zip")

        parity = compare_predictions(
            _detections(cpu_export["detection_rows"]),
            _detections(gpu_export["detection_rows"]),
        )
        cuda_ops, cpu_ops = require_cuda_profile(
            gpu_export["manifest"]["compute"]["operator_profile"]
        )
        if gpu["provider"] != "CUDAExecutionProvider":
            raise AssertionError(f"Explicit NVIDIA session used {gpu['provider']}.")
        if "RTX 3090" not in (gpu.get("actual_device") or ""):
            raise AssertionError(f"Unexpected active GPU: {gpu.get('actual_device')}")
        if cpu["provider"] != "CPUExecutionProvider":
            raise AssertionError(f"CPU session used {cpu['provider']}.")
        if "monitor.mp4" in cpu_export["names"]:
            raise AssertionError("Data-only session unexpectedly contains monitor.mp4.")
        if video_export["frame_rows"] or video_export["detection_rows"]:
            raise AssertionError("Video-only session unexpectedly persisted data rows.")
        if "monitor.mp4" not in video_export["names"]:
            raise AssertionError("Video-only session did not contain monitor.mp4.")
        video_only_decode = decode_video(video_export["video_path"])
        gpu_decode = decode_video(gpu_export["video_path"])
        verify_overlay_manifest(gpu_export["manifest"], recording_overlays)

        sustained_source = _upload_video(client, sustained_video)
        sustained, sustained_elapsed = _run_session(
            client,
            model["id"],
            sustained_source["id"],
            compute="nvidia",
            record_data=True,
            record_monitor=True,
            preview_overlays=preview_overlays,
            recording_overlays=recording_overlays,
            duration_seconds=arguments.duration_seconds,
        )
        require_sustained_duration(
            sustained_elapsed, minimum_seconds=arguments.duration_seconds
        )
        sustained_export = _export(
            client, sustained["id"], output / "sustained.zip"
        )
        sustained_cuda_ops, sustained_cpu_ops = require_cuda_profile(
            sustained_export["manifest"]["compute"]["operator_profile"]
        )
        verify_overlay_manifest(sustained_export["manifest"], recording_overlays)
        sustained_decode = decode_video(sustained_export["video_path"])
        if sustained_decode[0] != sustained["frame_count"]:
            raise AssertionError(
                "Saved sustained video frame count differs from the session: "
                f"{sustained_decode[0]} != {sustained['frame_count']}"
            )
        required_frame_fields = {
            "timestamp_ms",
            "inference_status",
            "latency_ms",
            "source_width",
            "source_height",
        }
        if not sustained_export["frame_rows"] or not required_frame_fields <= set(
            sustained_export["frame_rows"][0]
        ):
            raise AssertionError("Sustained frames.csv is missing required data fields.")
        if not sustained_export["detection_rows"]:
            raise AssertionError("Sustained detections.csv is empty.")
        if sustained.get("peak_vram_mb") is None:
            raise AssertionError("Sustained GPU session did not record peak VRAM.")
        capture_devices = _request(client, "GET", "/api/capture-devices")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "tested_commit": commit,
        "bundle_path": str(arguments.bundle_path.resolve()),
        "driver": _driver_info(),
        "gpu_checks": {
            "explicit_cuda_provider": True,
            "actual_device": gpu["actual_device"],
            "cuda_operator_events": cuda_ops,
            "cpu_operator_events": cpu_ops,
            "cuda_operators": gpu_export["manifest"]["compute"]["operator_profile"][
                "cuda_operators"
            ],
            "cpu_operators": gpu_export["manifest"]["compute"]["operator_profile"][
                "cpu_operators"
            ],
        },
        "parity": {
            "class_ids_match": parity.class_ids_match,
            "max_score_delta": parity.max_score_delta,
            "max_geometry_delta_px": parity.max_geometry_delta,
            "score_tolerance": 1e-4,
            "geometry_tolerance_px": 0.5,
        },
        "recording_modes": {
            "data_only": True,
            "video_only": True,
            "both": True,
            "video_only_decode": video_only_decode,
            "both_decode": gpu_decode,
        },
        "sustained": {
            "elapsed_seconds": sustained_elapsed,
            "status": sustained["status"],
            "input_fps": sustained["input_fps"],
            "inference_fps": sustained["inference_fps"],
            "median_latency_ms": sustained["median_latency_ms"],
            "p95_latency_ms": sustained["p95_latency_ms"],
            "dropped_frames": sustained["dropped_frames"],
            "peak_vram_mb": sustained["peak_vram_mb"],
            "vram_measurement": "nvidia-smi memory.used total fallback under WDDM",
            "frame_count": sustained["frame_count"],
            "detection_count": sustained["detection_count"],
            "video_decode": sustained_decode,
            "cuda_operator_events": sustained_cuda_ops,
            "cpu_operator_events": sustained_cpu_ops,
            "frames_csv_rows": len(sustained_export["frame_rows"]),
            "detections_csv_rows": len(sustained_export["detection_rows"]),
            "export": str((output / "sustained.zip").resolve()),
        },
        "open_checks": {
            "physical_camera": (
                "passed" if capture_devices else "open: no Windows capture camera connected"
            ),
            "nvenc": "open: software MP4 encoder is the accepted baseline",
            "jetson": "open: separate JetPack runtime profile",
            "dji_mapping": "open: retained on broader plan",
        },
    }
    report_path = output / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
