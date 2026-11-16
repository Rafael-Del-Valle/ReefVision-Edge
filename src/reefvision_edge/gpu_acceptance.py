from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from reefvision_edge.domain import Detection


@dataclass(frozen=True)
class ParitySummary:
    class_ids_match: bool
    max_score_delta: float
    max_geometry_delta: float


def compare_predictions(
    cpu: Sequence[Detection],
    gpu: Sequence[Detection],
    *,
    score_tolerance: float = 1e-4,
    geometry_tolerance: float = 0.5,
) -> ParitySummary:
    cpu_classes = tuple(item.class_id for item in cpu)
    gpu_classes = tuple(item.class_id for item in gpu)
    if cpu_classes != gpu_classes:
        raise AssertionError(
            f"CPU/GPU class IDs differ: CPU {cpu_classes}, GPU {gpu_classes}."
        )
    score_deltas = [
        abs(cpu_item.score - gpu_item.score)
        for cpu_item, gpu_item in zip(cpu, gpu, strict=True)
    ]
    geometry_deltas = [
        abs(cpu_value - gpu_value)
        for cpu_item, gpu_item in zip(cpu, gpu, strict=True)
        for cpu_value, gpu_value in zip(cpu_item.box, gpu_item.box, strict=True)
    ]
    max_score_delta = max(score_deltas, default=0.0)
    max_geometry_delta = max(geometry_deltas, default=0.0)
    if max_score_delta > score_tolerance:
        raise AssertionError(
            "CPU/GPU confidence delta "
            f"{max_score_delta:.8f} exceeds {score_tolerance:.8f}."
        )
    if max_geometry_delta > geometry_tolerance:
        raise AssertionError(
            "CPU/GPU overlay geometry delta "
            f"{max_geometry_delta:.4f}px exceeds {geometry_tolerance:.4f}px."
        )
    return ParitySummary(True, max_score_delta, max_geometry_delta)


def require_cuda_profile(profile: Mapping[str, object]) -> tuple[int, int]:
    counts_value = profile.get("provider_counts")
    counts = counts_value if isinstance(counts_value, Mapping) else {}
    cuda_count = int(counts.get("CUDAExecutionProvider", 0))
    cpu_count = int(counts.get("CPUExecutionProvider", 0))
    if cuda_count < 1:
        raise AssertionError("ONNX Runtime profiling recorded no CUDA-executed operators.")
    return cuda_count, cpu_count


def require_sustained_duration(
    elapsed_seconds: float, *, minimum_seconds: float = 60.0
) -> None:
    if elapsed_seconds < minimum_seconds:
        raise AssertionError(
            f"Sustained GPU session must run at least {minimum_seconds:.1f} seconds; "
            f"observed {elapsed_seconds:.2f}."
        )


def verify_overlay_manifest(
    manifest: Mapping[str, object], expected: Mapping[str, bool]
) -> None:
    recording_value = manifest.get("recording")
    recording = recording_value if isinstance(recording_value, Mapping) else {}
    actual = recording.get("recording_overlays")
    if actual != expected:
        raise AssertionError(
            f"Saved-video overlay settings differ: expected {dict(expected)}, got {actual}."
        )


def decode_video(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"Saved monitor video could not be opened: {path}")
    frames = 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while True:
            readable, _ = capture.read()
            if not readable:
                break
            frames += 1
    finally:
        capture.release()
    if frames < 1 or width < 1 or height < 1:
        raise AssertionError(f"Saved monitor video contains no decodable frames: {path}")
    return frames, width, height
