from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest
from tests.fixtures.build_gpu_acceptance_model import build_gpu_acceptance_model

from reefvision_edge.domain import Detection
from reefvision_edge.gpu_acceptance import (
    compare_predictions,
    decode_video,
    require_cuda_profile,
    require_sustained_duration,
    verify_overlay_manifest,
)


def test_cpu_gpu_parity_requires_exact_classes_and_bounded_scores_geometry() -> None:
    cpu = (
        Detection(0, "coral", 0.90000, (10.0, 20.0, 30.0, 40.0)),
        Detection(1, "fish", 0.80000, (50.0, 60.0, 70.0, 80.0)),
    )
    gpu = (
        Detection(0, "coral", 0.90008, (10.4, 19.6, 30.2, 40.1)),
        Detection(1, "fish", 0.79995, (49.8, 60.1, 70.0, 80.3)),
    )

    summary = compare_predictions(cpu, gpu)

    assert summary.class_ids_match is True
    assert summary.max_score_delta == pytest.approx(0.00008)
    assert summary.max_geometry_delta == pytest.approx(0.4)

    with pytest.raises(AssertionError, match="class IDs"):
        compare_predictions(cpu, (gpu[1], gpu[0]))
    with pytest.raises(AssertionError, match="confidence"):
        compare_predictions(cpu, (Detection(0, "coral", 0.901, cpu[0].box), gpu[1]))
    with pytest.raises(AssertionError, match="geometry"):
        compare_predictions(cpu, (Detection(0, "coral", cpu[0].score, (10.6, 20, 30, 40)), gpu[1]))


def test_cuda_profile_and_sustained_duration_are_hard_acceptance_gates() -> None:
    profile = {
        "provider_counts": {"CUDAExecutionProvider": 7, "CPUExecutionProvider": 2},
        "cuda_operators": ["Add", "MatMul", "ReduceMean"],
        "cpu_operators": ["Shape"],
        "cuda_executed": True,
    }

    assert require_cuda_profile(profile) == (7, 2)
    require_sustained_duration(60.0)

    with pytest.raises(AssertionError, match="CUDA-executed"):
        require_cuda_profile({"provider_counts": {"CPUExecutionProvider": 9}})
    with pytest.raises(AssertionError, match=r"60\.0 seconds"):
        require_sustained_duration(59.99)


def test_overlay_manifest_and_saved_video_are_reopened(tmp_path: Path) -> None:
    requested = {
        "timer": True,
        "trails": False,
        "boxes": False,
        "masks": False,
        "labels": True,
    }
    verify_overlay_manifest(
        {"recording": {"recording_overlays": requested}}, requested
    )
    with pytest.raises(AssertionError, match="overlay settings"):
        verify_overlay_manifest(
            {"recording": {"recording_overlays": {**requested, "boxes": True}}},
            requested,
        )

    path = tmp_path / "recorded.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
    )
    assert writer.isOpened()
    for value in (0, 80, 160):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    assert decode_video(path) == (3, 64, 48)


def test_gpu_acceptance_model_is_input_dependent_and_detector_shaped(
    tmp_path: Path,
) -> None:
    path = build_gpu_acceptance_model(tmp_path / "gpu-acceptance.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    black = session.run(None, {name: np.zeros((1, 3, 64, 64), dtype=np.float32)})[0]
    white = session.run(None, {name: np.ones((1, 3, 64, 64), dtype=np.float32)})[0]

    assert black.shape == (1, 6, 2)
    assert white.shape == (1, 6, 2)
    assert black[0, 4, 0] == pytest.approx(0.9)
    assert white[0, 4, 0] == pytest.approx(0.91)
    assert np.array_equal(black[:, :4], white[:, :4])
