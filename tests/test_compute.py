from __future__ import annotations

import json
from pathlib import Path

import pytest

from reefvision_edge.compute import (
    ComputeChoice,
    ComputeDeviceError,
    parse_ort_profile,
    resolve_compute,
)


def test_cpu_choice_uses_only_cpu_and_reports_actual_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reefvision_edge.compute.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    resolution = resolve_compute(ComputeChoice.CPU)

    assert resolution.requested is ComputeChoice.CPU
    assert resolution.providers == ("CPUExecutionProvider",)
    assert resolution.active_provider == "CPUExecutionProvider"
    assert resolution.actual_device == "CPU"
    assert resolution.fell_back is False
    assert resolution.explanation == "CPU was selected explicitly."


def test_auto_falls_back_with_visible_reason_when_cuda_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reefvision_edge.compute.ort.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    resolution = resolve_compute(ComputeChoice.AUTO)

    assert resolution.providers == ("CPUExecutionProvider",)
    assert resolution.fell_back is True
    assert resolution.actual_device == "CPU"
    assert "CUDAExecutionProvider is not available" in resolution.explanation


def test_explicit_nvidia_fails_actionably_when_cuda_provider_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "reefvision_edge.compute.ort.get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    with pytest.raises(ComputeDeviceError, match="NVIDIA GPU mode requires CUDAExecutionProvider"):
        resolve_compute(ComputeChoice.NVIDIA_GPU)


def test_nvidia_choice_configures_cuda_then_cpu_and_names_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reefvision_edge.compute.preload_nvidia_runtime", lambda: None)
    monkeypatch.setattr(
        "reefvision_edge.compute.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        "reefvision_edge.compute.detect_nvidia_device",
        lambda: "NVIDIA GeForce RTX 3090",
    )

    resolution = resolve_compute(ComputeChoice.NVIDIA_GPU)

    assert resolution.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert resolution.active_provider == "CUDAExecutionProvider"
    assert resolution.actual_device == "NVIDIA GeForce RTX 3090"
    assert resolution.fell_back is False


def test_profile_parser_reports_cuda_and_cpu_operator_assignments(tmp_path: Path) -> None:
    profile = tmp_path / "ort-profile.json"
    profile.write_text(
        json.dumps(
            [
                {
                    "cat": "Node",
                    "name": "Conv_0_kernel_time",
                    "args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"},
                },
                {
                    "cat": "Node",
                    "name": "Add_1_kernel_time",
                    "args": {"provider": "CUDAExecutionProvider", "op_name": "Add"},
                },
                {
                    "cat": "Node",
                    "name": "Shape_2_kernel_time",
                    "args": {"provider": "CPUExecutionProvider", "op_name": "Shape"},
                },
                {"cat": "Session", "name": "model_run", "args": {}},
            ]
        ),
        encoding="utf-8",
    )

    summary = parse_ort_profile(profile)

    assert summary.provider_counts == {
        "CUDAExecutionProvider": 2,
        "CPUExecutionProvider": 1,
    }
    assert summary.cuda_operators == ("Add", "Conv")
    assert summary.cpu_operators == ("Shape",)
    assert summary.cuda_executed is True


def test_profile_parser_rejects_malformed_profile(tmp_path: Path) -> None:
    profile = tmp_path / "bad.json"
    profile.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON event list"):
        parse_ort_profile(profile)
