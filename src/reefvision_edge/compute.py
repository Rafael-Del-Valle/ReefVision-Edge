from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import onnxruntime as ort


class ComputeChoice(StrEnum):
    AUTO = "auto"
    NVIDIA_GPU = "nvidia"
    CPU = "cpu"


class ComputeDeviceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComputeResolution:
    requested: ComputeChoice
    providers: tuple[str, ...]
    active_provider: str
    actual_device: str
    explanation: str
    fell_back: bool


@dataclass(frozen=True)
class OperatorProfile:
    provider_counts: dict[str, int]
    cuda_operators: tuple[str, ...]
    cpu_operators: tuple[str, ...]

    @property
    def cuda_executed(self) -> bool:
        return self.provider_counts.get("CUDAExecutionProvider", 0) > 0


def preload_nvidia_runtime() -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(cuda=True, cudnn=True, msvc=True, directory="")
    except TypeError:
        preload(cuda=True, cudnn=True, msvc=True)


def detect_nvidia_device() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return "NVIDIA GPU"
    name = completed.stdout.splitlines()[0].strip() if completed.stdout.strip() else ""
    return name or "NVIDIA GPU"


def _cpu_resolution(
    requested: ComputeChoice, explanation: str, *, fell_back: bool
) -> ComputeResolution:
    available = ort.get_available_providers()
    if "CPUExecutionProvider" not in available:
        raise ComputeDeviceError(
            "CPUExecutionProvider is not available in this ONNX Runtime installation."
        )
    return ComputeResolution(
        requested=requested,
        providers=("CPUExecutionProvider",),
        active_provider="CPUExecutionProvider",
        actual_device="CPU",
        explanation=explanation,
        fell_back=fell_back,
    )


def cpu_fallback(requested: ComputeChoice, reason: str) -> ComputeResolution:
    return _cpu_resolution(
        requested,
        f"NVIDIA GPU initialization failed; using CPU. {reason}",
        fell_back=True,
    )


def resolve_compute(choice: ComputeChoice | str) -> ComputeResolution:
    requested = ComputeChoice(choice)
    if requested is ComputeChoice.CPU:
        return _cpu_resolution(
            requested, "CPU was selected explicitly.", fell_back=False
        )

    try:
        preload_nvidia_runtime()
    except Exception as exc:
        if requested is ComputeChoice.NVIDIA_GPU:
            raise ComputeDeviceError(
                "NVIDIA GPU mode could not load the bundled CUDA/cuDNN runtime: "
                f"{exc}"
            ) from exc
        return cpu_fallback(requested, f"CUDA/cuDNN DLL preload failed: {exc}")

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        reason = (
            "CUDAExecutionProvider is not available. Install or rebuild with the "
            "Windows CUDA runtime profile and verify the NVIDIA driver."
        )
        if requested is ComputeChoice.NVIDIA_GPU:
            raise ComputeDeviceError(
                "NVIDIA GPU mode requires CUDAExecutionProvider. " + reason
            )
        return _cpu_resolution(requested, reason + " Using CPU.", fell_back=True)

    providers = ["CUDAExecutionProvider"]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return ComputeResolution(
        requested=requested,
        providers=tuple(providers),
        active_provider="CUDAExecutionProvider",
        actual_device=detect_nvidia_device(),
        explanation=(
            "NVIDIA GPU was selected explicitly."
            if requested is ComputeChoice.NVIDIA_GPU
            else "Auto selected the available NVIDIA CUDA provider."
        ),
        fell_back=False,
    )


def parse_ort_profile(path: Path) -> OperatorProfile:
    events: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("ONNX Runtime profile must contain a JSON event list.")
    provider_counts: Counter[str] = Counter()
    operators: dict[str, set[str]] = {
        "CUDAExecutionProvider": set(),
        "CPUExecutionProvider": set(),
    }
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        arguments = event.get("args")
        if not isinstance(arguments, dict):
            continue
        provider = arguments.get("provider")
        if not isinstance(provider, str) or not provider:
            continue
        provider_counts[provider] += 1
        operator = arguments.get("op_name")
        if isinstance(operator, str) and operator:
            operators.setdefault(provider, set()).add(operator)
    return OperatorProfile(
        provider_counts=dict(provider_counts),
        cuda_operators=tuple(sorted(operators["CUDAExecutionProvider"])),
        cpu_operators=tuple(sorted(operators["CPUExecutionProvider"])),
    )
