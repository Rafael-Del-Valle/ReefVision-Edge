from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from reefvision_edge.compute import (
    ComputeChoice,
    ComputeDeviceError,
    ComputeResolution,
    OperatorProfile,
    cpu_fallback,
    parse_ort_profile,
    resolve_compute,
)
from reefvision_edge.domain import Detection, InferenceResult, RegisteredModel
from reefvision_edge.geometry import letterbox, restore_box


class _AssignedSubgraph(Protocol):
    ep_name: str

    def get_nodes(self) -> list[object]: ...


class _AssignmentSession(Protocol):
    def get_provider_graph_assignment_info(self) -> list[_AssignedSubgraph]: ...

    def get_providers(self) -> list[str]: ...


def provider_assignment_summary(session: _AssignmentSession) -> str:
    try:
        assignments = session.get_provider_graph_assignment_info()
    except (AttributeError, RuntimeError):
        providers = session.get_providers()
        return (
            f"assignment unavailable; configured {' -> '.join(providers)}"
            if providers
            else "assignment unavailable"
        )
    counts: Counter[str] = Counter()
    for subgraph in assignments:
        counts[subgraph.ep_name] += len(subgraph.get_nodes())
    if not counts:
        providers = session.get_providers()
        return providers[0] if providers else "no executable graph nodes"
    ordered = [provider for provider in session.get_providers() if counts[provider]]
    ordered.extend(provider for provider in counts if provider not in ordered)
    accelerated = any(provider != "CPUExecutionProvider" for provider in ordered)
    parts = []
    for provider in ordered:
        fallback = " fallback" if provider == "CPUExecutionProvider" and accelerated else ""
        parts.append(f"{provider}{fallback} ({counts[provider]} nodes)")
    return " -> ".join(parts)


def _iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def non_maximum_suppression(
    detections: list[Detection], threshold: float
) -> tuple[Detection, ...]:
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    while ordered:
        selected = ordered.pop(0)
        kept.append(selected)
        ordered = [
            candidate
            for candidate in ordered
            if candidate.class_id != selected.class_id
            or _iou(candidate.box, selected.box) <= threshold
        ]
    return tuple(kept)


class YoloOnnxDetector:
    def __init__(
        self,
        model: RegisteredModel,
        compute_choice: ComputeChoice | str = ComputeChoice.AUTO,
        enable_profiling: bool = False,
    ) -> None:
        self.model = model
        self.compute = resolve_compute(compute_choice)
        self._requested_profiling = enable_profiling
        self._cuda_validation_pending = (
            self.compute.active_provider == "CUDAExecutionProvider"
        )
        options = ort.SessionOptions()
        options.enable_profiling = enable_profiling or self._cuda_validation_pending
        if hasattr(ort.InferenceSession, "get_provider_graph_assignment_info"):
            options.add_session_config_entry(
                "session.record_ep_graph_assignment_info", "1"
            )
        try:
            self.session = ort.InferenceSession(
                str(model.path),
                sess_options=options,
                providers=list(self.compute.providers),
            )
        except Exception as exc:
            if self.compute.requested is not ComputeChoice.AUTO:
                raise ComputeDeviceError(
                    f"{self.compute.requested.value} inference could not initialize: {exc}"
                ) from exc
            self.compute = cpu_fallback(self.compute.requested, str(exc))
            try:
                self.session = ort.InferenceSession(
                    str(model.path),
                    sess_options=options,
                    providers=list(self.compute.providers),
                )
            except Exception as cpu_exc:
                raise ComputeDeviceError(
                    f"CPU fallback could not initialize ONNX Runtime: {cpu_exc}"
                ) from cpu_exc
        self.input_name = self.session.get_inputs()[0].name
        self.provider = provider_assignment_summary(self.session)
        self.profile_path: Path | None = None
        self.profile: OperatorProfile | None = None

    def _fall_back_to_cpu_after_validation(self, reason: str) -> None:
        if self.profile_path is not None:
            self.profile_path.unlink(missing_ok=True)
        self.compute = cpu_fallback(self.compute.requested, reason)
        options = ort.SessionOptions()
        options.enable_profiling = self._requested_profiling
        self.session = ort.InferenceSession(
            str(self.model.path),
            sess_options=options,
            providers=list(self.compute.providers),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.profile_path = None
        self.profile = None
        self.provider = provider_assignment_summary(self.session)

    @property
    def resolution(self) -> ComputeResolution:
        return self.compute

    def finish_profile(self) -> OperatorProfile | None:
        if not self.session.get_session_options().enable_profiling:
            return None
        if self.profile is None:
            self.profile_path = Path(self.session.end_profiling())
            self.profile = parse_ort_profile(self.profile_path)
        return self.profile

    def predict(self, frame: NDArray[np.uint8]) -> InferenceResult:
        started = time.perf_counter()
        metadata = self.model.metadata
        transformed, mapping = letterbox(
            frame, width=metadata.input_width, height=metadata.input_height
        )
        rgb = transformed[:, :, ::-1]
        input_tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        output_value = self.session.run(None, {self.input_name: input_tensor})[0]
        if self._cuda_validation_pending:
            profile = self.finish_profile()
            self._cuda_validation_pending = False
            if profile is None or not profile.cuda_executed:
                reason = (
                    "CUDAExecutionProvider initialized but executed zero operators for "
                    "this model. Check model/operator CUDA support and the bundled runtime."
                )
                if self.compute.requested is ComputeChoice.NVIDIA_GPU:
                    raise ComputeDeviceError("NVIDIA GPU mode " + reason)
                self._fall_back_to_cpu_after_validation(reason)
                output_value = self.session.run(
                    None, {self.input_name: input_tensor}
                )[0]
        output = output_value[0].T
        self.provider = provider_assignment_summary(self.session)
        detections: list[Detection] = []
        for candidate in output:
            scores = candidate[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score < metadata.confidence_threshold:
                continue
            center_x, center_y, width, height = (float(value) for value in candidate[:4])
            box = restore_box(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
                mapping,
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=metadata.classes[class_id],
                    score=score,
                    box=box,
                )
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return InferenceResult(
            detections=non_maximum_suppression(detections, metadata.iou_threshold),
            latency_ms=latency_ms,
            provider=self.provider,
        )
