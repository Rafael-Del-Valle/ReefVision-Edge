from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from reefvision_edge.domain import Detection, InferenceResult, RegisteredModel
from reefvision_edge.geometry import letterbox, restore_box


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
    def __init__(self, model: RegisteredModel) -> None:
        self.model = model
        self.session = ort.InferenceSession(
            str(model.path), providers=[model.provider]
        )
        self.input_name = self.session.get_inputs()[0].name

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
        output = self.session.run(None, {self.input_name: input_tensor})[0][0].T
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
            provider=self.session.get_providers()[0],
        )
