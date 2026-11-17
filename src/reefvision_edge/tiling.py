from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from reefvision_edge.domain import Detection, InferenceResult

THESIS_TILE_WIDTH = 1920
THESIS_TILE_HEIGHT = 1344
THESIS_TILE_OVERLAP = 0.4


class TileWindow(NamedTuple):
    x: int
    y: int
    width: int
    height: int


def _anchors(length: int, tile_length: int, overlap: float) -> tuple[int, ...]:
    if length <= tile_length:
        return (0,)
    stride = max(1, round(tile_length * (1 - overlap)))
    anchors = list(range(0, length - tile_length + 1, stride))
    final = length - tile_length
    if anchors[-1] != final:
        anchors.append(final)
    return tuple(anchors)


def tile_windows(
    frame_width: int,
    frame_height: int,
    tile_width: int = THESIS_TILE_WIDTH,
    tile_height: int = THESIS_TILE_HEIGHT,
    overlap: float = THESIS_TILE_OVERLAP,
) -> tuple[TileWindow, ...]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile dimensions must be positive.")
    if not 0 <= overlap < 1:
        raise ValueError("Tile overlap must be in [0, 1).")
    width = min(frame_width, tile_width)
    height = min(frame_height, tile_height)
    return tuple(
        TileWindow(x, y, width, height)
        for y in _anchors(frame_height, height, overlap)
        for x in _anchors(frame_width, width, overlap)
    )


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
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


def merge_tiled_detections(
    detections: Sequence[Detection], iou_threshold: float
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
            or _iou(candidate.box, selected.box) <= iou_threshold
        ]
    return tuple(kept)


def run_tiled_inference(
    frame: NDArray[np.uint8],
    predict_tile: Callable[[NDArray[np.uint8]], InferenceResult],
    *,
    iou_threshold: float,
) -> InferenceResult:
    detections: list[Detection] = []
    latency_ms = 0.0
    provider = ""
    height, width = frame.shape[:2]
    for window in tile_windows(width, height):
        tile = frame[
            window.y : window.y + window.height,
            window.x : window.x + window.width,
        ]
        result = predict_tile(tile)
        latency_ms += result.latency_ms
        provider = result.provider
        detections.extend(
            Detection(
                class_id=detection.class_id,
                class_name=detection.class_name,
                score=detection.score,
                box=(
                    detection.box[0] + window.x,
                    detection.box[1] + window.y,
                    detection.box[2] + window.x,
                    detection.box[3] + window.y,
                ),
            )
            for detection in result.detections
        )
    return InferenceResult(
        detections=merge_tiled_detections(detections, iou_threshold),
        latency_ms=latency_ms,
        provider=provider,
    )
