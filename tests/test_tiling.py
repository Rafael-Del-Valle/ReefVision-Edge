from __future__ import annotations

import numpy as np

from reefvision_edge.domain import Detection, InferenceResult
from reefvision_edge.tiling import merge_tiled_detections, run_tiled_inference, tile_windows


def test_tile_windows_cover_small_exact_and_irregular_frames() -> None:
    assert tile_windows(1280, 720) == ((0, 0, 1280, 720),)
    assert tile_windows(1920, 1344) == ((0, 0, 1920, 1344),)

    windows = tile_windows(4000, 2200)

    assert windows[0] == (0, 0, 1920, 1344)
    assert windows[-1] == (2080, 856, 1920, 1344)
    covered = np.zeros((2200, 4000), dtype=np.uint8)
    for x, y, width, height in windows:
        covered[y : y + height, x : x + width] = 1
    assert covered.all()


def test_merge_removes_same_class_duplicates_but_keeps_other_classes() -> None:
    detections = [
        Detection(0, "open", 0.9, (100, 100, 200, 200)),
        Detection(0, "open", 0.8, (105, 105, 205, 205)),
        Detection(1, "closed", 0.7, (105, 105, 205, 205)),
    ]

    merged = merge_tiled_detections(detections, 0.45)

    assert [(item.class_id, item.score) for item in merged] == [(0, 0.9), (1, 0.7)]


def test_tiled_inference_restores_full_frame_coordinates_and_aggregates_latency() -> None:
    frame = np.zeros((1344, 3000, 3), dtype=np.uint8)

    def predict(tile: np.ndarray) -> InferenceResult:
        return InferenceResult(
            detections=(Detection(0, "open", 0.8, (10, 20, 110, 120)),),
            latency_ms=3.5,
            provider="CUDAExecutionProvider",
        )

    result = run_tiled_inference(frame, predict, iou_threshold=0.45)

    assert result.latency_ms == 7.0
    assert result.provider == "CUDAExecutionProvider"
    assert [item.box for item in result.detections] == [
        (10, 20, 110, 120),
        (1090, 20, 1190, 120),
    ]
