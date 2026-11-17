from __future__ import annotations

import os
from pathlib import Path

import pytest
from scripts.acceptance_windows_model_import import compare_detection_sets


def test_parity_comparison_matches_by_class_confidence_and_geometry() -> None:
    pytorch = [
        {"class_id": 0, "confidence": 0.91, "box": [10.0, 20.0, 110.0, 120.0]},
        {"class_id": 1, "confidence": 0.73, "box": [200.0, 40.0, 280.0, 130.0]},
    ]
    onnx = [
        {"class_id": 1, "confidence": 0.729, "box": [200.2, 39.9, 280.1, 130.2]},
        {"class_id": 0, "confidence": 0.912, "box": [9.9, 20.1, 110.2, 119.8]},
    ]

    result = compare_detection_sets(
        pytorch,
        onnx,
        confidence_tolerance=0.01,
        box_tolerance_px=0.5,
    )

    assert result["passed"] is True
    assert result["class_counts_match"] is True
    assert result["max_confidence_delta"] == pytest.approx(0.002)
    assert result["max_box_delta_px"] == pytest.approx(0.2)


def test_parity_comparison_reports_class_count_mismatch() -> None:
    result = compare_detection_sets(
        [{"class_id": 0, "confidence": 0.9, "box": [0, 0, 10, 10]}],
        [{"class_id": 1, "confidence": 0.9, "box": [0, 0, 10, 10]}],
        confidence_tolerance=0.01,
        box_tolerance_px=0.5,
    )

    assert result["passed"] is False
    assert result["class_counts_match"] is False


@pytest.mark.skipif(
    os.environ.get("REEFVISION_EDGE_RUN_MODEL_IMPORT_ACCEPTANCE") != "1",
    reason="set REEFVISION_EDGE_RUN_MODEL_IMPORT_ACCEPTANCE=1 for packaged RTX acceptance",
)
def test_packaged_model_import_acceptance_report_exists() -> None:
    report = Path(os.environ["REEFVISION_EDGE_MODEL_IMPORT_REPORT"])
    assert report.is_file()
    value = __import__("json").loads(report.read_text(encoding="utf-8"))
    assert value["converter"]["status"] == "passed"
    assert value["cuda"]["operators"] > 0
    assert value["full_frame"]["status"] == "completed"
    assert value["thesis_tiling"]["status"] == "completed"
