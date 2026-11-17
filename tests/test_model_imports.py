from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

from reefvision_edge.db import Database
from reefvision_edge.model_imports import ModelImportError, ModelImportService
from reefvision_edge.model_registry import ModelRegistry

FAKE_CONVERTER = Path(__file__).parent / "fixtures" / "fake_converter.py"


def service(tmp_path: Path) -> ModelImportService:
    database = Database(tmp_path / "edge.sqlite3")
    database.initialize()
    registry = ModelRegistry(database, tmp_path / "models")
    return ModelImportService(
        registry,
        tmp_path / "model-imports",
        converter_command=(sys.executable, str(FAKE_CONVERTER)),
    )


def wait_for_status(
    imports: ModelImportService, import_id: str, expected: set[str]
):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = imports.get(import_id)
        if record.status in expected:
            return record
        time.sleep(0.02)
    raise AssertionError(f"Import {import_id} did not reach {expected}")


def test_untrusted_checkpoint_is_rejected_before_staging(tmp_path: Path) -> None:
    imports = service(tmp_path)

    with pytest.raises(ModelImportError, match="trusted"):
        imports.inspect(io.BytesIO(b"checkpoint"), "best.pt", trusted=False)

    assert not (tmp_path / "model-imports").exists()


def test_inspection_is_async_and_conversion_requires_a_second_action(tmp_path: Path) -> None:
    imports = service(tmp_path)
    started = time.perf_counter()

    record = imports.inspect(
        io.BytesIO(b"slow checkpoint"),
        "best.pt",
        trusted=True,
        config_stream=io.BytesIO(b"imgsz: 640\n"),
        config_filename="args.yaml",
    )

    assert time.perf_counter() - started < 0.3
    assert record.status == "inspecting"
    inspected = wait_for_status(imports, record.id, {"awaiting_confirmation", "failed"})
    assert inspected.status == "awaiting_confirmation"
    assert inspected.inspection is not None
    assert inspected.inspection["class_names"] == ["coral-open", "coral-closed"]
    assert imports.registry.list() == []

    converting = imports.convert(
        record.id,
        model_name="Synthetic thesis detector",
        confidence_threshold=0.25,
        iou_threshold=0.45,
    )
    assert converting.status == "converting"
    completed = wait_for_status(imports, record.id, {"completed", "failed"})
    assert completed.status == "completed"
    assert completed.model_id is not None
    registered = imports.registry.get(completed.model_id)
    assert registered.metadata.classes == {0: "coral-open", 1: "coral-closed"}
    assert (registered.path.parent / "source.pt").read_bytes() == b"slow checkpoint"
    assert (registered.path.parent / "source-config.yaml").read_bytes() == b"imgsz: 640\n"


def test_converter_failure_is_actionable_and_does_not_register_model(
    tmp_path: Path,
) -> None:
    imports = service(tmp_path)
    record = imports.inspect(
        io.BytesIO(b"malformed checkpoint"), "best.pt", trusted=True
    )

    failed = wait_for_status(imports, record.id, {"failed"})

    assert failed.error == "Checkpoint could not be loaded."
    assert imports.registry.list() == []


def test_restart_marks_unfinished_job_interrupted(tmp_path: Path) -> None:
    imports_dir = tmp_path / "model-imports" / "job-1"
    imports_dir.mkdir(parents=True)
    (imports_dir / "job.json").write_text(
        json.dumps(
            {
                "id": "job-1",
                "status": "converting",
                "filename": "best.pt",
                "source_sha256": "a" * 64,
                "inspection": None,
                "error": None,
                "model_id": None,
                "created_at": "2026-11-16T21:00:00Z",
                "updated_at": "2026-11-16T21:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    restarted = service(tmp_path)

    record = restarted.get("job-1")
    assert record.status == "interrupted"
    assert record.error is not None
    assert "restart" in record.error.casefold()
