from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from reefvision_edge.domain import ConversionProvenance
from reefvision_edge.model_registry import ModelRegistry


class ModelImportError(ValueError):
    pass


@dataclass(frozen=True)
class ModelImportRecord:
    id: str
    status: str
    filename: str
    source_sha256: str
    inspection: dict[str, object] | None
    error: str | None
    model_id: str | None
    created_at: str
    updated_at: str


class ModelImportService:
    def __init__(
        self,
        registry: ModelRegistry,
        storage_dir: Path,
        *,
        converter_command: tuple[str, ...] | None = None,
        max_checkpoint_bytes: int = 512 * 1024 * 1024,
        max_config_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.registry = registry
        self.storage_dir = storage_dir
        self.converter_command = converter_command or (
            sys.executable,
            "-m",
            "reefvision_edge.converter_worker",
        )
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.max_config_bytes = max_config_bytes
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._operations: dict[str, str] = {}
        if storage_dir.is_dir():
            for path in storage_dir.glob("*/job.json"):
                record = self._read_record(path.parent.name)
                if record.status in {"inspecting", "converting"}:
                    self._write_record(
                        replace(
                            record,
                            status="interrupted",
                            error="Conversion was interrupted by a service restart; inspect again.",
                            updated_at=_now(),
                        )
                    )

    def inspect(
        self,
        checkpoint_stream: BinaryIO,
        filename: str,
        *,
        trusted: bool,
        config_stream: BinaryIO | None = None,
        config_filename: str | None = None,
    ) -> ModelImportRecord:
        if not trusted:
            raise ModelImportError(
                "Explicitly confirm that this local PyTorch checkpoint is trusted before "
                "loading it."
            )
        if Path(filename).suffix.casefold() != ".pt":
            raise ModelImportError("Trusted checkpoint import requires an Ultralytics .pt file.")
        checkpoint = checkpoint_stream.read(self.max_checkpoint_bytes + 1)
        if not checkpoint:
            raise ModelImportError("The checkpoint file is empty.")
        if len(checkpoint) > self.max_checkpoint_bytes:
            raise ModelImportError("The checkpoint exceeds the configured upload limit.")
        config: bytes | None = None
        if config_stream is not None:
            if config_filename is None or Path(config_filename).suffix.casefold() not in {
                ".yaml",
                ".yml",
            }:
                raise ModelImportError("The optional configuration must be args.yaml or data.yaml.")
            config = config_stream.read(self.max_config_bytes + 1)
            if len(config) > self.max_config_bytes:
                raise ModelImportError("The configuration exceeds the upload limit.")

        import_id = str(uuid.uuid4())
        job_dir = self.storage_dir / import_id
        job_dir.mkdir(parents=True, exist_ok=False)
        _durable_write(job_dir / "checkpoint.pt", checkpoint)
        if config is not None:
            _durable_write(job_dir / "source-config.yaml", config)
        now = _now()
        record = ModelImportRecord(
            id=import_id,
            status="inspecting",
            filename=Path(filename).name,
            source_sha256=hashlib.sha256(checkpoint).hexdigest(),
            inspection=None,
            error=None,
            model_id=None,
            created_at=now,
            updated_at=now,
        )
        self._write_record(record)
        self._start(import_id, "inspect", {})
        return record

    def convert(
        self,
        import_id: str,
        *,
        model_name: str,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> ModelImportRecord:
        record = self.get(import_id)
        if record.status != "awaiting_confirmation":
            raise ModelImportError(
                f"Checkpoint {import_id} cannot convert from status {record.status}."
            )
        if not model_name.strip():
            raise ModelImportError("Model name is required.")
        if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise ModelImportError("Confidence and IoU thresholds must be between 0 and 1.")
        converting = replace(
            record,
            status="converting",
            error=None,
            updated_at=_now(),
        )
        self._write_record(converting)
        self._start(
            import_id,
            "convert",
            {
                "model_name": model_name.strip(),
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
            },
        )
        return converting

    def get(self, import_id: str) -> ModelImportRecord:
        record = self._read_record(import_id)
        return self._refresh(record)

    def list(self) -> list[ModelImportRecord]:
        if not self.storage_dir.is_dir():
            return []
        records = [self.get(path.parent.name) for path in self.storage_dir.glob("*/job.json")]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def shutdown(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
        self._processes.clear()
        self._operations.clear()

    def _start(self, import_id: str, operation: str, extra: dict[str, object]) -> None:
        job_dir = self.storage_dir / import_id
        request = {
            "schema": "reefvision-edge-model-conversion/v1",
            "operation": operation,
            "checkpoint_path": str(job_dir / "checkpoint.pt"),
            "config_path": (
                str(job_dir / "source-config.yaml")
                if (job_dir / "source-config.yaml").is_file()
                else None
            ),
            "original_filename": self._read_record(import_id).filename,
            "output_dir": str(job_dir / "derived"),
            **extra,
        }
        request_path = job_dir / f"{operation}-request.json"
        result_path = job_dir / f"{operation}-result.json"
        _write_json(request_path, request)
        command = (
            *self.converter_command,
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        )
        log = (job_dir / f"{operation}.log").open("ab")
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        converter_environment = os.environ.copy()
        converter_environment["ULTRALYTICS_OFFLINE"] = "true"
        converter_environment["YOLO_CONFIG_DIR"] = str(job_dir / "ultralytics-config")
        try:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
                creationflags=creation_flags,
                env=converter_environment,
            )
        except OSError as exc:
            failed = replace(
                self._read_record(import_id),
                status="failed",
                error=f"The model converter could not start: {exc}",
                updated_at=_now(),
            )
            self._write_record(failed)
            raise ModelImportError(failed.error or "The model converter could not start.") from exc
        finally:
            log.close()
        self._processes[import_id] = process
        self._operations[import_id] = operation

    def _refresh(self, record: ModelImportRecord) -> ModelImportRecord:
        process = self._processes.get(record.id)
        if process is None or process.poll() is None:
            return record
        operation = self._operations.pop(record.id)
        self._processes.pop(record.id, None)
        result_path = self.storage_dir / record.id / f"{operation}-result.json"
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._fail(
                record,
                f"The model converter exited without a valid result: {exc}",
            )
        if not result.get("ok"):
            return self._fail(record, str(result.get("error") or "Model conversion failed."))
        try:
            inspection_value = result["inspection"]
            if not isinstance(inspection_value, dict):
                raise ModelImportError("Converter inspection result is invalid.")
            if operation == "inspect":
                updated = replace(
                    record,
                    status="awaiting_confirmation",
                    inspection=inspection_value,
                    error=None,
                    updated_at=_now(),
                )
            else:
                job_dir = self.storage_dir / record.id
                registered = self.registry.register_converted(
                    job_dir / "checkpoint.pt",
                    Path(str(result["onnx_path"])),
                    result["metadata"],
                    ConversionProvenance.model_validate(result["provenance"]),
                    source_config_path=(
                        job_dir / "source-config.yaml"
                        if (job_dir / "source-config.yaml").is_file()
                        else None
                    ),
                )
                updated = replace(
                    record,
                    status="completed",
                    inspection=inspection_value,
                    error=None,
                    model_id=registered.id,
                    updated_at=_now(),
                )
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(record, f"Converted model could not be registered: {exc}")
        self._write_record(updated)
        return updated

    def _fail(self, record: ModelImportRecord, message: str) -> ModelImportRecord:
        failed = replace(record, status="failed", error=message, updated_at=_now())
        self._write_record(failed)
        return failed

    def _read_record(self, import_id: str) -> ModelImportRecord:
        path = self.storage_dir / import_id / "job.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return ModelImportRecord(**value)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ModelImportError(
                f"Model import {import_id} was not found or is invalid."
            ) from exc

    def _write_record(self, record: ModelImportRecord) -> None:
        _write_json(self.storage_dir / record.id / "job.json", asdict(record))


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _durable_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
