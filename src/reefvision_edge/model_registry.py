from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnxruntime as ort
from pydantic import ValidationError

from reefvision_edge.db import Database
from reefvision_edge.domain import ModelMetadata, RegisteredModel


class ModelRegistrationError(ValueError):
    pass


def canonical_metadata_bytes(metadata: ModelMetadata) -> bytes:
    return json.dumps(
        metadata.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def select_provider(available: list[str]) -> str:
    for provider in (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ):
        if provider in available:
            return provider
    raise ModelRegistrationError("No supported ONNX Runtime execution provider is available.")


class ModelRegistry:
    def __init__(
        self,
        database: Database,
        storage_dir: Path,
        *,
        max_model_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.database = database
        self.storage_dir = storage_dir
        self.max_model_bytes = max_model_bytes

    def register(
        self,
        model_stream: BinaryIO,
        filename: str,
        metadata_value: Mapping[str, object],
    ) -> RegisteredModel:
        if Path(filename).suffix.casefold() != ".onnx":
            raise ModelRegistrationError(
                "Only local ONNX object detectors are supported; checkpoint formats are rejected."
            )
        try:
            metadata = ModelMetadata.model_validate(metadata_value)
        except ValidationError as exc:
            raise ModelRegistrationError(str(exc)) from exc
        model_bytes = model_stream.read(self.max_model_bytes + 1)
        if not model_bytes:
            raise ModelRegistrationError("The ONNX model file is empty.")
        if len(model_bytes) > self.max_model_bytes:
            raise ModelRegistrationError(
                f"The ONNX model exceeds the {self.max_model_bytes}-byte upload limit."
            )

        available = ort.get_available_providers()
        provider = select_provider(available)
        try:
            session = ort.InferenceSession(model_bytes, providers=[provider])
        except Exception as exc:
            raise ModelRegistrationError(f"The ONNX model could not be loaded: {exc}") from exc
        self._validate_session(session, metadata)

        model_id = str(uuid.uuid4())
        target = self.storage_dir / model_id
        temporary = self.storage_dir / f".{model_id}.tmp"
        metadata_bytes = canonical_metadata_bytes(metadata)
        try:
            temporary.mkdir(parents=True, exist_ok=False)
            self._durable_write(temporary / "model.onnx", model_bytes)
            self._durable_write(temporary / "metadata.json", metadata_bytes)
            temporary.replace(target)
            registered = RegisteredModel(
                id=model_id,
                metadata=metadata,
                path=target / "model.onnx",
                model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                provider=session.get_providers()[0],
                available_providers=tuple(available),
            )
            with self.database.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO models (
                        id, name, path, model_sha256, metadata_sha256,
                        metadata_json, provider, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        registered.id,
                        metadata.name,
                        str(registered.path),
                        registered.model_sha256,
                        registered.metadata_sha256,
                        metadata_bytes.decode("utf-8"),
                        registered.provider,
                    ),
                )
            return registered
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            if target.exists():
                shutil.rmtree(target)
            raise

    def get(self, model_id: str) -> RegisteredModel:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, path, model_sha256, metadata_sha256, metadata_json, provider
                FROM models WHERE id = ?
                """,
                (model_id,),
            ).fetchone()
        if row is None:
            raise ModelRegistrationError(f"Registered model {model_id} was not found.")
        return RegisteredModel(
            id=row[0],
            metadata=ModelMetadata.model_validate_json(row[4]),
            path=Path(row[1]),
            model_sha256=row[2],
            metadata_sha256=row[3],
            provider=row[5],
            available_providers=tuple(ort.get_available_providers()),
        )

    def list(self) -> list[RegisteredModel]:
        with self.database.connect() as connection:
            identifiers = connection.execute(
                "SELECT id FROM models ORDER BY created_at DESC"
            ).fetchall()
        return [self.get(row[0]) for row in identifiers]

    @staticmethod
    def _durable_write(path: Path, value: bytes) -> None:
        with path.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _validate_session(session: ort.InferenceSession, metadata: ModelMetadata) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        expected_input = [1, 3, metadata.input_height, metadata.input_width]
        if (
            len(inputs) != 1
            or inputs[0].shape != expected_input
            or inputs[0].type != "tensor(float)"
        ):
            raise ModelRegistrationError(
                f"Model input must be float32 NCHW with fixed shape {expected_input}."
            )
        expected_channels = 4 + len(metadata.classes)
        shape = outputs[0].shape if len(outputs) == 1 else None
        if (
            shape is None
            or len(shape) != 3
            or shape[0] != 1
            or shape[1] != expected_channels
            or not isinstance(shape[2], int)
            or shape[2] < 1
        ):
            raise ModelRegistrationError(
                "Model output does not match [1, 4 + class_count, candidate_count]."
            )
        sample = np.zeros(expected_input, dtype=np.float32)
        result = session.run(None, {inputs[0].name: sample})
        if len(result) != 1 or result[0].shape != tuple(shape):
            raise ModelRegistrationError("Model dry-run output does not match its declared output.")
