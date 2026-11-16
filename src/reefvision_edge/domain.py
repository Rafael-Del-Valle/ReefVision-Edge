from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreprocessingMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color_order: Literal["rgb"]
    scale: Literal["0_1"]
    letterbox: Literal["centered"]


class OutputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layout: Literal["1x(4+classes)xN"]
    box_format: Literal["xywh"]
    score_format: Literal["class_probabilities"]


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160)
    format: Literal["ultralytics_yolo_detect_v1"]
    input_width: int = Field(ge=32, le=4096, multiple_of=32)
    input_height: int = Field(ge=32, le=4096, multiple_of=32)
    classes: dict[int, str] = Field(min_length=1)
    preprocessing: PreprocessingMetadata
    output: OutputMetadata
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)

    @model_validator(mode="after")
    def validate_class_map(self) -> "ModelMetadata":
        if sorted(self.classes) != list(range(len(self.classes))):
            raise ValueError("Class IDs must be contiguous integers starting at zero.")
        if any(not name.strip() for name in self.classes.values()):
            raise ValueError("Class names must not be empty.")
        return self


@dataclass(frozen=True)
class RegisteredModel:
    id: str
    metadata: ModelMetadata
    path: Path
    model_sha256: str
    metadata_sha256: str
    provider: str
    available_providers: tuple[str, ...]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    latency_ms: float
    provider: str
