from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LATENCY_STATISTICS_WINDOW_FRAMES = 4096


class InferenceProfile(StrEnum):
    FULL_FRAME = "full_frame"
    THESIS_TILING = "thesis_tiling"


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


class ConversionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_format: Literal["ultralytics_pt"]
    source_filename: str = Field(min_length=1, max_length=255)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_task: str = Field(min_length=1, max_length=80)
    architecture_family: str = Field(min_length=1, max_length=160)
    exporter_name: str = Field(min_length=1, max_length=80)
    exporter_version: str = Field(min_length=1, max_length=80)
    runtime_versions: dict[str, str] = Field(min_length=1)
    export_settings: dict[str, bool | int | float | str]
    class_names: tuple[str, ...] = Field(min_length=1)
    input_width: int = Field(ge=32, le=4096, multiple_of=32)
    input_height: int = Field(ge=32, le=4096, multiple_of=32)
    preprocessing: PreprocessingMetadata
    output: OutputMetadata
    confidence_threshold: float = Field(ge=0, le=1)
    iou_threshold: float = Field(ge=0, le=1)
    converted_at: str = Field(min_length=1, max_length=80)


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
    conversion: ConversionProvenance | None = None
    supported_profiles: tuple[InferenceProfile, ...] = (InferenceProfile.FULL_FRAME,)

    @model_validator(mode="after")
    def validate_class_map(self) -> ModelMetadata:
        if sorted(self.classes) != list(range(len(self.classes))):
            raise ValueError("Class IDs must be contiguous integers starting at zero.")
        if any(not name.strip() for name in self.classes.values()):
            raise ValueError("Class names must not be empty.")
        if not self.supported_profiles:
            raise ValueError("At least one inference profile must be supported.")
        if self.supported_profiles[0] is not InferenceProfile.FULL_FRAME:
            raise ValueError("The full-frame inference profile must be first.")
        if len(set(self.supported_profiles)) != len(self.supported_profiles):
            raise ValueError("Inference profiles must not contain duplicates.")
        if self.conversion is not None:
            expected_names = tuple(self.classes[index] for index in range(len(self.classes)))
            if self.conversion.class_names != expected_names:
                raise ValueError("Conversion class order must match the registered class map.")
            if (
                self.conversion.input_width != self.input_width
                or self.conversion.input_height != self.input_height
            ):
                raise ValueError("Conversion input shape must match registered model metadata.")
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


@dataclass(frozen=True)
class OverlaySettings:
    timer: bool = True
    trails: bool = False
    boxes: bool = True
    masks: bool = False
    labels: bool = True
