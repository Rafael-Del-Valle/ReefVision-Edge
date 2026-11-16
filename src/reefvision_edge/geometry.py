from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LetterboxMapping:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale: float
    pad_x: int
    pad_y: int


def letterbox(
    frame: NDArray[np.uint8], width: int, height: int
) -> tuple[NDArray[np.uint8], LetterboxMapping]:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    horizontal = width - resized_width
    vertical = height - resized_height
    left = horizontal // 2
    top = vertical // 2
    transformed = cast(
        NDArray[np.uint8],
        cv2.copyMakeBorder(
        resized,
        top,
        vertical - top,
        left,
        horizontal - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
        ),
    )
    mapping = LetterboxMapping(
        source_width=source_width,
        source_height=source_height,
        target_width=width,
        target_height=height,
        scale=scale,
        pad_x=left,
        pad_y=top,
    )
    return transformed, mapping


def restore_box(
    box: tuple[float, float, float, float], mapping: LetterboxMapping
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = box
    restored = (
        (x_min - mapping.pad_x) / mapping.scale,
        (y_min - mapping.pad_y) / mapping.scale,
        (x_max - mapping.pad_x) / mapping.scale,
        (y_max - mapping.pad_y) / mapping.scale,
    )
    return (
        min(max(restored[0], 0.0), float(mapping.source_width)),
        min(max(restored[1], 0.0), float(mapping.source_height)),
        min(max(restored[2], 0.0), float(mapping.source_width)),
        min(max(restored[3], 0.0), float(mapping.source_height)),
    )
