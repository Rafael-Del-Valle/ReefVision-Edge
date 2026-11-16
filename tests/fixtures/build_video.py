from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_mjpeg_video(
    path: Path,
    *,
    frame_count: int = 5,
    width: int = 160,
    height: int = 90,
    fps: float = 10.0,
) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPEG fixture writer is unavailable")
    try:
        for index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (
                (index * 20) % 256,
                (index * 30) % 256,
                (index * 40) % 256,
            )
            cv2.putText(
                frame,
                str(index),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path
