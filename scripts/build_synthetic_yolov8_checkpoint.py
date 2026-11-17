from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

CLASS_NAMES = (
    "Acropora_open",
    "Acropora_closed",
    "Montipora_open",
    "Montipora_closed",
    "Pocillopora_open",
    "Pocillopora_closed",
)


def synthetic_frame(index: int, width: int = 1280, height: int = 720) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    blue = (35 + 25 * np.sin((x + index * 9) / 80)).astype(np.uint8)
    green = (75 + 35 * np.sin((y + index * 5) / 55)).astype(np.uint8)
    red = (55 + 20 * np.cos((x + y) / 110)).astype(np.uint8)
    frame = np.dstack((blue, green, red))
    for offset, color in ((0, (80, 150, 220)), (280, (180, 110, 75))):
        center = (260 + offset + (index * 7) % 180, 360 + int(45 * np.sin(index / 4)))
        cv2.circle(frame, center, 110, color, -1, cv2.LINE_AA)
        for angle in range(0, 360, 30):
            radians = np.deg2rad(angle + index * 2)
            endpoint = (
                int(center[0] + 145 * np.cos(radians)),
                int(center[1] + 145 * np.sin(radians)),
            )
            cv2.line(frame, center, endpoint, color, 18, cv2.LINE_AA)
    return frame


def build_checkpoint(path: Path) -> None:
    torch.manual_seed(20261116)
    model = DetectionModel("yolov8n.yaml", nc=len(CLASS_NAMES), verbose=False)
    model.names = dict(enumerate(CLASS_NAMES))
    model.train()
    with torch.no_grad():
        for _ in range(8):
            model(torch.rand(4, 3, 320, 320))
    model.eval()
    head = model.model[-1]
    with torch.no_grad():
        for branch in head.cv3:
            branch[-1].weight.zero_()
            branch[-1].weight[0].normal_(mean=0.0, std=100.0)
            branch[-1].bias.fill_(-12)
            branch[-1].bias[0] = 4
    wrapper = YOLO("yolov8n.yaml")
    wrapper.model = model
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(path)


def _write_video(path: Path, *, width: int, height: int, frame_count: int, fps: float) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}.")
    try:
        for index in range(frame_count):
            writer.write(synthetic_frame(index, width=width, height=height))
    finally:
        writer.release()
    return path


def build_media(output: Path) -> tuple[list[Path], Path, Path]:
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for index in (0, 12, 24):
        path = images / f"coral-{index:02d}.png"
        if not cv2.imwrite(str(path), synthetic_frame(index)):
            raise RuntimeError(f"Could not write {path}.")
        image_paths.append(path)
    video = _write_video(
        output / "synthetic-coral.avi",
        width=1280,
        height=720,
        frame_count=45,
        fps=15.0,
    )
    tiled_video = _write_video(
        output / "synthetic-coral-tiled.avi",
        width=2304,
        height=1536,
        frame_count=18,
        fps=6.0,
    )
    return image_paths, video, tiled_video


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "synthetic-yolov8n-six-class.pt"
    build_checkpoint(checkpoint)
    images, video, tiled_video = build_media(args.output)
    config = args.output / "args.yaml"
    config.write_text(
        "task: detect\nmodel: yolov8n.yaml\nimgsz: 320\nrect: false\n"
        "conf: 0.999\niou: 1.0\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    files = [checkpoint, config, video, tiled_video, *images]
    manifest = {
        "schema": "reefvision-edge-synthetic-model/v1",
        "checkpoint": str(checkpoint.resolve()),
        "class_names": list(CLASS_NAMES),
        "input_size": [320, 320],
        "rect": False,
        "confidence_threshold": 0.999,
        "iou_threshold": 1.0,
        "tiled_video": {
            "path": str(tiled_video.resolve()),
            "source_size": [2304, 1536],
            "frames": 18,
            "fps": 6.0,
            "tile_size": [1920, 1344],
            "overlap": 0.4,
        },
        "files": {str(path.relative_to(args.output)): sha256(path) for path in files},
    }
    (args.output / "fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
