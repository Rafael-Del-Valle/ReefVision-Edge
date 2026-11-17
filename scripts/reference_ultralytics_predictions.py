from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    model = YOLO(str(args.checkpoint))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}.")
    predictions: list[dict[str, object]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                rect=False,
                conf=args.confidence,
                iou=args.iou,
                device="cpu",
                verbose=False,
            )[0]
            for class_id, confidence, box in zip(
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.xyxy.tolist(),
                strict=True,
            ):
                if box[2] <= box[0] or box[3] <= box[1]:
                    # ReefVision-Edge's source-coordinate contract excludes clipped
                    # zero-area boxes. The deliberately biased synthetic head emits
                    # some at the letterbox boundary; real detections must be valid.
                    continue
                predictions.append(
                    {
                        "frame_index": frame_index,
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "box": [float(value) for value in box],
                    }
                )
            frame_index += 1
    finally:
        capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": frame_index, "detections": len(predictions)}))


if __name__ == "__main__":
    main()
