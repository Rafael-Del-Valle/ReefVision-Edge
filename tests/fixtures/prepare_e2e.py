from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from tests.fixtures.build_tiny_model import build_tiny_yolo_model
from tests.fixtures.build_video import build_mjpeg_video
from tests.test_model_registry import valid_metadata


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.fixtures.prepare_e2e OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    if output.name != ".e2e-fixtures" or output.parent.name != "web":
        raise SystemExit(f"refusing to replace unexpected fixture directory: {output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    build_tiny_yolo_model(output / "tiny.onnx")
    build_mjpeg_video(output / "reef.avi")
    (output / "metadata.json").write_text(
        json.dumps(valid_metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
