from __future__ import annotations

import os
import sys
from pathlib import Path

from reefvision_edge.cli import main

if __name__ == "__main__":
    log_file = os.environ.get("REEFVISION_EDGE_LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        stream = open(log_file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = stream
        sys.stderr = stream
    main()
