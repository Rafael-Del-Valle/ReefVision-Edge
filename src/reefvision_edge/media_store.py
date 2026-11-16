from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from reefvision_edge.db import Database
from reefvision_edge.sources import SourceError, VideoSource, probe_video


class MediaStore:
    def __init__(self, database: Database, storage_dir: Path) -> None:
        self.database = database
        self.storage_dir = storage_dir

    def import_video(self, stream: BinaryIO, filename: str) -> VideoSource:
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.casefold()
        if suffix not in {".avi", ".mp4", ".mov", ".mkv", ".m4v"}:
            raise SourceError("Supported video containers are AVI, MP4, MOV, M4V, and MKV.")
        source_id = str(uuid.uuid4())
        target_dir = self.storage_dir / source_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / f"source{suffix}"
        digest = hashlib.sha256()
        try:
            with target.open("wb") as output:
                while chunk := stream.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            probe = probe_video(target)
            source = VideoSource(
                id=source_id,
                name=safe_name,
                path=target,
                sha256=digest.hexdigest(),
                probe=probe,
            )
            metadata = json.dumps(
                {
                    "width": probe.width,
                    "height": probe.height,
                    "fps": probe.fps,
                    "frame_count": probe.frame_count,
                    "codec": probe.codec,
                    "container": probe.container,
                    "timestamp_basis": probe.timestamp_basis,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            with self.database.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO sources (
                        id, kind, name, path, sha256, metadata_json, created_at
                    ) VALUES (?, 'video', ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (source.id, source.name, str(source.path), source.sha256, metadata),
                )
            return source
        except OSError as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise SourceError(f"Video could not be stored: {exc}") from exc
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
