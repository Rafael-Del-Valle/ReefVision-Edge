from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from reefvision_edge.db import Database
from reefvision_edge.sources import SourceError, V4L2CameraSource, VideoSource, probe_video


class MediaStore:
    def __init__(
        self,
        database: Database,
        storage_dir: Path,
        *,
        max_video_bytes: int = 256 * 1024 * 1024 * 1024,
    ) -> None:
        self.database = database
        self.storage_dir = storage_dir
        self.max_video_bytes = max_video_bytes

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
        stored_bytes = 0
        try:
            with target.open("wb") as output:
                while chunk := stream.read(1024 * 1024):
                    stored_bytes += len(chunk)
                    if stored_bytes > self.max_video_bytes:
                        raise SourceError(
                            f"Video exceeds the {self.max_video_bytes}-byte upload limit."
                        )
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

    def get(self, source_id: str) -> VideoSource:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, path, sha256, metadata_json
                FROM sources WHERE id = ? AND kind = 'video'
                """,
                (source_id,),
            ).fetchone()
        if row is None:
            raise SourceError(f"Recorded source {source_id} was not found.")
        metadata = json.loads(row[4])
        from reefvision_edge.sources import VideoProbe

        return VideoSource(
            id=row[0],
            name=row[1],
            path=Path(row[2]),
            sha256=row[3],
            probe=VideoProbe(**metadata),
        )

    def get_record(self, source_id: str) -> dict[str, object]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, name, path, sha256, metadata_json, created_at
                FROM sources WHERE id = ?
                """,
                (source_id,),
            ).fetchone()
        if row is None:
            raise SourceError(f"Source {source_id} was not found.")
        return {
            "id": row[0],
            "kind": row[1],
            "name": row[2],
            "path": row[3],
            "sha256": row[4],
            "metadata": json.loads(row[5]),
            "created_at": row[6],
        }

    def list(self) -> list[dict[str, object]]:
        with self.database.connect() as connection:
            identifiers = connection.execute(
                "SELECT id FROM sources ORDER BY created_at DESC"
            ).fetchall()
        return [self.get_record(row[0]) for row in identifiers]

    def register_camera(self, device: Path) -> dict[str, object]:
        V4L2CameraSource(device)
        source_id = str(uuid.uuid4())
        metadata = json.dumps(
            {"device": str(device), "timestamp_basis": "live capture order"},
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    id, kind, name, path, sha256, metadata_json, created_at
                ) VALUES (?, 'camera', ?, ?, NULL, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (source_id, device.name, str(device), metadata),
            )
        return self.get_record(source_id)
