from __future__ import annotations

from enum import StrEnum


class SessionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
