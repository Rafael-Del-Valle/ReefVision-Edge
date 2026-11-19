from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from reefvision_edge.db import Database


FIELD_SWITCH_CONFIG_KEY = "field_switch_config_v1"


class FieldSwitchPreferences(BaseModel):
    """Persistent behavior for the optional physical collection switch.

    The hardware controller reads this before beginning a new acquisition.
    Saving these preferences alone does not activate GPIO hardware.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["logging_only", "inference_and_logging"]


def load_field_switch_preferences(database: Database) -> FieldSwitchPreferences:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (FIELD_SWITCH_CONFIG_KEY,)
        ).fetchone()
    if row is None:
        return FieldSwitchPreferences(mode="inference_and_logging")
    return FieldSwitchPreferences.model_validate_json(row[0])


def save_field_switch_preferences(
    database: Database, preferences: FieldSwitchPreferences
) -> FieldSwitchPreferences:
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (FIELD_SWITCH_CONFIG_KEY, preferences.model_dump_json()),
        )
    return load_field_switch_preferences(database)
