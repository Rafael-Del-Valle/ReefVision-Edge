from __future__ import annotations

from pathlib import Path

from tests.test_sessions import prepared_services

from reefvision_edge.media_store import MediaStore
from reefvision_edge.model_registry import ModelRegistry
from reefvision_edge.sessions import SessionRunner, SessionService


def test_unavailable_source_finishes_failed_with_actionable_error(tmp_path: Path) -> None:
    database, model, source, previews, _, _ = prepared_services(tmp_path)
    source.path.unlink()
    runner = SessionRunner(
        database,
        ModelRegistry(database, tmp_path / "models"),
        MediaStore(database, tmp_path / "media"),
        previews,
    )
    service = SessionService(database, runner)
    session = service.create(model.id, source.id, sample_every=1)

    runner.run(session.id)

    failed = service.get(session.id)
    assert failed.status == "failed"
    assert "source" in (failed.error_message or "").casefold()
    assert failed.completed_at is not None


def test_database_write_failure_preserves_earlier_frames_and_fails_session(
    tmp_path: Path,
) -> None:
    database, model, source, _previews, runner, service = prepared_services(tmp_path)
    with database.connect() as connection:
        connection.executescript(
            """
            CREATE TRIGGER simulate_disk_full
            BEFORE INSERT ON frames
            WHEN NEW.frame_index >= 2
            BEGIN
                SELECT RAISE(FAIL, 'database or disk is full');
            END;
            """
        )
    session = service.create(model.id, source.id, sample_every=1)

    runner.run(session.id)

    failed = service.get(session.id)
    assert failed.status == "failed"
    assert "disk" in (failed.error_message or "").casefold()
    with database.connect() as connection:
        committed = connection.execute(
            "SELECT frame_index FROM frames WHERE session_id = ? ORDER BY frame_index",
            (session.id,),
        ).fetchall()
    assert committed == [(0,), (1,)]


def test_service_restart_recovers_running_session(tmp_path: Path) -> None:
    database, model, source, _, _, service = prepared_services(tmp_path)
    session = service.create(model.id, source.id, sample_every=1)
    with database.connect() as connection:
        connection.execute(
            "UPDATE sessions SET status = 'running' WHERE id = ?", (session.id,)
        )

    recovered = database.recover_interrupted_sessions()

    assert recovered == 1
    assert service.get(session.id).status == "interrupted"
