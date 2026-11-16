import type { ModelRecord, SessionRecord, SourceRecord } from "../types";
import { StatusBadge } from "../components/StatusBadge";

export function Sessions({
  sessions,
  models,
  sources,
  onOpen,
}: {
  sessions: SessionRecord[];
  models: ModelRecord[];
  sources: SourceRecord[];
  onOpen: (id: string) => void;
}) {
  return (
    <section className="view" aria-labelledby="sessions-title">
      <div className="view-heading">
        <div>
          <p className="eyebrow">Durable local record</p>
          <h2 id="sessions-title">Saved sessions</h2>
          <p>
            Completed, interrupted, and failed runs remain available after
            restart.
          </p>
        </div>
        <span className="quiet-chip">{sessions.length} total</span>
      </div>
      <div className="session-list">
        {sessions.length ? (
          sessions.map((session) => {
            const model = models.find((item) => item.id === session.model_id);
            const source = sources.find(
              (item) => item.id === session.source_id,
            );
            return (
              <article className="session-card panel" key={session.id}>
                <div className="session-main">
                  <div className="session-icon" aria-hidden="true">
                    RV
                  </div>
                  <div>
                    <div className="session-title-line">
                      <h3>{source?.name ?? "Unknown source"}</h3>
                      <StatusBadge status={session.status} />
                    </div>
                    <p>
                      {model?.name ?? "Unknown model"} · every{" "}
                      {session.sample_every} frame
                      {session.sample_every === 1 ? "" : "s"}
                    </p>
                    {session.error_message ? (
                      <p className="session-error">{session.error_message}</p>
                    ) : null}
                  </div>
                </div>
                <div className="session-stats">
                  <span>{session.frame_count} frames</span>
                  <span>{session.detection_count} detections</span>
                  <span>
                    <strong>
                      {session.average_latency_ms?.toFixed(1) ?? "—"}
                    </strong>{" "}
                    ms
                  </span>
                </div>
                <div className="session-actions">
                  <time>{new Date(session.created_at).toLocaleString()}</time>
                  <button
                    className="secondary small"
                    onClick={() => onOpen(session.id)}
                  >
                    Open monitor
                  </button>
                  <a
                    className="button-link small"
                    href={`/api/sessions/${session.id}/export`}
                  >
                    Export
                  </a>
                </div>
              </article>
            );
          })
        ) : (
          <div className="empty-card panel">
            <h3>No sessions yet</h3>
            <p>Start a recorded or live run from Monitor.</p>
          </div>
        )}
      </div>
    </section>
  );
}
