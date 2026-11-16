import type { SessionStatus } from "../types";

const labels: Record<SessionStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  canceled: "Canceled",
  interrupted: "Interrupted",
  failed: "Failed",
};

export function StatusBadge({ status }: { status: SessionStatus }) {
  return (
    <span
      className="status-badge"
      data-state={status}
      role="status"
      aria-label={status}
    >
      <span className="status-dot" aria-hidden="true" />
      {labels[status]}
    </span>
  );
}
