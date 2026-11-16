import { FormEvent, useMemo, useState } from "react";

import type { ModelRecord, SessionRecord, SourceRecord } from "../types";
import { Metric } from "../components/Metric";
import { StatusBadge } from "../components/StatusBadge";

interface MonitorProps {
  models: ModelRecord[];
  sources: SourceRecord[];
  sessions: SessionRecord[];
  activeSessionId: string | null;
  onActiveSession: (id: string | null) => void;
  onStart: (
    modelId: string,
    sourceId: string,
    sampleEvery: number,
  ) => Promise<void>;
  onStop: (sessionId: string) => Promise<void>;
  onModelUpload: (model: File, metadata: File) => Promise<void>;
  onVideoUpload: (video: File) => Promise<void>;
  onCamera: (device: string) => Promise<void>;
}

export function Monitor({
  models,
  sources,
  sessions,
  activeSessionId,
  onActiveSession,
  onStart,
  onStop,
  onModelUpload,
  onVideoUpload,
  onCamera,
}: MonitorProps) {
  const [modelId, setModelId] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [sampleEvery, setSampleEvery] = useState(1);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [cameraDevice, setCameraDevice] = useState("/dev/video0");

  const selectedModel = modelId || models[0]?.id || "";
  const selectedSource = sourceId || sources[0]?.id || "";
  const active = useMemo(
    () =>
      sessions.find((item) => item.id === activeSessionId) ??
      sessions[0] ??
      null,
    [activeSessionId, sessions],
  );
  const provider =
    active?.provider ??
    models.find((item) => item.id === selectedModel)?.provider;
  const running = active?.status === "running" || active?.status === "queued";

  async function act(name: string, action: () => Promise<void>) {
    setBusy(name);
    setActionError(null);
    try {
      await action();
    } catch (error) {
      setActionError(
        error instanceof Error ? error.message : "The action failed.",
      );
    } finally {
      setBusy(null);
    }
  }

  async function uploadModel(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const model = form.get("model-file");
    const metadata = form.get("metadata-file");
    if (
      !(model instanceof File) ||
      !(metadata instanceof File) ||
      !model.size ||
      !metadata.size
    ) {
      setActionError("Choose both an ONNX model and its JSON metadata.");
      return;
    }
    await act("model", () => onModelUpload(model, metadata));
    event.currentTarget.reset();
  }

  async function uploadVideo(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const video = form.get("video-file");
    if (!(video instanceof File) || !video.size) {
      setActionError("Choose a local video to import.");
      return;
    }
    await act("video", () => onVideoUpload(video));
    event.currentTarget.reset();
  }

  return (
    <section className="view" aria-labelledby="monitor-title">
      <div className="view-heading">
        <div>
          <p className="eyebrow">Jarvis · offline field station</p>
          <h2 id="monitor-title">Field monitor</h2>
          <p>Run local detections and keep the latest frame responsive.</p>
        </div>
        <div className="connection-state">
          <span className="connection-light" aria-hidden="true" />
          Local service ready
        </div>
      </div>

      {actionError ? (
        <div className="inline-alert" role="alert">
          {actionError}
        </div>
      ) : null}

      <div className="monitor-grid">
        <div className="preview-panel panel">
          <div className="panel-title-row">
            <div>
              <span className="panel-kicker">Live preview</span>
              <h3>
                {active
                  ? `Session ${active.id.slice(0, 8)}`
                  : "Waiting for a run"}
              </h3>
            </div>
            {active ? (
              <StatusBadge status={active.status} />
            ) : (
              <span className="quiet-chip">Idle</span>
            )}
          </div>
          <div className="preview-stage">
            {active && active.frame_count > 0 ? (
              <img
                key={`${active.id}-${active.updated_at}`}
                src={`/api/previews/${active.id}?v=${encodeURIComponent(active.updated_at)}`}
                alt="Latest inference frame with detection overlays"
              />
            ) : (
              <div className="preview-empty">
                <div className="reticle" aria-hidden="true" />
                <strong>No frame yet</strong>
                <span>Select a model and source, then start inference.</span>
              </div>
            )}
          </div>
          <div className="metrics-grid" aria-label="Session metrics">
            <Metric
              label="Inference"
              value={
                active?.inference_fps
                  ? `${active.inference_fps.toFixed(1)} FPS`
                  : "—"
              }
            />
            <Metric
              label="Latency"
              value={
                active?.average_latency_ms != null
                  ? `${active.average_latency_ms.toFixed(1)} ms`
                  : "—"
              }
            />
            <Metric
              label="Frames"
              value={String(active?.frame_count ?? 0)}
              detail={`${active?.dropped_frames ?? 0} dropped`}
            />
            <Metric
              label="Detections"
              value={String(active?.detection_count ?? 0)}
            />
          </div>
        </div>

        <aside className="control-stack" aria-label="Inference controls">
          <div className="panel control-panel">
            <div className="panel-title-row compact">
              <div>
                <span className="panel-kicker">Run setup</span>
                <h3>Inference controls</h3>
              </div>
              <span className="provider-pill">
                {provider === "CPUExecutionProvider"
                  ? "CPU baseline"
                  : (provider ?? "No model")}
              </span>
            </div>

            <label>
              <span>Model</span>
              <select
                value={selectedModel}
                onChange={(event) => setModelId(event.target.value)}
              >
                {models.length ? (
                  models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))
                ) : (
                  <option value="">Register a model first</option>
                )}
              </select>
            </label>
            <label>
              <span>Source</span>
              <select
                value={selectedSource}
                onChange={(event) => setSourceId(event.target.value)}
              >
                {sources.length ? (
                  sources.map((source) => (
                    <option key={source.id} value={source.id}>
                      {source.name} · {source.kind}
                    </option>
                  ))
                ) : (
                  <option value="">Import a source first</option>
                )}
              </select>
            </label>
            <label>
              <span>Sample every N frames</span>
              <input
                aria-label="Sample every N frames"
                type="number"
                min={1}
                max={10000}
                value={sampleEvery}
                onChange={(event) => setSampleEvery(Number(event.target.value))}
              />
              <small>
                Frame 0 is always included; sampling is deterministic.
              </small>
            </label>
            <div className="action-row">
              <button
                className="primary"
                type="button"
                disabled={
                  !selectedModel || !selectedSource || running || busy !== null
                }
                onClick={() =>
                  void act("run", () =>
                    onStart(selectedModel, selectedSource, sampleEvery),
                  )
                }
              >
                {busy === "run" ? "Starting…" : "Start inference"}
              </button>
              <button
                className="secondary danger-action"
                type="button"
                disabled={!active || !running || busy !== null}
                onClick={() =>
                  active && void act("stop", () => onStop(active.id))
                }
              >
                Stop
              </button>
            </div>
            {active ? (
              <div className="active-session-select">
                <label htmlFor="active-session">Displayed session</label>
                <select
                  id="active-session"
                  value={active.id}
                  onChange={(event) => onActiveSession(event.target.value)}
                >
                  {sessions.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.id.slice(0, 8)} · {item.status}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}
          </div>

          <details className="panel import-panel">
            <summary>Register model or source</summary>
            <form onSubmit={(event) => void uploadModel(event)}>
              <h4>Local ONNX model</h4>
              <label>
                <span>ONNX file</span>
                <input name="model-file" type="file" accept=".onnx" />
              </label>
              <label>
                <span>Metadata JSON</span>
                <input
                  name="metadata-file"
                  type="file"
                  accept="application/json,.json"
                />
              </label>
              <button className="secondary" disabled={busy !== null}>
                {busy === "model" ? "Validating…" : "Register model"}
              </button>
            </form>
            <form onSubmit={(event) => void uploadVideo(event)}>
              <h4>Recorded video</h4>
              <label>
                <span>Video file</span>
                <input
                  name="video-file"
                  type="file"
                  accept="video/*,.avi,.mkv"
                />
              </label>
              <button className="secondary" disabled={busy !== null}>
                {busy === "video" ? "Importing…" : "Import video"}
              </button>
            </form>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void act("camera", () => onCamera(cameraDevice));
              }}
            >
              <h4>Linux V4L2 camera</h4>
              <label>
                <span>Device path</span>
                <input
                  value={cameraDevice}
                  onChange={(event) => setCameraDevice(event.target.value)}
                />
              </label>
              <button className="secondary" disabled={busy !== null}>
                Connect camera
              </button>
            </form>
          </details>
        </aside>
      </div>
    </section>
  );
}
