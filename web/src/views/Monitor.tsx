import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { FrameSurface, type FrameTelemetry } from "../components/FrameSurface";
import { Metric } from "../components/Metric";
import { StatusBadge } from "../components/StatusBadge";
import type {
  CaptureDevice,
  ComputeDevice,
  ModelRecord,
  MonitorStatus,
  OverlayCapability,
  OverlayName,
  OverlaySettings,
  SessionCreateOptions,
  SessionRecord,
  SourceRecord,
} from "../types";

const defaultOverlays: OverlaySettings = {
  timer: true,
  trails: false,
  boxes: true,
  masks: false,
  labels: true,
};

function loadOverlaySettings(key: string): OverlaySettings {
  try {
    const stored = window.localStorage.getItem(key);
    if (!stored) return defaultOverlays;
    const parsed = JSON.parse(stored) as Partial<OverlaySettings>;
    return (Object.keys(defaultOverlays) as OverlayName[]).reduce(
      (settings, name) => ({
        ...settings,
        [name]:
          typeof parsed[name] === "boolean"
            ? parsed[name]
            : defaultOverlays[name],
      }),
      { ...defaultOverlays },
    );
  } catch {
    return defaultOverlays;
  }
}

function useOverlaySettings(key: string) {
  const [settings, setSettings] = useState<OverlaySettings>(() =>
    loadOverlaySettings(key),
  );
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(settings));
    } catch {
      // Preferences remain usable for this page when browser storage is blocked.
    }
  }, [key, settings]);
  return [settings, setSettings] as const;
}

interface MonitorProps {
  models: ModelRecord[];
  sources: SourceRecord[];
  sessions: SessionRecord[];
  captureDevices: CaptureDevice[];
  monitorStatus: MonitorStatus;
  captureOwned: boolean;
  overlayCapabilities: Record<OverlayName, OverlayCapability>;
  activeSessionId: string | null;
  onActiveSession: (id: string | null) => void;
  onStart: (options: SessionCreateOptions) => Promise<void>;
  onStop: (sessionId: string) => Promise<void>;
  onPreview: (sourceId: string) => Promise<void>;
  onVideoUpload: (video: File) => Promise<void>;
  onCamera: (device: string) => Promise<void>;
  onRefreshCameras: () => Promise<void>;
}

interface OverlayControlsProps {
  title: string;
  prefix: "Preview" | "Recording";
  value: OverlaySettings;
  capabilities: Record<OverlayName, OverlayCapability>;
  onChange: (value: OverlaySettings) => void;
}

function OverlayControls({
  title,
  prefix,
  value,
  capabilities,
  onChange,
}: OverlayControlsProps) {
  return (
    <fieldset className="overlay-controls">
      <legend>{title}</legend>
      {(Object.keys(value) as OverlayName[]).map((name) => {
        const unavailable = value[name] && !capabilities[name].supported;
        return (
          <div className="overlay-option" key={name}>
            <label className="check-row">
              <input
                aria-label={`${prefix} ${name}`}
                type="checkbox"
                checked={value[name]}
                onChange={(event) =>
                  onChange({ ...value, [name]: event.target.checked })
                }
              />
              <span>{name[0].toUpperCase() + name.slice(1)}</span>
            </label>
            {unavailable ? (
              <small className="overlay-unavailable-reason">
                {capabilities[name].reason}
              </small>
            ) : null}
          </div>
        );
      })}
    </fieldset>
  );
}

function monitorLabel(status: MonitorStatus): string {
  if (status.state === "replaying") return "Replay preview";
  if (status.state === "receiving") return "Live camera preview";
  if (status.state === "connecting") return "Connecting source";
  if (status.state === "disconnected") return "Camera disconnected";
  if (status.state === "error") return "Preview error";
  if (status.state === "stalled") return "Preview stalled";
  return "Idle";
}

export function Monitor({
  models,
  sources,
  sessions,
  captureDevices,
  monitorStatus,
  captureOwned,
  overlayCapabilities,
  activeSessionId,
  onActiveSession,
  onStart,
  onStop,
  onPreview,
  onVideoUpload,
  onCamera,
  onRefreshCameras,
}: MonitorProps) {
  const [modelId, setModelId] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [sampleEvery, setSampleEvery] = useState(1);
  const [computeDevice, setComputeDevice] = useState<ComputeDevice>("auto");
  const [recordData, setRecordData] = useState(true);
  const [recordMonitor, setRecordMonitor] = useState(false);
  const [previewOverlays, setPreviewOverlays] = useOverlaySettings(
    "reefvision-edge.preview-overlays",
  );
  const [recordingOverlays, setRecordingOverlays] = useOverlaySettings(
    "reefvision-edge.recording-overlays",
  );
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [cameraDevice, setCameraDevice] = useState("");
  const [frameTelemetry, setFrameTelemetry] = useState<FrameTelemetry | null>(
    null,
  );

  const selectedModel = modelId || models[0]?.id || "";
  const selectedSource = sourceId || sources[0]?.id || "";
  const selectedCamera = cameraDevice || captureDevices[0]?.id || "";
  useEffect(() => {
    if (
      cameraDevice &&
      !captureDevices.some((device) => device.id === cameraDevice)
    )
      setCameraDevice("");
  }, [cameraDevice, captureDevices]);
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
  const sourcePreview = monitorStatus.frame_index >= 0;
  const frameEndpoint = sourcePreview
    ? "/api/monitor/frame"
    : active && active.frame_count > 0
      ? `/api/previews/${active.id}`
      : null;
  const frameStreamKey = sourcePreview
    ? `monitor:${monitorStatus.source_id ?? "unknown"}`
    : active
      ? `session:${active.id}`
      : "idle";
  const frameLabel = sourcePreview
    ? "Live source preview"
    : "Latest inference frame with detection overlays";
  const updateFrameTelemetry = useCallback(
    (value: FrameTelemetry) => setFrameTelemetry(value),
    [],
  );
  useEffect(() => setFrameTelemetry(null), [frameStreamKey]);

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

  const runOptions: SessionCreateOptions = {
    modelId: selectedModel,
    sourceId: selectedSource,
    sampleEvery,
    computeDevice,
    recordData,
    recordMonitor,
    previewOverlays,
    recordingOverlays,
    enableProfiling: computeDevice !== "cpu",
  };

  return (
    <section className="view" aria-labelledby="monitor-title">
      <div className="view-heading">
        <div>
          <p className="eyebrow">Jarvis · offline field station</p>
          <h2 id="monitor-title">Field monitor</h2>
          <p>
            Preview Windows sources, run local inference, and record each
            artifact independently.
          </p>
        </div>
        <div className="connection-state">
          <span className="connection-light" aria-hidden="true" />
          {monitorLabel(monitorStatus)}
        </div>
      </div>

      {actionError ? (
        <div className="inline-alert" role="alert">
          {actionError}
        </div>
      ) : null}
      {monitorStatus.detail ? (
        <div className="inline-alert" role="status">
          {monitorStatus.detail}
        </div>
      ) : null}

      <div className="monitor-grid">
        <div className="preview-panel panel">
          <div className="panel-title-row">
            <div>
              <span className="panel-kicker">Live preview</span>
              <h3>
                {sourcePreview
                  ? monitorLabel(monitorStatus)
                  : active
                    ? `Session ${active.id.slice(0, 8)}`
                    : "Waiting for a source"}
              </h3>
            </div>
            {sourcePreview ? (
              <span className="quiet-chip">{monitorLabel(monitorStatus)}</span>
            ) : active ? (
              <StatusBadge status={active.status} />
            ) : (
              <span className="quiet-chip">Idle</span>
            )}
          </div>
          <div className={`preview-stage${frameEndpoint ? " has-frame" : ""}`}>
            {frameEndpoint ? (
              <FrameSurface
                endpoint={frameEndpoint}
                streamKey={frameStreamKey}
                label={frameLabel}
                targetFps={30}
                onTelemetry={updateFrameTelemetry}
              />
            ) : (
              <div className="preview-empty">
                <div className="reticle" aria-hidden="true" />
                <strong>No frame yet</strong>
                <span>Select a source and choose Preview source.</span>
              </div>
            )}
          </div>
          <div className="metrics-grid" aria-label="Session metrics">
            <Metric
              label="Input"
              value={
                active?.input_fps ? `${active.input_fps.toFixed(1)} FPS` : "—"
              }
            />
            <Metric
              label="Inference"
              value={
                active?.inference_fps
                  ? `${active.inference_fps.toFixed(1)} FPS`
                  : "—"
              }
            />
            <Metric
              label="Display"
              value={
                frameTelemetry
                  ? `${frameTelemetry.displayedFps.toFixed(1)} FPS`
                  : "—"
              }
            />
            <Metric
              label="Frame age"
              value={
                frameTelemetry?.frameAgeMs != null
                  ? `${frameTelemetry.frameAgeMs.toFixed(0)} ms`
                  : "—"
              }
            />
            <Metric
              label="Preview"
              value={
                frameTelemetry?.frameWidth && frameTelemetry.frameHeight
                  ? `${frameTelemetry.frameWidth} × ${frameTelemetry.frameHeight}`
                  : "—"
              }
            />
            <Metric
              label="Median"
              value={
                active?.median_latency_ms != null
                  ? `${active.median_latency_ms.toFixed(1)} ms`
                  : "—"
              }
              detail="up to last 4,096 frames"
            />
            <Metric
              label="p95"
              value={
                active?.p95_latency_ms != null
                  ? `${active.p95_latency_ms.toFixed(1)} ms`
                  : "—"
              }
              detail="up to last 4,096 frames"
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
            <Metric
              label="Peak VRAM"
              value={
                active?.peak_vram_mb != null
                  ? `${active.peak_vram_mb.toFixed(0)} MB`
                  : "—"
              }
            />
            <Metric label="Device" value={active?.actual_device ?? "—"} />
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

            {active?.provider_reason ? (
              <p className="provider-reason">{active.provider_reason}</p>
            ) : null}
            <label>
              <span>Model</span>
              <select
                aria-label="Model"
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
                aria-label="Source"
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
            <button
              className="secondary preview-button"
              type="button"
              disabled={!selectedSource || busy !== null}
              onClick={() =>
                void act("preview", () => onPreview(selectedSource))
              }
            >
              {busy === "preview" ? "Opening…" : "Preview source"}
            </button>
            <label>
              <span>Compute device</span>
              <select
                aria-label="Compute device"
                value={computeDevice}
                onChange={(event) =>
                  setComputeDevice(event.target.value as ComputeDevice)
                }
              >
                <option value="auto">Auto</option>
                <option value="nvidia">NVIDIA GPU</option>
                <option value="cpu">CPU</option>
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

            <div className="recording-toggles">
              <label className="check-row">
                <input
                  aria-label="Record data"
                  type="checkbox"
                  checked={recordData}
                  onChange={(event) => setRecordData(event.target.checked)}
                />
                <span>Record data</span>
              </label>
              <label className="check-row">
                <input
                  aria-label="Record monitor"
                  type="checkbox"
                  checked={recordMonitor}
                  onChange={(event) => setRecordMonitor(event.target.checked)}
                />
                <span>Record monitor</span>
              </label>
            </div>

            <div className="overlay-grid">
              <OverlayControls
                title="Live preview overlays"
                prefix="Preview"
                value={previewOverlays}
                capabilities={overlayCapabilities}
                onChange={setPreviewOverlays}
              />
              <OverlayControls
                title="Saved video overlays"
                prefix="Recording"
                value={recordingOverlays}
                capabilities={overlayCapabilities}
                onChange={setRecordingOverlays}
              />
            </div>
            <div className="action-row">
              <button
                className="primary"
                type="button"
                disabled={
                  !selectedModel || !selectedSource || running || busy !== null
                }
                onClick={() => void act("run", () => onStart(runOptions))}
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
            <summary>Register a source</summary>
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
                if (selectedCamera)
                  void act("camera", () => onCamera(selectedCamera));
              }}
            >
              <h4>Windows capture device</h4>
              <label>
                <span>Camera device</span>
                <select
                  value={selectedCamera}
                  onChange={(event) => setCameraDevice(event.target.value)}
                >
                  {captureDevices.length ? (
                    captureDevices.map((device) => (
                      <option key={device.id} value={device.id}>
                        {device.label}
                      </option>
                    ))
                  ) : (
                    <option value="">No camera connected</option>
                  )}
                </select>
              </label>
              <button
                className="secondary"
                disabled={!selectedCamera || busy !== null}
              >
                Connect camera
              </button>
              <button
                className="secondary"
                type="button"
                disabled={captureOwned || busy !== null}
                onClick={() =>
                  void act("camera-refresh", () => onRefreshCameras())
                }
              >
                {busy === "camera-refresh" ? "Refreshing…" : "Refresh cameras"}
              </button>
            </form>
          </details>
        </aside>
      </div>
    </section>
  );
}
