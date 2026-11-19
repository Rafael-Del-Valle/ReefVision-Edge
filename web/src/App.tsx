import { useCallback, useEffect, useState } from "react";

import { api } from "./api";
import type {
  Capabilities,
  CaptureDevice,
  FieldSwitchPreferences,
  ModelRecord,
  MonitorStatus,
  PublicSettings,
  SessionRecord,
  SessionCreateOptions,
  SourceRecord,
} from "./types";
import { Monitor } from "./views/Monitor";
import { Models } from "./views/Models";
import { Sessions } from "./views/Sessions";
import { Settings } from "./views/Settings";

type View = "monitor" | "models" | "sessions" | "settings";

interface StationData {
  models: ModelRecord[];
  sources: SourceRecord[];
  sessions: SessionRecord[];
  capabilities: Capabilities;
  settings: PublicSettings;
  fieldSwitch: FieldSwitchPreferences;
  captureDevices: CaptureDevice[];
  monitorStatus: MonitorStatus;
}

function upsertById<T extends { id: string }>(values: T[], value: T): T[] {
  return [value, ...values.filter((item) => item.id !== value.id)];
}

function captureIsOwned(
  sessions: SessionRecord[],
  sources: SourceRecord[],
  monitorStatus: MonitorStatus,
): boolean {
  if (["connecting", "receiving", "stalled"].includes(monitorStatus.state))
    return true;
  const cameraIds = new Set(
    sources
      .filter((source) => source.kind === "camera")
      .map((source) => source.id),
  );
  const activeCameraSessions = sessions.filter(
    (session) =>
      ["queued", "running"].includes(session.status) &&
      cameraIds.has(session.source_id),
  );
  if (monitorStatus.state === "disconnected")
    return activeCameraSessions.some(
      (session) => session.source_id !== monitorStatus.source_id,
    );
  return activeCameraSessions.length > 0;
}

export function App() {
  const [view, setView] = useState<View>("monitor");
  const [data, setData] = useState<StationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [models, sources, sessions, capabilities, settings, fieldSwitch, monitorStatus] =
      await Promise.all([
        api.models(),
        api.sources(),
        api.sessions(),
        api.capabilities(),
        api.settings(),
        api.fieldSwitchConfig(),
        api.monitorStatus(),
      ]);
    const captureDevices = captureIsOwned(sessions, sources, monitorStatus)
      ? []
      : await api.captureDevices();
    setData({
      models,
      sources,
      sessions,
      capabilities,
      settings,
      fieldSwitch,
      captureDevices,
      monitorStatus,
    });
    setActiveSessionId((current) => current ?? sessions[0]?.id ?? null);
    setError(null);
  }, []);

  const pollActiveState = useCallback(async () => {
    const [sessions, monitorStatus] = await Promise.all([
      api.sessions(),
      api.monitorStatus(),
    ]);
    setData((current) =>
      current ? { ...current, sessions, monitorStatus } : current,
    );
  }, []);

  const refreshCameras = useCallback(async () => {
    if (
      !data ||
      captureIsOwned(data.sessions, data.sources, data.monitorStatus)
    )
      return;
    const captureDevices = await api.captureDevices();
    setData((current) => (current ? { ...current, captureDevices } : current));
  }, [data]);

  useEffect(() => {
    let active = true;
    void load().catch((reason: unknown) => {
      if (active)
        setError(reason instanceof Error ? reason.message : "Unknown error");
    });
    return () => {
      active = false;
    };
  }, [load, retry]);

  useEffect(() => {
    if (
      !data?.sessions.some((session) =>
        ["queued", "running"].includes(session.status),
      ) &&
      !["connecting", "receiving", "replaying"].includes(
        data?.monitorStatus.state ?? "idle",
      )
    )
      return;
    const timer = window.setInterval(
      () => void pollActiveState().catch(() => undefined),
      500,
    );
    return () => window.clearInterval(timer);
  }, [data?.sessions, data?.monitorStatus.state, pollActiveState]);

  if (error || !data) {
    return (
      <main className="startup-shell">
        <div className="startup-card panel">
          <div className="brand-mark">RV</div>
          <p className="eyebrow">ReefVision-Edge</p>
          {error ? (
            <>
              <h1 role="alert">Could not load station data</h1>
              <p>{error}</p>
              <button
                className="primary"
                onClick={() => setRetry((value) => value + 1)}
              >
                Retry
              </button>
            </>
          ) : (
            <>
              <h1>Connecting to local service</h1>
              <p>No network connection is required.</p>
              <div className="loading-line" />
            </>
          )}
        </div>
      </main>
    );
  }

  const openSession = (id: string) => {
    setActiveSessionId(id);
    setView("monitor");
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">RV</div>
          <div>
            <strong>ReefVision-Edge</strong>
            <span>Jarvis field station</span>
          </div>
        </div>
        <nav role="tablist" aria-label="Primary views">
          {(["monitor", "models", "sessions", "settings"] as const).map(
            (item) => (
              <button
                key={item}
                role="tab"
                aria-selected={view === item}
                onClick={() => setView(item)}
              >
                {item[0].toUpperCase() + item.slice(1)}
              </button>
            ),
          )}
        </nav>
        <div className="topbar-status">
          <span className="connection-light" />
          Offline ready
        </div>
      </header>
      <main>
        {view === "monitor" ? (
          <Monitor
            models={data.models}
            sources={data.sources}
            sessions={data.sessions}
            captureDevices={data.captureDevices}
            monitorStatus={data.monitorStatus}
            captureOwned={captureIsOwned(
              data.sessions,
              data.sources,
              data.monitorStatus,
            )}
            overlayCapabilities={data.capabilities.overlay_capabilities}
            activeSessionId={activeSessionId}
            onActiveSession={setActiveSessionId}
            onStart={async (options: SessionCreateOptions) => {
              const created = await api.createSession(options);
              setActiveSessionId(created.id);
              await api.startSession(created.id);
              await pollActiveState();
            }}
            onStop={async (id) => {
              await api.stopSession(id);
              await pollActiveState();
            }}
            onVideoUpload={async (video) => {
              const imported = await api.importVideo(video);
              setData((current) =>
                current
                  ? {
                      ...current,
                      sources: upsertById(current.sources, imported),
                    }
                  : current,
              );
            }}
            onCamera={async (device) => {
              const registered = await api.registerCamera(device);
              setData((current) =>
                current
                  ? {
                      ...current,
                      sources: upsertById(current.sources, registered),
                    }
                  : current,
              );
            }}
            onRefreshCameras={refreshCameras}
            onPreview={async (sourceId) => {
              await api.selectMonitor(sourceId);
              await pollActiveState();
            }}
          />
        ) : null}
        {view === "sessions" ? (
          <Sessions
            sessions={data.sessions}
            models={data.models}
            sources={data.sources}
            onOpen={openSession}
          />
        ) : null}
        {view === "models" ? (
          <Models
            models={data.models}
            onModelRegistered={async (registered) => {
              const models = registered
                ? upsertById(data.models, registered)
                : await api.models();
              setData((current) =>
                current ? { ...current, models } : current,
              );
            }}
          />
        ) : null}
        {view === "settings" ? (
          <Settings
            capabilities={data.capabilities}
            settings={data.settings}
            fieldSwitch={data.fieldSwitch}
            onFieldSwitchChange={async (preferences) => {
              const saved = await api.saveFieldSwitchConfig(preferences);
              setData((current) =>
                current ? { ...current, fieldSwitch: saved } : current,
              );
            }}
          />
        ) : null}
      </main>
      <footer>
        <span>ReefVision-Edge v0.1</span>
        <span>Local-first · no cloud dependency</span>
      </footer>
    </div>
  );
}
