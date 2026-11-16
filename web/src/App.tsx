import { useCallback, useEffect, useState } from "react";

import { api } from "./api";
import type {
  Capabilities,
  ModelRecord,
  PublicSettings,
  SessionRecord,
  SourceRecord,
} from "./types";
import { Monitor } from "./views/Monitor";
import { Sessions } from "./views/Sessions";
import { Settings } from "./views/Settings";

type View = "monitor" | "sessions" | "settings";

interface StationData {
  models: ModelRecord[];
  sources: SourceRecord[];
  sessions: SessionRecord[];
  capabilities: Capabilities;
  settings: PublicSettings;
}

export function App() {
  const [view, setView] = useState<View>("monitor");
  const [data, setData] = useState<StationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [models, sources, sessions, capabilities, settings] =
      await Promise.all([
        api.models(),
        api.sources(),
        api.sessions(),
        api.capabilities(),
        api.settings(),
      ]);
    setData({ models, sources, sessions, capabilities, settings });
    setActiveSessionId((current) => current ?? sessions[0]?.id ?? null);
    setError(null);
  }, []);

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
      )
    )
      return;
    const timer = window.setInterval(
      () => void load().catch(() => undefined),
      500,
    );
    return () => window.clearInterval(timer);
  }, [data?.sessions, load]);

  async function refresh() {
    await load();
  }

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
          {(["monitor", "sessions", "settings"] as const).map((item) => (
            <button
              key={item}
              role="tab"
              aria-selected={view === item}
              onClick={() => setView(item)}
            >
              {item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
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
            activeSessionId={activeSessionId}
            onActiveSession={setActiveSessionId}
            onStart={async (modelId, sourceId, sampleEvery) => {
              const created = await api.createSession(
                modelId,
                sourceId,
                sampleEvery,
              );
              setActiveSessionId(created.id);
              await api.startSession(created.id);
              await refresh();
            }}
            onStop={async (id) => {
              await api.stopSession(id);
              await refresh();
            }}
            onModelUpload={async (model, metadata) => {
              await api.registerModel(model, metadata);
              await refresh();
            }}
            onVideoUpload={async (video) => {
              await api.importVideo(video);
              await refresh();
            }}
            onCamera={async (device) => {
              await api.registerCamera(device);
              await refresh();
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
        {view === "settings" ? (
          <Settings capabilities={data.capabilities} settings={data.settings} />
        ) : null}
      </main>
      <footer>
        <span>ReefVision-Edge v0.1</span>
        <span>Local-first · no cloud dependency</span>
      </footer>
    </div>
  );
}
