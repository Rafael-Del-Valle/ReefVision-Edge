import { useEffect, useState } from "react";

import type {
  Capabilities,
  FieldSwitchMode,
  FieldSwitchPreferences,
  PublicSettings,
} from "../types";

export function Settings({
  capabilities,
  settings,
  fieldSwitch,
  onFieldSwitchChange,
}: {
  capabilities: Capabilities;
  settings: PublicSettings;
  fieldSwitch: FieldSwitchPreferences;
  onFieldSwitchChange: (preferences: FieldSwitchPreferences) => Promise<void>;
}) {
  const [mode, setMode] = useState<FieldSwitchMode>(fieldSwitch.mode);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => setMode(fieldSwitch.mode), [fieldSwitch.mode]);

  const save = async () => {
    setSaving(true);
    setNotice(null);
    try {
      await onFieldSwitchChange({ mode });
      setNotice("Preference saved on this device for the next collection session.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not save preference.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="view" aria-labelledby="settings-title">
      <div className="view-heading">
        <div>
          <p className="eyebrow">Station configuration</p>
          <h2 id="settings-title">Station settings</h2>
          <p>
            Runtime facts are read-only here; deployment settings come from the
            service environment.
          </p>
        </div>
        <span className="connection-state">
          <span className="connection-light" />
          Offline ready
        </span>
      </div>
      <div className="settings-grid">
        <article className="panel settings-card">
          <span className="panel-kicker">Autonomous field mode</span>
          <h3>Physical switch action</h3>
          <label htmlFor="field-switch-mode">When the collection switch is turned ON</label>
          <select
            id="field-switch-mode"
            value={mode}
            disabled={saving}
            onChange={(event) => {
              setMode(event.target.value as FieldSwitchMode);
              setNotice(null);
            }}
          >
            <option value="inference_and_logging">Start inference and detection logging</option>
            <option value="logging_only">Start logging; leave inference running</option>
          </select>
          <p className="card-note">
            In logging-only mode the camera and inference continue while logging is off.
            In inference-and-logging mode, switching OFF stops both. The choice is saved
            locally and applies to the next physical switch activation, not a session
            already in progress. A Jetson hardware controller is required to operate
            the physical switch; saving this setting does not enable hardware by itself.
          </p>
          <button
            className="primary"
            disabled={saving || mode === fieldSwitch.mode}
            onClick={() => void save()}
          >
            {saving ? "Saving..." : "Save switch behavior"}
          </button>
          {notice ? <p role="status">{notice}</p> : null}
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Network</span>
          <h3>
            {settings.auth_required ? "Authenticated LAN" : "Loopback only"}
          </h3>
          <dl>
            <div>
              <dt>Bind address</dt>
              <dd>
                {settings.bind_host}:{settings.bind_port}
              </dd>
            </div>
            <div>
              <dt>Authentication</dt>
              <dd>
                {settings.auth_required
                  ? "Required"
                  : "Not required on loopback"}
              </dd>
            </div>
          </dl>
          <p className="card-note">
            Use an SSH tunnel for laptop access. LAN binding refuses startup
            without credentials.
          </p>
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Inference providers</span>
          <h3>Available runtime</h3>
          <ul className="provider-list">
            {capabilities.available_providers.map((provider) => (
              <li key={provider}>
                <span
                  className={
                    provider === "CPUExecutionProvider"
                      ? "provider-state active"
                      : "provider-state"
                  }
                />
                {provider}
              </li>
            ))}
          </ul>
          <p className="card-note">
            Provider selection is recorded per session. CPU is the portable
            baseline.
          </p>
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Local storage</span>
          <h3>Managed data</h3>
          <code>{settings.data_dir}</code>
          <p className="card-note">
            Imported sources are copied intact; models, SQLite records,
            previews, and exports stay local.
          </p>
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Capture</span>
          <h3>
            {capabilities.camera_backend === "V4L2"
              ? "V4L2 available"
              : "Camera unavailable"}
          </h3>
          <p className="card-note">
            Live capture requires a Linux UVC/V4L2 device. Hardware verification
            is reported separately.
          </p>
        </article>
      </div>
    </section>
  );
}
