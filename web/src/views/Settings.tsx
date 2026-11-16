import type { Capabilities, PublicSettings } from "../types";

export function Settings({
  capabilities,
  settings,
}: {
  capabilities: Capabilities;
  settings: PublicSettings;
}) {
  return (
    <section className="view" aria-labelledby="settings-title">
      <div className="view-heading">
        <div>
          <p className="eyebrow">Station configuration</p>
          <h2 id="settings-title">Station settings</h2>
          <p>Runtime facts are read-only here; deployment settings come from the service environment.</p>
        </div>
        <span className="connection-state"><span className="connection-light" />Offline ready</span>
      </div>
      <div className="settings-grid">
        <article className="panel settings-card">
          <span className="panel-kicker">Network</span><h3>{settings.auth_required ? "Authenticated LAN" : "Loopback only"}</h3>
          <dl><div><dt>Bind address</dt><dd>{settings.bind_host}:{settings.bind_port}</dd></div><div><dt>Authentication</dt><dd>{settings.auth_required ? "Required" : "Not required on loopback"}</dd></div></dl>
          <p className="card-note">Use an SSH tunnel for laptop access. LAN binding refuses startup without credentials.</p>
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Inference providers</span><h3>Available runtime</h3>
          <ul className="provider-list">{capabilities.available_providers.map((provider) => <li key={provider}><span className={provider === "CPUExecutionProvider" ? "provider-state active" : "provider-state"} />{provider}</li>)}</ul>
          <p className="card-note">Provider selection is recorded per session. CPU is the portable baseline.</p>
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Local storage</span><h3>Managed data</h3>
          <code>{settings.data_dir}</code>
          <p className="card-note">Imported sources are copied intact; models, SQLite records, previews, and exports stay local.</p>
        </article>
        <article className="panel settings-card">
          <span className="panel-kicker">Capture</span><h3>{capabilities.camera_backend === "V4L2" ? "V4L2 available" : "Camera unavailable"}</h3>
          <p className="card-note">Live capture requires a Linux UVC/V4L2 device. Hardware verification is reported separately.</p>
        </article>
      </div>
    </section>
  );
}
