# ReefVision-Edge v0.1 Windows MSI design

## Intent

Ship the verified ReefVision-Edge v0.1 application as a Windows x64 MSI that installs and runs independently of the source checkout, opens the existing dashboard from a Start-menu launcher, stores state per user, and coexists with desktop ReefVision.

## Constraints

- Preserve application commit `bdc9d8795961c71cf55af244addd2c18281f2ec8` behavior and its ReefVision-derived theme.
- Reuse the verified Windows dependency set in `.artifacts/reefvision-edge-offline-v0.1-bdc9d87`.
- Bind the installed backend only to `127.0.0.1`.
- The installed program must not require Python, Node, the repository, a terminal, a dev server, or runtime downloads.
- Use distinct product, upgrade, application, install, Start-menu, and data identities from ReefVision.
- Produce `C:\Users\rafae\Downloads\ReefVision-Edge-0.1.0-windows-x64.msi`.

## Selected approach

Mirror the original ReefVision Windows architecture: a Tauri desktop launcher manages a PyInstaller onedir FastAPI backend, and WiX packages both into an MSI. The launcher uses the installed WebView2 runtime already present on supported Windows 11 systems, starts the backend hidden, waits for loopback readiness, and opens the existing dashboard in a desktop window. Closing the launcher terminates its backend. Data lives under `%LOCALAPPDATA%\ReefVision-Edge`.

Alternatives rejected:

- A browser-only launcher would not match ReefVision's desktop lifecycle and would leave a background service after the browser closed.
- A portable Python tree would be larger, less deterministic, and easier to couple accidentally to the build-machine Python installation.
- An Electron shell would introduce a second UI runtime and abandon the original ReefVision packaging reference.

## Components

- `packaging/windows/backend_entry.py`: PyInstaller entry point for the existing CLI.
- `packaging/windows/launcher/`: minimal Tauri launcher with a distinct identifier and icon.
- `packaging/windows/wix/`: per-user WiX definition with independent ARP and Start-menu entries.
- `scripts/build_windows_msi.ps1`: reproducible backend, launcher, payload, MSI, and verification build.
- `tests/test_windows_packaging.py`: static and executable packaging contract tests.

## Acceptance

The MSI installs under `%LOCALAPPDATA%\Programs\ReefVision-Edge`, registers separately in Installed Apps, adds a ReefVision-Edge Start-menu shortcut, launches with no console, serves only on loopback, completes a recorded-video inference using installed code, exports a valid ZIP, and continues to work with the source checkout temporarily unavailable.

## Non-goals

- No Jarvis/Linux changes, driver or firewall changes, remote publishing, auto-update service, file associations, or replacement of the existing ReefVision installation.
