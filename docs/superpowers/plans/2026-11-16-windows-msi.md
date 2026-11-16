# ReefVision-Edge v0.1 Windows MSI implementation plan

**Goal:** Package, install, and acceptance-test the verified ReefVision-Edge v0.1 application as an independent Windows x64 MSI.

**Architecture:** A Tauri launcher owns a hidden PyInstaller backend process and a WebView window pointed at its loopback dashboard. A per-user WiX MSI installs the launcher/resources and Start-menu shortcut under a product identity distinct from ReefVision.

**Tech stack:** Python 3.12, PyInstaller onedir, Rust/Tauri 2, WiX 3.14, PowerShell build orchestration, pytest, Playwright/HTTP acceptance.

## Task 1: Installer contracts and launcher tests

**Files:** Create `tests/test_windows_packaging.py`, `packaging/windows/launcher/src/main.rs`, `packaging/windows/launcher/Cargo.toml`, `packaging/windows/launcher/build.rs`, `packaging/windows/launcher/tauri.conf.json`, `packaging/windows/backend_entry.py`.

- [ ] Write failing pytest assertions for unique product identity, loopback URL, per-user data/install paths, Start-menu shortcut, hidden backend, and required packaged static assets.
- [ ] Write Rust unit tests for loopback URL, `%LOCALAPPDATA%\ReefVision-Edge`, backend resource resolution, and startup timeout behavior; confirm they fail before implementation.
- [ ] Implement the minimal launcher/backend entry point and make focused tests pass.

## Task 2: Self-contained backend payload

**Files:** Create `packaging/windows/reefvision-edge-backend.spec`; modify `pyproject.toml`; create `scripts/build_windows_msi.ps1`.

- [ ] Add a failing packaged-backend smoke that requires `--version`, `/api/health`, the dashboard HTML, and no dependency on the checkout.
- [ ] Install the pinned build-only PyInstaller tool, build from the verified environment and application wheel, and stage the onedir output.
- [ ] Move the checkout temporarily during the smoke test and confirm the staged backend still serves health/dashboard from loopback.

## Task 3: MSI and Start-menu packaging

**Files:** Create `packaging/windows/wix/main.wxs`, `packaging/windows/icon.svg`, generated launcher icons; modify `scripts/build_windows_msi.ps1`.

- [ ] Write failing MSI table checks for product/version/manufacturer, per-user install directory, distinct upgrade code, ARP entry, and Start-menu shortcut.
- [ ] Build the release launcher and harvest the backend resources with WiX.
- [ ] Link `ReefVision-Edge-0.1.0-windows-x64.msi`, verify its payload/tables, and copy it directly to Windows Downloads.

## Task 4: Installed-build acceptance

**Files:** Create `scripts/acceptance_windows_installed.ps1`; update `README.md`, `docs/verification-report.md`.

- [ ] Install the MSI and verify the registered product, install location, executable, resources, shortcut, and per-user data isolation.
- [ ] Launch from the Start menu, verify no terminal appears and the themed dashboard loads.
- [ ] Use installed endpoints with the deterministic ONNX/video fixtures to register, run, persist, preview, and export a session; validate the export with installed code.
- [ ] Temporarily make the source checkout unavailable and repeat launch/health verification.
- [ ] Inspect Installed Apps for separate ReefVision and ReefVision-Edge entries.
- [ ] Open Downloads with the final MSI selected and record exact results.

## Review focus

- Product/upgrade codes must never collide with ReefVision.
- Launcher must never bind beyond loopback or inherit dev data paths.
- PyInstaller must include ONNX Runtime, OpenCV, Pydantic native modules, dashboard assets, and license data.
- MSI install/uninstall must not modify ReefVision files, registry entries, or data.
- Installed smoke must execute the installed binaries, not repository code.
