# ReefVision-Edge v0.1 verification report

Date: 2026-11-16 (America/New_York)

## Reference audit

- Desktop ReefVision baseline: `e94dad6a3bece7b776bf45776bc5d06cfbc2c0dc`.
- Stylesheet order: `clarity-overhaul.css` from `main.tsx`, followed by `design-system.css` and `app.css` from `App.tsx`.
- Exact semantic theme tokens were recovered and asserted in the production dashboard.
- Desktop annotation imports accept image-associated CoralNet, COCO, and YOLO data. No pinned contract accepts frame-level ReefVision-Edge session detections, so desktop compatibility is not claimed.
- Desktop training produces local Ultralytics checkpoints; Edge documents an explicit local Ultralytics ONNX export step and validates the result.

## Verified host

- OS: Microsoft Windows 11 Home, version `10.0.26300`, build `26300`, 64-bit.
- CPU: AMD Ryzen Threadripper PRO 9955WX, 16 cores / 32 logical processors.
- GPU present: NVIDIA GeForce RTX 3090, driver `32.0.16.1742`; this does not imply ONNX acceleration availability.
- Python `3.12.10`; Node `v26.13.0`; npm `11.20.0`.
- NumPy `2.2.6`; OpenCV `4.14.0` with FFmpeg enabled; ONNX Runtime `1.32.0`.
- ONNX Runtime providers discovered: `AzureExecutionProvider`, `CPUExecutionProvider`.
- Provider actually selected and verified: `CPUExecutionProvider`.
- System `ffmpeg` command: unavailable. Decoder evidence comes from the OpenCV build and actual synthetic fixture run.

Windows Application Control blocked NumPy `2.5.5` on this host. NumPy `2.2.6` loaded successfully, so the project caps NumPy below `2.3` until newer builds are requalified.

## Automated evidence

- Backend: 46 tests pass, including model/metadata rejection and upload limits, run-time provider reselection and graph-assignment/fallback reporting, deterministic ONNX inference, letterbox reversal/clipping, media copying/probing, early decoder failure, unavailable camera, bounded work/cancellation/shutdown, atomic counters, stale-run reconciliation, simulated disk/database-full failure, zero-detection export round trip, API workflow, packaged assets, CLI LAN safety, and fresh-app session/preview reopen/export.
- Frontend: 4 Vitest component tests pass.
- TypeScript: clean project typecheck.
- Production dashboard: Vite build succeeds with bundled local assets.
- Browser: 3 Playwright scenarios pass against the running production app: complete upload/run/reload/export workflow, computed theme parity, and screenshot capture.
- Recorded fixture: AVI/MJPG, 160×90, 10 FPS, 5 frames. Sampling every two frames persisted indices `0, 2, 4` and timestamps `0, 200, 400 ms`.
- Deterministic ONNX fixture: fixed 640×640 input and two known detections; the known coral box maps back to original coordinates `(60, 35, 100, 55)` on the 160×90 fixture.
- Restart: a completed session and its persisted latest-frame preview were opened and exported from a new FastAPI application instance using the same SQLite/data directory.
- Security: non-loopback startup without credentials is rejected; authenticated LAN middleware covers UI, API, previews, and exports. Loopback mode rejects untrusted Host headers and cross-origin mutations.

## Installed Windows MSI acceptance

- Package: `C:\Users\rafae\Downloads\ReefVision-Edge-0.1.0-windows-x64.msi` (286,833,390 bytes), SHA-256 `DD958390200A81012A8DCACDA6CB5A834750790DDDA7A4F6197391B712D261CF`.
- Installed product: `ReefVision-Edge` version `0.1.0`, publisher `ReefVision`, product code `{14B14561-3052-42F7-8C46-D593E91FCD0D}`.
- Independent identity: upgrade code `{C2F68AF2-90BF-4CD5-982C-42987B44F5E0}`; the reference ReefVision MSI uses `{5EDA8EE0-2A5D-5B08-A8DF-9BECFC3A4351}`.
- Application: `%LOCALAPPDATA%\Programs\ReefVision-Edge\ReefVision-Edge.exe`; service payload: `backend\reefvision-edge-backend\reefvision-edge-backend.exe`.
- Data: `%LOCALAPPDATA%\ReefVision-Edge`; Start-menu shortcut: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\ReefVision-Edge\ReefVision-Edge.lnk`.
- Start-menu launch visibly opened the ReefVision-themed dashboard without a terminal. The running service executable resolved to the installed backend and listened only on `127.0.0.1:8765`.
- Deterministic installed inference completed session `c71405dc-23f2-4c5f-86ea-ec561f89d44f`: a newly uploaded model and video produced 3 sampled frames, 6 detections, `CPUExecutionProvider`, a persisted preview, and a validated `reefvision-edge-session/v1` ZIP containing `manifest.json` and `detections.csv`. The acceptance script rejects reused registration/source IDs and verifies both uploaded SHA-256 values.
- The launcher was closed, the checkout's `src` directory was moved out of reach, and the same Start-menu shortcut relaunched successfully. Health still reported ReefVision-Edge `0.1.0` from the installed backend; the checkout was then restored.
- A quiet uninstall/reinstall isolation cycle preserved temporary ReefVision file and registry sentinels plus the ReefVision-Edge per-user data marker, removed the Edge executable/shortcut/Installed Apps registration, and then restored the final installed product.
- Forced termination of the launcher closed its Windows Job Object and terminated the backend. Relaunch from the Start shortcut again returned the expected product/version from the installed executable on `127.0.0.1`.
- The embedded offline WebView2 payload is now an explicit build input pinned to SHA-256 `D3CBE8FEDB104282D745FFF106993EFEAD51AE94F5595A4E75804446A974F9C3`; a missing or different payload stops the build.
- Windows Installed Apps visibly showed `ReefVision-Edge` version `0.1.0` as its own entry. The original ReefVision product was not installed on this host, so the search correctly showed only the Edge entry; simultaneous listing could not be observed, while the distinct product/upgrade codes, install directory, shortcut folder, and data directory were verified.
- File Explorer was left open to Downloads with the final MSI selected.

Exact final command output is recorded in the Git task history and final delivery report.

## Screenshot review

Monitor, Sessions, and Settings were rendered at 1440×900 and 390×844. The reviewed images show readable hierarchy, no horizontal clipping, accessible controls, visible provider/state, original-frame overlays, and exact ReefVision semantic colors. Files are under `docs/screenshots/`.

## Not verified on this host

- Jarvis hardware model, ARM64 CPU, OS, JetPack/Jetson Linux version, storage, and resource behavior.
- TensorRT/CUDA ONNX Runtime providers on Jetson.
- Native systemd startup on Jarvis.
- Sustained recorded-video performance and thermal/resource behavior on Jarvis.
- Live UVC/V4L2 capture, camera unplug behavior on physical hardware, and USB HDMI capture.
- DJI Mini 3/RM330 or Sony A7RIII real footage codecs/containers.
- Direct DJI controller streaming and Sony live capture.
- Biological accuracy of any real coral model.
- Desktop ReefVision import of Edge session exports; no matching pinned import contract exists.

These items must remain labeled unverified until run on the named equipment. No flashing, driver upgrade, reboot, firewall change, remote repository creation, push, or release publication was performed.
