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
- GPU: NVIDIA GeForce RTX 3090 (24 GB), NVIDIA driver `617.42`, CUDA driver API `13.4`.
- Python `3.12.10`; Node `v26.13.0`; npm `11.20.0`.
- NumPy `2.2.6`; OpenCV `4.14.0` with FFmpeg enabled; ONNX Runtime `1.32.0`.
- Packaged Windows CUDA runtime: ONNX Runtime GPU `1.32.0`, CUDA runtime `13.4.92`, cuDNN `9.28.1.4`, cuBLAS `13.4.1.3`.
- Packaged providers discovered: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`. TensorRT is not packaged or accepted; actual inference was verified on `CUDAExecutionProvider`.
- System `ffmpeg` command: unavailable. Decoder evidence comes from the OpenCV build and actual synthetic fixture run.

Windows Application Control blocked NumPy `2.5.5` on this host. NumPy `2.2.6` loaded successfully, so the project caps NumPy below `2.3` until newer builds are requalified.

## Automated evidence

- Backend: 91 tests pass and 3 opt-in Windows tests skip in the default run. The focused packaged-backend test also passes against the frozen executable with the checkout source moved out of reach.
- Frontend: 5 Vitest component tests pass.
- TypeScript: clean project typecheck.
- Production dashboard: Vite build succeeds with bundled local assets.
- Browser: 3 Playwright scenarios pass against the running production app: complete upload/run/reload/export workflow, computed theme parity, and screenshot capture.
- Recorded fixture: AVI/MJPG, 160×90, 10 FPS, 5 frames. Sampling every two frames persisted indices `0, 2, 4` and timestamps `0, 200, 400 ms`.
- Deterministic ONNX fixture: fixed 640×640 input and two known detections; the known coral box maps back to original coordinates `(60, 35, 100, 55)` on the 160×90 fixture.
- Restart: a completed session and its persisted latest-frame preview were opened and exported from a new FastAPI application instance using the same SQLite/data directory.
- Security: non-loopback startup without credentials is rejected; authenticated LAN middleware covers UI, API, previews, and exports. Loopback mode rejects untrusted Host headers and cross-origin mutations.

## Windows RTX 3090 and Monitor acceptance

- Tested commit: `ab4b94d262d5df04ac2c61f50795bd98b5b69cf0`.
- Offline bundle: `C:\Users\rafae\Documents\ChatGPT\ReefVision-Edge\.artifacts\reefvision-edge-offline-windows-cuda-v0.1-ab4b94d` (1,324,565,177 bytes). Its metadata identifies `windows-x64`, runtime profile `cuda`, and the tested commit; every pinned runtime wheel is hash-locked. Frozen-backend diagnostics reported the same commit and runtime profile before acceptance ran.
- Frozen backend: `.artifacts\windows-msi\backend\reefvision-edge-backend\reefvision-edge-backend.exe`. It ran without the source checkout or runtime downloads and listened on loopback only.
- Explicit NVIDIA GPU mode selected `NVIDIA GeForce RTX 3090` and `CUDAExecutionProvider`. Production code now accepts explicit GPU only after profiling the first inference and observing at least one CUDA-executed operator. A second packaged backend was run with CUDA hidden: explicit NVIDIA failed with an actionable zero-CUDA error, while Auto completed on `CPUExecutionProvider` and persisted the visible fallback reason.
- ONNX Runtime profiling recorded CUDA execution for `Gemm`, `ReduceMean`, `Reshape`, and `Unsqueeze`. Both the short parity run and the sustained run recorded four profiled CUDA operator events and no CPU operator events; profiling is intentionally ended after the first inference because it is the production acceptance gate.
- CPU/GPU parity passed on the same model and 12-frame clip: class IDs matched exactly, maximum confidence delta was `0.00000012` (tolerance `0.0001`), and maximum geometry delta was `0.0 px` (tolerance `0.5 px`).
- Independent recording passed in all three combinations: data only, monitor video only, and both. The data export persisted timestamps, frame status, class, confidence, and geometry. Both 12-frame monitor recordings decoded at `160×90`.
- Sustained replay with Record data and Record monitor enabled ran for `60.234 s`: input `126.348 FPS`, inference `742.301 FPS`, median inference latency `0.980 ms`, p95 `4.920 ms`, `0` dropped frames, 7,594 frame rows, and 15,188 detection rows. The reopened monitor video decoded all 7,594 frames at `160×90`.
- Peak process VRAM is reported as unavailable. Windows WDDM returned `[N/A]` for per-process compute memory on this workstation; the app no longer substitutes or displays total system-wide GPU memory as though it belonged to ReefVision-Edge.
- The packaged Monitor UI visibly showed the CUDA provider, RTX 3090, performance/drop metrics, separate Record data and Record monitor controls, separate live/saved overlay settings, and the unsupported trails/masks explanations. It also showed `No camera connected` in Windows capture-device discovery.
- Timer, boxes, and labels are supported independently for live preview and saved video. Trails remain disabled until tracker IDs exist; masks remain disabled until the active model outputs segmentation masks.
- GPU video encoding was not required for acceptance. The packaged software MP4 path remained enabled and its saved clips passed decode and overlay-manifest checks.
- Acceptance artifacts: `.artifacts\windows-gpu-acceptance-final-ab4b94d\result.json`, `.artifacts\windows-gpu-acceptance-final-ab4b94d\fallback\fallback-result.json`, `.artifacts\windows-gpu-acceptance-final-ab4b94d\sustained.zip`, exported CSV files, ONNX Runtime operator profile, and decoded monitor video.

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
- Physical Windows camera capture and unplug/reconnect behavior (no capture camera was connected); live UVC/V4L2 capture and USB HDMI capture on Jetson.
- DJI Mini 3/RM330 or Sony A7RIII real footage codecs/containers.
- Direct DJI controller streaming and Sony live capture.
- Biological accuracy of any real coral model.
- Desktop ReefVision import of Edge session exports; no matching pinned import contract exists.

These items must remain labeled unverified until run on the named equipment. No flashing, driver upgrade, reboot, firewall change, remote repository creation, push, or release publication was performed.
