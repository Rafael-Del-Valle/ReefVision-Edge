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

- Backend: 135 tests pass and 4 opt-in Windows tests skip in the default run. The focused packaged model-import and frozen-backend acceptance rerun also passes all 4 gated tests.
- Frontend: 18 Vitest component tests pass.
- TypeScript: clean project typecheck.
- Production dashboard: Vite build succeeds with bundled local assets.
- Browser: 3 Playwright scenarios pass against the running production app: complete upload/run/reload/export workflow, computed theme parity, and screenshot capture.
- Recorded fixture: AVI/MJPG, 160×90, 10 FPS, 5 frames. Sampling every two frames persisted indices `0, 2, 4` and timestamps `0, 200, 400 ms`.
- Deterministic ONNX fixture: fixed 640×640 input and two known detections; the known coral box maps back to original coordinates `(60, 35, 100, 55)` on the 160×90 fixture.
- Restart: a completed session and its persisted latest-frame preview were opened and exported from a new FastAPI application instance using the same SQLite/data directory.
- Security: non-loopback startup without credentials is rejected; authenticated LAN middleware covers UI, API, previews, and exports. Loopback mode rejects untrusted Host headers and cross-origin mutations.

## Windows RTX 3090 and Monitor acceptance

- Tested commit: `f93308e7e374d28e9cadccc4b391c57ea53e62de`.
- Offline bundle: `C:\Users\rafae\Documents\ChatGPT\ReefVision-Edge\.artifacts\reefvision-edge-offline-windows-cuda-v0.1-f93308e` (1,324,945,814 bytes). Its metadata identifies `windows-x64`, runtime profile `cuda`, and the tested commit; every pinned runtime wheel is hash-locked. Frozen-backend diagnostics reported the same commit and runtime profile before acceptance ran.
- Frozen backend: `.artifacts\windows-msi\backend\reefvision-edge-backend\reefvision-edge-backend.exe`. It ran without the source checkout or runtime downloads and listened on loopback only.
- Explicit NVIDIA GPU mode selected `NVIDIA GeForce RTX 3090` and `CUDAExecutionProvider`. Production code now accepts explicit GPU only after profiling the first inference and observing at least one CUDA-executed operator. A second packaged backend was run with CUDA hidden: explicit NVIDIA failed with an actionable zero-CUDA error, while Auto completed on `CPUExecutionProvider` and persisted the visible fallback reason.
- ONNX Runtime profiling recorded CUDA execution for `Gemm`, `ReduceMean`, `Reshape`, and `Unsqueeze`. Both the short parity run and the sustained run recorded four profiled CUDA operator events and no CPU operator events; profiling is intentionally ended after the first inference because it is the production acceptance gate.
- CPU/GPU parity passed on the same model and 12-frame clip: class IDs matched exactly, maximum confidence delta was `0.00000012` (tolerance `0.0001`), and maximum geometry delta was `0.0 px` (tolerance `0.5 px`).
- Independent recording passed in all three combinations: data only, monitor video only, and both. The data export persisted timestamps, frame status, class, confidence, and geometry. Both 12-frame monitor recordings decoded at `160×90`.
- Sustained replay with Record data and Record monitor enabled ran for `60.172 s`: input `120.936 FPS`, inference `653.195 FPS`, rolling median inference latency `1.020 ms`, rolling p95 `5.011 ms`, `0` dropped frames, 7,264 frame rows, and 14,528 detection rows. The reopened monitor video decoded all 7,264 frames at `160×90`. Median and p95 use the most recent 4,096 inferred frames, are recomputed at finalization, and are labeled with that window in Monitor and the export manifest.
- Peak process VRAM is reported as unavailable. Windows WDDM returned `[N/A]` for per-process compute memory on this workstation; the app no longer substitutes or displays total system-wide GPU memory as though it belonged to ReefVision-Edge.
- The packaged Monitor UI visibly showed the CUDA provider, RTX 3090, performance/drop metrics, separate Record data and Record monitor controls, separate live/saved overlay settings, and the unsupported trails/masks explanations. It also showed `No camera connected` in Windows capture-device discovery.
- Timer, boxes, and labels are supported independently for live preview and saved video. Trails remain disabled until tracker IDs exist; masks remain disabled until the active model outputs segmentation masks.
- GPU video encoding was not required for acceptance. The packaged software MP4 path remained enabled and its saved clips passed decode and overlay-manifest checks.
- Acceptance artifacts: `.artifacts\windows-gpu-acceptance-final-f93308e\result.json`, `.artifacts\windows-gpu-acceptance-final-f93308e\fallback\fallback-result.json`, `.artifacts\windows-gpu-acceptance-final-f93308e\sustained.zip`, exported CSV files, ONNX Runtime operator profile, and decoded monitor video.

## Trusted checkpoint import and thesis tiling acceptance

- Tested application commit: `cc9f6c0820b48072bf3bbe6108875abc992d5c1f`.
- Offline Windows CUDA bundle: `C:\Users\rafae\Documents\ChatGPT\ReefVision-Edge\.artifacts\reefvision-edge-offline-v0.1-cc9f6c0`. Its metadata identifies `windows-x64`, runtime profile `cuda`, and the exact tested commit. The bundle contains 374 files totaling 1,609,952,356 bytes; `SHA256SUMS.txt` has SHA-256 `0B20FD63C37063DE1B9A23AA61E1BD24F9958FD0FD03389EE09D10B38DCB2CF2`.
- Models now provides an explicit trusted-file workflow for Ultralytics YOLOv8 detection `.pt` checkpoints. Inspection and conversion run in an isolated child process; review of the detected task and ordered class labels precedes a separate confirmation and conversion action. Unsupported tasks and architectures return actionable compatibility errors. Existing direct ONNX registration remains available.
- The bundled converter is pinned to 48 hash-locked CPython 3.12 Windows x64 wheels. Its generated license inventory records the declared license and redistribution review for every bundled dependency and includes 278 packaged license files. Ultralytics and the project are identified as `AGPL-3.0-only`; distributable builds include the corresponding notices and source-offer information.
- The converter preserves the source checkpoint and optional configuration next to the derived model in the local registry. It records source, configuration, and ONNX hashes; ordered class names; input shape; exporter, Python, PyTorch, ONNX, and runtime versions; preprocessing; thresholds; and output contract. Export is static-batch FP32 ONNX, opset 17, `dynamic=false`, `simplify=false`, `nms=false`, with external application NMS. Research checkpoints and media remain excluded from Git and release payloads.
- Synthetic checkpoint: six ordered coral-state classes, 320×320 model input, `rect=false`; source SHA-256 `ADD1A93D1856E96571D2088E46FC05F15C43023EFDF3E3E581B7EF1D0835E9C5`, derived ONNX SHA-256 `C78D9F2C193A985AA0CC3C1DEAA8A074C555C7F6147D0B5CC4900138B828451F`. The frozen converter reported Ultralytics `8.4.215`, PyTorch `2.15.0+cpu`, ONNX `1.23.2`, and Python `3.12.10`.
- Matched PyTorch/ONNX parity on the synthetic frames passed: 4,500 detections, exact class IDs and per-frame/per-class counts, maximum confidence delta `0.00001253` (tolerance `0.01`), and maximum source-coordinate box delta `0.000818 px` (tolerance `1 px`). The synthetic parity fixture uses IoU `1.0` so NMS is deliberately a no-op; class-aware duplicate merging is covered independently by tiling unit tests and awaits validation against the real thesis settings.
- Explicit NVIDIA GPU mode selected the RTX 3090 and `CUDAExecutionProvider`. ONNX Runtime profiling recorded 175 CUDA operator events and no CPU operator events in both full-frame and tiled runs. CUDA-executed operator types included convolution, resize, activation, pooling, concatenation, reshape, transpose, split, softmax, and arithmetic operators.
- Full-frame replay recorded data and monitor video together for 45 frames at 1280×720: input `20.452 FPS`, inference `64.543 FPS`, median latency `8.858 ms`, p95 `10.619 ms`, and 0 dropped frames. The exported CSV contained 4,500 detections; the saved monitor clip decoded to 45 frames at 1280×720.
- Thesis tiling keeps 1920×1344 source tiles distinct from the 320×320 model input, uses 40% overlap, restores boxes to full-frame coordinates, and merges duplicates by class. The actual multi-tile 2304×1536 replay recorded 18 frames: input `5.962 FPS`, inference `24.705 FPS`, median latency `34.904 ms`, p95 `55.244 ms`, and 0 dropped frames. Its CSV contained 7,200 detections; its saved video decoded to 18 frames at 2304×1536.
- Both runs saved the imported class names to the detection CSV and exercised the same independent recording controls as direct ONNX models. Live timer was off while the saved-video timer was on; boxes and labels were enabled. Peak per-process VRAM remains unavailable because Windows WDDM did not expose it.
- Acceptance report: `.artifacts\synthetic-model-import-acceptance\outputs-cc9f6c0\result.json`; full-frame CSV/video/ZIP are under its `full-frame` folder, and tiled CSV/video/ZIP are under `thesis-tiling`.
- Real-model acceptance remains open only for user-supplied thesis artifacts: the actual `best.pt`, matching `args.yaml` or `data.yaml` when available, and representative coral images or a short clip. That run must confirm the checkpoint's class order, fill any missing resize/threshold/overlap-merge settings from the thesis workflow, and pass PyTorch-versus-ONNX parity plus full-frame and tiled RTX 3090 recording acceptance.

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
