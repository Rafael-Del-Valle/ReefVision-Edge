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
