# Windows Live View, Recording, and RTX 3090 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship and prove a self-contained Windows Monitor workflow with capture discovery, independent data/video recording, and actual RTX 3090 ONNX Runtime CUDA execution.

**Architecture:** Add a runtime-profile boundary around ONNX Runtime, a Windows-aware capture/preview service, and additive session schema fields for compute, recordings, overlays, metrics, and profiles. Keep the existing local FastAPI/React structure and software video encoding, then drive a packaged acceptance harness against a deterministic local model and clip.

**Tech Stack:** Python 3.12, FastAPI, SQLite, OpenCV, ONNX Runtime GPU 1.32.0, CUDA 13.4/cuDNN 9 redistributable wheels, React/TypeScript, PyInstaller, PowerShell.

**Spec:** `docs/superpowers/specs/2026-11-16-windows-live-gpu-design.md`

## Global Constraints

- Work on `codex/reefvision-edge-v0.1` in the existing checkout.
- Keep the service loopback-only and the bundle fully offline after creation.
- Explicit NVIDIA GPU mode must not silently fall back to CPU.
- Auto mode may fall back only with a visible, persisted explanation.
- Preview and saved-video overlay settings are independent.
- Timer, boxes, and labels are supported; masks and trails are disabled with reasons until model/tracker output supports them.
- Physical-camera acceptance stays open when no capture camera is connected.
- DJI mapping, Jetson deployment, and required NVENC support are out of scope.

## Review Focus

- Missing or mismatched CUDA/cuDNN DLL: Auto uses CPU with the loader reason; NVIDIA fails before processing a frame.
- Camera index present but open/read fails: Monitor reaches `disconnected` or `error` without hanging or leaking the handle.
- Video writer cannot open or disk write fails: session fails clearly and finalizes any recoverable artifacts.
- `record_data=false`: aggregate metrics still update while frame/detection tables and CSV remain empty.
- Cancellation during active recording: writer/profile close once, session is reopenable, and partial video is decodable when frames were written.

---

### Task 1: Compute runtime selection and profiling

**Files:**
- Create: `src/reefvision_edge/compute.py`
- Modify: `src/reefvision_edge/domain.py`
- Modify: `src/reefvision_edge/inference.py`
- Modify: `src/reefvision_edge/model_registry.py`
- Test: `tests/test_compute.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces: `ComputeChoice`, `ComputeResolution`, `resolve_compute(choice)`, `parse_ort_profile(path)`, and `YoloOnnxDetector(model, compute_choice, enable_profiling)`.
- Consumes: existing `RegisteredModel`, preprocessing, and postprocessing contracts.

- [ ] Write tests proving CPU selection, Auto fallback reason, explicit NVIDIA failure, CUDA provider creation, and profile CUDA/CPU operator counts; run them and observe expected failures.
- [ ] Implement runtime DLL preloading, device discovery, provider selection, actionable errors, and JSON profile parsing.
- [ ] Adapt detector construction and model registration so registration validates on CPU while sessions choose their own compute device.
- [ ] Run `python -m pytest tests/test_compute.py tests/test_inference.py tests/test_model_registry.py -q` and the full Python suite.
- [ ] Commit `feat: add explicit Windows compute selection and profiling`.

### Task 2: Session schema, settings, and metrics

**Files:**
- Modify: `src/reefvision_edge/db.py`
- Modify: `src/reefvision_edge/domain.py`
- Modify: `src/reefvision_edge/sessions.py`
- Modify: `src/reefvision_edge/api.py`
- Test: `tests/test_database.py`
- Test: `tests/test_sessions.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `ComputeChoice`, `ComputeResolution`, detector profiling result from Task 1.
- Produces: `OverlaySettings`, expanded `SessionRecord`, and `SessionService.create(..., compute_device, record_data, record_monitor, preview_overlays, recording_overlays, enable_profiling)`.

- [ ] Write migration and session tests for additive fields, defaults, independent flags/settings, quantiles, FPS, drop counts, device/provider reason, profile summary, and the `record_data=false` behavior; run RED.
- [ ] Add idempotent schema migration and expanded API models without breaking existing databases or clients.
- [ ] Update the runner to collect latency samples, input/inference FPS, peak NVIDIA VRAM, frame status, and compute/profile details.
- [ ] Run `python -m pytest tests/test_database.py tests/test_sessions.py tests/test_api.py -q` and the full Python suite.
- [ ] Commit `feat: persist compute recording and latency metrics`.

### Task 3: Windows capture discovery and live preview

**Files:**
- Create: `src/reefvision_edge/capture_devices.py`
- Create: `src/reefvision_edge/monitor.py`
- Modify: `src/reefvision_edge/sources.py`
- Modify: `src/reefvision_edge/app.py`
- Modify: `src/reefvision_edge/api.py`
- Test: `tests/test_capture_devices.py`
- Test: `tests/test_monitor.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces: `CaptureDevice`, `discover_capture_devices()`, and `MonitorPreviewService` selection/status/JPEG endpoints.
- Consumes: media-store source records and session camera ownership rule from Task 2.

- [ ] Write tests for Media Foundation/DirectShow discovery, moving paced replay, state changes, read failure, reconnect selection, and handle release; run RED.
- [ ] Implement bounded probing and Windows camera source behavior, retaining Linux V4L2 behavior.
- [ ] Implement the monitor preview worker and API endpoints, releasing preview ownership before live session start.
- [ ] Run `python -m pytest tests/test_capture_devices.py tests/test_monitor.py tests/test_sources.py tests/test_api.py -q` and the full Python suite.
- [ ] Commit `feat: add Windows capture discovery and monitor preview`.

### Task 4: Independent monitor-video recording and overlays

**Files:**
- Create: `src/reefvision_edge/monitor_video.py`
- Modify: `src/reefvision_edge/preview.py`
- Modify: `src/reefvision_edge/sessions.py`
- Modify: `src/reefvision_edge/config.py`
- Test: `tests/test_monitor_video.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Produces: `render_monitor_frame(frame, detections, timestamp_ms, settings)` and `MonitorVideoWriter`.
- Consumes: `OverlaySettings` and session flags from Task 2.

- [ ] Write tests proving independent preview/recording rendering, supported/unavailable overlays, playable MP4, data-only, video-only, both, writer failure, and cancellation finalization; run RED.
- [ ] Extract deterministic overlay rendering and implement a software MP4 writer with safe finalization.
- [ ] Integrate data persistence and video recording independently in the session loop.
- [ ] Run `python -m pytest tests/test_monitor_video.py tests/test_sessions.py tests/test_exports.py -q` and the full Python suite.
- [ ] Commit `feat: add independent data and monitor recording`.

### Task 5: Exports and usable Monitor UI

**Files:**
- Modify: `src/reefvision_edge/exports.py`
- Modify: `docs/export-schema-v1.md`
- Modify: `web/src/App.tsx`
- Modify: `web/src/styles.css`
- Test: `tests/test_exports.py`
- Test: `web/src/App.test.tsx`

**Interfaces:**
- Consumes: expanded session/source/status/overlay APIs from Tasks 2–4.
- Produces: additive export artifacts and the complete Windows Monitor workflow.

- [ ] Write backend export tests and UI behavior tests for source selection, live states, compute choices, fallback explanation, independent recording controls, overlay availability, and session metrics; run RED.
- [ ] Export frame status CSV, optional detections/video/profile metadata, settings, device, and performance data while preserving v1 fields.
- [ ] Build the source picker, moving preview, disconnect/error UI, compute selector, independent record toggles, separate overlay panels, and actual provider/metrics display.
- [ ] Run `python -m pytest tests/test_exports.py tests/test_api.py -q`, `npm test -- --run`, `npm run build`, and the full Python suite.
- [ ] Commit `feat: build Windows monitor recording workflow`.

### Task 6: Reproducible CUDA bundle

**Files:**
- Modify: `pyproject.toml`
- Create: `packaging/windows/runtime-windows-cuda.lock`
- Create: `packaging/windows/runtime-windows-cpu.lock`
- Create: `scripts/build_windows_runtime_bundle.ps1`
- Modify: `scripts/build_offline_bundle.ps1`
- Modify: `scripts/build_windows_msi.ps1`
- Modify: `packaging/windows/reefvision-edge-backend.spec`
- Modify: `THIRD_PARTY_NOTICES.md`
- Test: `tests/test_windows_packaging.py`
- Test: `tests/test_windows_packaged_backend.py`

**Interfaces:**
- Produces: exact Windows CPU/CUDA runtime profiles and a frozen backend containing ONNX Runtime plus NVIDIA DLLs.
- Consumes: preload/provider checks from Task 1.

- [ ] Write executable packaging tests for lock resolution, hashes, bundled DLL discovery, no runtime download, and provider diagnostics; run RED.
- [ ] Split base/runtime dependencies, pin the tested Windows CUDA stack, and make the local bundle builder download/cache/hash the wheels.
- [ ] Collect ONNX Runtime and NVIDIA DLL namespaces into PyInstaller and expose bundle diagnostics.
- [ ] Run `python -m pytest tests/test_windows_packaging.py tests/test_windows_packaged_backend.py -q` and the full Python suite.
- [ ] Commit `build: bundle reproducible Windows CUDA runtime`.

### Task 7: RTX 3090 acceptance and rebuilt bundle

**Files:**
- Create: `scripts/acceptance_windows_gpu.ps1`
- Create: `tests/fixtures/build_gpu_acceptance_model.py`
- Modify: `scripts/acceptance_windows_installed.ps1`
- Modify: `docs/verification-report.md`
- Test: `tests/test_windows_gpu_acceptance.py`

**Interfaces:**
- Consumes: all prior tasks and the rebuilt local Windows bundle.
- Produces: machine-readable acceptance report, ORT profile, CSV, MP4, parity result, sustained metrics, tested commit, and bundle path.

- [ ] Write harness tests for exact class parity, `1e-4` score tolerance, `0.5`-pixel geometry tolerance, CUDA event requirement, three recording modes, 60-second duration, export reopen/decode, and overlay-manifest match; run RED.
- [ ] Implement deterministic local compute model/clip generation and the unpacked/installed acceptance harness.
- [ ] Build the CUDA runtime bundle and Windows package without downloading at application runtime.
- [ ] Run explicit CPU and NVIDIA sessions on the same clip, parse the ORT profile, then run the sustained both-recordings replay on the RTX 3090.
- [ ] Reopen the session/export, inspect CSV contents, decode MP4 frames, verify overlay settings, and write exact metrics plus open hardware checks to the report.
- [ ] Run the complete Python/UI/package suites and `scripts/acceptance_windows_gpu.ps1`; record the tested commit and bundle path.
- [ ] Commit `test: prove Windows RTX 3090 monitor workflow`.

