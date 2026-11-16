# Monitor Preview Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the Windows physical-camera Monitor smooth and current while preserving CUDA inference, CSV data, and correctly timed monitor video.

**Architecture:** A stable memoized canvas pulls the latest JPEG through a single bounded async loop while session statistics continue polling separately. Backend endpoints expose coherent frame metadata, camera preview publication is capped at the display rate, and live timestamps pace the software video writer.

**Tech Stack:** React 19, TypeScript, FastAPI, OpenCV, Python 3.12, Vitest, pytest, Windows DirectShow, ONNX Runtime CUDA.

**Spec:** `docs/superpowers/specs/2026-11-16-monitor-preview-stability-design.md`

## Global Constraints

- Work in the requested current checkout and branch `codex/reefvision-edge-v0.1`; do not create another worktree.
- Keep capture bound to the existing size-one newest-frame queue and the service bound to loopback.
- Target 30 displayed FPS and 30 FPS software monitor recording; do not add runtime downloads or NVENC requirements.
- Preview throttling must not change inference, frame CSV, detection CSV, CUDA profiling, or overlay coordinate semantics.
- Keep Windows x64 and Jetson ARM64 runtime configuration separate.

## Review Focus

- A slow frame request must not create concurrent requests or an unbounded queue; Task 3 tests one in-flight fetch.
- A 404 before the first frame must recover without replacing the canvas; Task 3 tests retry behavior.
- A source/session identity change must cancel the old pump and prevent stale draws; Task 3 tests cleanup.
- A camera stop between preview intervals must persist the newest frame; Task 1 tests forced final publication.
- Timestamp jitter or repeated timestamps must not lengthen live video; Task 2 tests literal timestamps and decoded frame count.

---

### Task 1: Coherent latest-frame publication

**Files:**
- Modify: `src/reefvision_edge/monitor.py`
- Modify: `src/reefvision_edge/preview.py`
- Modify: `src/reefvision_edge/api.py`
- Modify: `src/reefvision_edge/sessions.py`
- Test: `tests/test_monitor.py`
- Test: `tests/test_api.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Produces: coherent JPEG response headers `X-Frame-Index` and `X-Frame-Published-At`; `PreviewFrame.published_at`; camera preview publication capped at `PREVIEW_MAX_FPS = 30.0` with final newest-frame persistence.
- Consumes: existing `MonitorStatus.updated_at`, `PreviewStore.put`, and camera session loop.

- [ ] Write failing tests proving monitor JPEG metadata is coherent, session preview responses expose publish time, camera preview writes are rate-limited, and stop forces the newest frame.
- [ ] Run the focused tests and verify failures are caused by missing metadata/throttling behavior.
- [ ] Implement atomic monitor snapshots, preview publish timestamps, response headers, and camera-only preview rate limiting with final publish.
- [ ] Run `pytest tests/test_monitor.py tests/test_api.py tests/test_sessions.py -q` and verify all pass.
- [ ] Commit as `fix: bound live preview publication`.

### Task 2: Timestamp-paced live monitor recording

**Files:**
- Modify: `src/reefvision_edge/sources.py`
- Modify: `src/reefvision_edge/monitor_video.py`
- Modify: `src/reefvision_edge/sessions.py`
- Test: `tests/test_sources.py`
- Test: `tests/test_monitor_video.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Produces: elapsed `SourceFrame.timestamp_ms` for live cameras; `MonitorVideoWriter.write(frame, timestamp_ms=...) -> bool` that selects frames on the configured output timeline.
- Consumes: Task 1 camera session loop and unchanged overlay rendering.

- [ ] Write failing tests with literal live timestamps showing they advance and that a 60 FPS input timeline produces a 30 FPS decoded output of matching duration.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement live elapsed timestamps and timestamp-paced writes; pass the source timestamp from SessionRunner.
- [ ] Run `pytest tests/test_sources.py tests/test_monitor_video.py tests/test_sessions.py -q` and verify all pass.
- [ ] Commit as `fix: preserve live recording timebase`.

### Task 3: Stable bounded Monitor canvas

**Files:**
- Create: `web/src/components/FrameSurface.tsx`
- Create: `web/src/components/FrameSurface.test.tsx`
- Modify: `web/src/views/Monitor.tsx`
- Modify: `web/src/styles/app.css`
- Modify: `web/src/App.test.tsx`

**Interfaces:**
- Produces: memoized `FrameSurface` with `endpoint`, `streamKey`, `label`, `targetFps`, and throttled telemetry callback; stable canvas role `img`.
- Consumes: Task 1 frame metadata headers and existing Monitor source/session selection.

- [ ] Write failing component tests proving the canvas node survives parent rerenders, fetches sequentially with one request in flight, recovers from an initial 404, and cancels stale draws when `streamKey` changes.
- [ ] Write a failing Monitor test proving stats can change without replacing the frame surface and that Display FPS/Frame age are separate metrics.
- [ ] Run `npm test -- FrameSurface.test.tsx App.test.tsx` from `web` and verify expected failures.
- [ ] Implement the canvas pump, throttled telemetry, Monitor integration, and canvas styling without changing the 500 ms statistics poll.
- [ ] Run `npm test -- FrameSurface.test.tsx App.test.tsx`, `npm run typecheck`, and `npm run format:check` from `web`.
- [ ] Commit as `fix: keep monitor frame surface stable`.

### Task 4: Package and physical-camera verification

**Files:**
- Modify generated static assets under `src/reefvision_edge/static/` via `npm run build`.
- Create acceptance artifacts under `.artifacts/windows-monitor-stability-<commit>-<timestamp>/`.
- Update: `docs/verification-report.md`

**Interfaces:**
- Consumes: Tasks 1-3 and the verified local Windows CUDA runtime/model.
- Produces: rebuilt offline Windows CUDA bundle, frozen backend, MSI if required by the build script, three-pass before/after metrics, exported CSV/video, and physical-camera report.

- [ ] Run the full Python and web suites, lint/type checks, and Windows packaged-backend tests.
- [ ] Build the Windows CUDA bundle/backend using the existing pinned local runtime with no downloads.
- [ ] Verify preview-only, CUDA inference, and CUDA plus both recordings on the Sony feed; record capture FPS, displayed FPS, median/p95 frame age, dropped frames, CUDA operators, and VRAM availability.
- [ ] Ask the user to wave and record their smooth-motion observation.
- [ ] Export and reopen the session; decode video; verify duration/timer; reconcile frame and detection CSV rows.
- [ ] Update the verification report, commit as `docs: record physical monitor stability acceptance`, and run final whole-branch review.

