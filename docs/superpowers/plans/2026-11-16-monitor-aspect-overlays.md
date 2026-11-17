# Monitor Aspect Ratio and Overlay Preferences Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the Sony source aspect ratio from DirectShow capture through the responsive Monitor canvas and saved video, while making unsupported Trails and Masks preferences editable and independently persistent.

**Architecture:** Negotiate and validate a landscape DirectShow mode at the source boundary, preserve each source frame unchanged through inference/overlay/recording, and let the mounted canvas expose decoded dimensions to CSS. Overlay capability flags describe whether a checked preference can currently render; they no longer disable the checkbox.

**Tech Stack:** Python 3.12, OpenCV, React 19, TypeScript, Vitest/Testing Library, pytest, existing Windows CUDA offline packaging.

**Spec:** `docs/superpowers/specs/2026-11-16-monitor-aspect-overlays-design.md`

## Global Constraints

- Work on `codex/reefvision-edge-v0.1` in the current checkout and preserve the mounted-canvas/newest-frame lag fixes.
- Do not open a second camera owner; restart only the known ReefVision Edge backend when the rebuilt backend is ready.
- Keep source pixels and proportions unchanged outside the detector's internal letterbox.
- Keep live and saved overlay preferences independent.

## Review Focus

- A camera that rejects 1920x1080 must fall through to 1280x720 or its unchanged actual mode.
- A real 4:3 source must remain 4:3 rather than being forced to 16:9.
- Window resizing must not stretch or crop the canvas.
- Checked unsupported preferences must survive Monitor unmount/remount and model selection changes.
- Recorder dimensions must equal source-frame dimensions.

---

### Task 1: Source and display aspect preservation

**Files:**
- Modify: `src/reefvision_edge/sources.py`
- Modify: `web/src/components/FrameSurface.tsx`
- Modify: `web/src/views/Monitor.tsx`
- Modify: `web/src/styles/app.css`
- Test: `tests/test_sources.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_monitor_video.py`
- Test: `web/src/components/FrameSurface.test.tsx`

**Interfaces:**
- Produces: `configure_windows_capture(capture) -> tuple[int, int, float]` and `FrameTelemetry.frameWidth/frameHeight`.
- Consumes: unchanged source `NDArray` frames and existing source-coordinate detections.

- [ ] Write failing tests for 16:9 mode fallback, intrinsic canvas dimensions/style contract, source-coordinate restoration, and source-sized MP4 output; run RED.
- [ ] Implement minimal capture negotiation, telemetry, preview dimension metric, and responsive CSS; run targeted tests GREEN.
- [ ] Run Python and web suites; commit.

### Task 2: Editable independent overlay preferences

**Files:**
- Modify: `src/reefvision_edge/monitor_video.py`
- Modify: `web/src/views/Monitor.tsx`
- Modify: `web/src/styles/app.css`
- Test: `tests/test_monitor_video.py`
- Test: `web/src/App.test.tsx`

**Interfaces:**
- Produces: editable persisted `previewOverlays` and `recordingOverlays`; capability reasons `Tracking required` and `Segmentation model required`.
- Consumes: current capability map and existing session-create overlay payloads.

- [ ] Write failing tests for enabled unsupported checkboxes, per-control reason display, independent persistence, and concise backend reasons; run RED.
- [ ] Implement the minimal preference persistence and capability messaging; run targeted tests GREEN.
- [ ] Run all quality gates and commit.

### Task 3: Windows CUDA bundle and physical Sony verification

**Files:**
- Create: `.artifacts/monitor-aspect-overlay-acceptance-<commit>/report.json`
- Rebuild: `.artifacts/reefvision-edge-offline-windows-cuda-v0.1-<commit>`

**Interfaces:**
- Consumes: Tasks 1-2 plus the pinned local Windows CUDA runtime.
- Produces: tested bundle, saved CSV/video/report, and physical measurements.

- [ ] Rebuild the web assets, offline CUDA bundle, and frozen Windows backend without runtime downloads.
- [ ] Restart the known camera-owning backend, verify actual Sony capture and preview dimensions, resize behavior, explicit CUDA, and independent overlay controls.
- [ ] Record 30-60 seconds with data and monitor recording, decode the MP4, inspect CSV/session settings, and report smoothness/overlay alignment.
- [ ] Run final full verification and whole-branch review.
