# ReefVision-Edge v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an offline-first, independently deployable ReefVision-Edge v0.1 that runs local ONNX object detection over recorded video or V4L2 cameras, persists sessions, previews overlays, and exports validated results.

**Architecture:** A FastAPI service owns settings, auth, SQLite repositories, media/model storage, an ONNX Runtime adapter, and bounded in-process session workers. A React/TypeScript/Vite dashboard is built into static assets served by FastAPI from the same origin; the browser polls small status and preview endpoints so field operation does not depend on WebSockets or external services.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, SQLite, ONNX Runtime, NumPy, OpenCV, React 19, TypeScript, Vite, Vitest, Playwright.

**Spec:** `docs/specs/ReefVision-Edge-v0.1.md`

## Global Constraints

- Product and runtime naming must remain `ReefVision-Edge`, independent from desktop ReefVision.
- Default bind is loopback; any non-loopback bind must refuse startup without Basic-auth credentials and protect UI, API, and preview routes.
- Runtime must require no network access and must never modify source media or model files.
- Canonical boxes are original-frame pixel coordinates; recorded sampling is deterministic and source frame identity is exportable.
- Theme values and stylesheet provenance must match ReefVision commit `e94dad6a3bece7b776bf45776bc5d06cfbc2c0dc`.
- Hardware and codec claims require actual evidence; unavailable Jarvis/DJI/Sony/UVC paths remain explicitly unverified.

## Review Focus

- A model whose tensor dimensions disagree with its class map must be rejected before registration; Task 2 tests the dry-run shape validator.
- A stopped/restarted job must never remain `running`; Task 4 tests cancellation and startup interruption recovery.
- Letterboxed detections near padding and frame edges must map and clip correctly; Task 2 tests coordinate reversal and clipping.
- Disk-write failure must retain a consistent `failed` session with committed earlier frames; Task 4 injects a storage failure.
- LAN mode must not expose static files or previews without auth; Task 1 tests all route classes through middleware.

---

### Task 1: Service foundation, settings, authentication, and database

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/reefvision_edge/config.py`, `src/reefvision_edge/auth.py`, `src/reefvision_edge/db.py`, `src/reefvision_edge/models.py`, `src/reefvision_edge/app.py`
- Test: `tests/test_config_auth.py`, `tests/test_database.py`

**Interfaces:**
- Produces: `Settings.from_env() -> Settings`, `require_safe_bind(settings) -> None`, `Database.initialize()`, `create_app(settings: Settings) -> FastAPI`.

- [ ] Write failing tests for loopback defaults, LAN credential refusal, Basic auth across `/`, `/api/health`, and `/api/previews/x`, schema initialization, and stale-running recovery.
- [ ] Run `python -m pytest tests/test_config_auth.py tests/test_database.py -v` and confirm failure from missing modules.
- [ ] Implement typed settings, safe-bind validation, whole-app auth middleware, SQLite WAL/foreign-key initialization, UTC timestamps, and schema/version metadata.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit `feat: establish secure edge service foundation`.

### Task 2: Model registration and ONNX inference adapter

**Files:**
- Create: `src/reefvision_edge/domain.py`, `src/reefvision_edge/model_registry.py`, `src/reefvision_edge/inference.py`, `src/reefvision_edge/geometry.py`, `tests/fixtures/build_tiny_model.py`
- Test: `tests/test_model_registry.py`, `tests/test_inference.py`

**Interfaces:**
- Consumes: `Database`, managed storage paths from `Settings`.
- Produces: `ModelMetadata`, `ModelRegistry.register(model_stream, metadata) -> RegisteredModel`, `YoloOnnxDetector.predict(frame) -> InferenceResult`, `letterbox()`, and `restore_box()`.

- [ ] Write failing tests that generate a deterministic constant-output ONNX graph, register it, assert SHA-256 provenance/provider selection, reject `.pt`/bad metadata/output mismatches, and recover a known original-coordinate box through letterbox.
- [ ] Run the focused tests and confirm missing implementation failures.
- [ ] Implement canonical metadata hashing, strict `ultralytics_yolo_detect_v1` validation, provider selection/reporting, RGB NCHW preprocessing, class-score thresholding, NMS, and inverse coordinate mapping.
- [ ] Run the focused tests and confirm they pass on CPU.
- [ ] Commit `feat: register and execute local yolo onnx models`.

### Task 3: Video and camera sources

**Files:**
- Create: `src/reefvision_edge/sources.py`, `src/reefvision_edge/media_store.py`, `tests/fixtures/build_video.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces: `MediaStore.import_video(stream, filename) -> VideoSource`, `RecordedVideoSource.frames(sample_every)`, `V4L2CameraSource.frames(queue_size=1)`, and `probe_video(path) -> VideoProbe`.

- [ ] Write failing tests for immutable upload copy/hash, deterministic frame indices/timestamps, unreadable video errors, unsupported camera host/unavailable device errors, and a bounded newest-frame queue.
- [ ] Run `python -m pytest tests/test_sources.py -v` and confirm failures.
- [ ] Implement OpenCV-backed probe/iteration with explicit codec metadata and Linux V4L2 capture guarded by platform/device validation.
- [ ] Run focused tests and record the actual synthetic fixture container/codec result.
- [ ] Commit `feat: add recorded and v4l2 field sources`.

### Task 4: Durable session engine and previews

**Files:**
- Create: `src/reefvision_edge/sessions.py`, `src/reefvision_edge/preview.py`
- Test: `tests/test_sessions.py`, `tests/test_failure_paths.py`

**Interfaces:**
- Consumes: registered model, source iterators, detector, database.
- Produces: `SessionService.create/start/stop/get/list`, `SessionRunner.run(session_id)`, `PreviewStore.put/get`, and immutable frame/detection records.

- [ ] Write failing end-to-end service tests for completed recorded inference, per-frame counts/metrics, original-coordinate overlay, cancellation, camera disconnect, injected disk-full failure, absent acceleration, and stale-running recovery after restart.
- [ ] Run focused tests and confirm failures.
- [ ] Implement a bounded worker executor, transactional frame writes, state transitions, cancellation tokens, metrics, JPEG overlay generation, and actionable structured failures.
- [ ] Run focused tests and confirm all terminal states and metrics.
- [ ] Commit `feat: persist and recover inference sessions`.

### Task 5: Versioned exports and round-trip validation

**Files:**
- Create: `src/reefvision_edge/exports.py`, `docs/export-schema-v1.md`, `examples/session-export/manifest.json`, `examples/session-export/detections.csv`
- Test: `tests/test_exports.py`

**Interfaces:**
- Consumes: session/model/source/frame/detection rows.
- Produces: `ExportService.build_zip(session_id) -> Path`, `validate_export(path) -> ExportSummary`.

- [ ] Write failing tests for exact manifest version/provenance fields, deterministic CSV rows, coordinate/time basis, ZIP member safety, and round-trip row/count/hash agreement.
- [ ] Run focused tests and confirm failures.
- [ ] Implement stable JSON/CSV serialization, atomic ZIP replacement, export validation, and a deterministic checked-in sample export.
- [ ] Run focused tests and validate the sample.
- [ ] Commit `feat: export validated edge session bundles`.

### Task 6: Typed API and ReefVision-themed dashboard

**Files:**
- Create: `src/reefvision_edge/api.py`, `web/package.json`, `web/src/api.ts`, `web/src/types.ts`, `web/src/main.tsx`, `web/src/App.tsx`, `web/src/views/Monitor.tsx`, `web/src/views/Sessions.tsx`, `web/src/views/Settings.tsx`, `web/src/styles/theme.css`, `web/src/styles/app.css`
- Test: `tests/test_api.py`, `web/src/App.test.tsx`, `web/src/styles/theme.test.ts`

**Interfaces:**
- Produces: REST routes for health/capabilities, models, sources, sessions, preview, stop, export, and settings; dashboard navigation and workflows consuming those routes.

- [ ] Write failing backend API tests and frontend component/theme-token tests covering successful workflow plus invalid metadata, unreadable video, unavailable camera, and provider/session status messages.
- [ ] Run `python -m pytest tests/test_api.py -v` and `npm --prefix web test` and confirm failures.
- [ ] Implement typed endpoints, multipart uploads, static frontend serving, and the compact Monitor/Sessions/Settings UI with exact semantic tokens and local assets only.
- [ ] Run API/frontend tests, `npm --prefix web run typecheck`, and `npm --prefix web run build`.
- [ ] Commit `feat: deliver edge monitoring dashboard`.

### Task 7: Browser acceptance, theme parity, and screenshots

**Files:**
- Create: `web/playwright.config.ts`, `web/tests/e2e/field-workflow.spec.ts`, `web/tests/e2e/theme.spec.ts`, `docs/screenshots/*.png`

**Interfaces:**
- Consumes: production frontend build and running FastAPI test server with deterministic fixture data.
- Produces: browser assertions and laptop/compact screenshots for all three views.

- [ ] Write Playwright tests that register the tiny ONNX model, upload the synthetic clip, run/stop/open/export a session, reload after service restart, and assert computed colors for surfaces, text, action, selected nav, focus, and semantic status.
- [ ] Run browser tests and confirm the initial missing-flow failures.
- [ ] Complete any minimal API/UI wiring needed by the tests, preserving task boundaries.
- [ ] Run Playwright at laptop and compact viewports and save reviewed screenshots.
- [ ] Commit `test: verify field workflow and reefvision theme`.

### Task 8: Offline packaging, deployment recipe, and acceptance report

**Files:**
- Create: `README.md`, `THIRD_PARTY_NOTICES.md`, `deploy/reefvision-edge.service`, `scripts/build_offline_bundle.ps1`, `scripts/build_offline_bundle.sh`, `docs/verification-report.md`
- Modify: `pyproject.toml`, `src/reefvision_edge/app.py`

**Interfaces:**
- Produces: `reefvision-edge` console command, reproducible environment/setup/deployment steps, offline wheel/npm preparation guidance, SSH tunnel and LAN auth instructions, model export/registration recipe, and evidence matrix.

- [ ] Add packaging smoke tests for clean temporary data startup with bundled frontend and local fixtures.
- [ ] Implement console entry point and production static-asset fallback; write systemd/storage/tunnel/auth/export/codec documentation and provenance notice.
- [ ] Run Ruff, mypy, full pytest, frontend format check/typecheck/test/build, Playwright, offline clean-start/restart/export validation, and record exact results.
- [ ] Record Windows host results and explicitly list unverified Jarvis/JetPack/TensorRT/CUDA/UVC/DJI/Sony paths.
- [ ] Commit `docs: ship ReefVision-Edge v0.1 deployment evidence`.
