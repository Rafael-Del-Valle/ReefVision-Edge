# Trusted Ultralytics Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline, separately packaged trusted YOLOv8 detection checkpoint converter, a Models import workflow, and full-frame/thesis-tiled ONNX inference with Windows RTX 3090 acceptance.

**Architecture:** The FastAPI service stages trusted files and launches a converter subprocess for inspection and export; it never imports torch or Ultralytics. Converted checkpoints and ONNX files are atomically registered together with provenance. Runtime inference remains ONNX Runtime and selects either the existing full-frame path or deterministic source tiling plus class-aware merge.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLite/local JSON job records, Ultralytics 8.4.215, PyTorch 2.15.0, ONNX 1.23.2, ONNX Runtime GPU 1.32.0, React/TypeScript, PyInstaller, WiX.

**Spec:** `docs/superpowers/specs/2026-11-16-trusted-ultralytics-import-design.md`

## Global Constraints

- The main service process must never import or deserialize PyTorch checkpoints.
- `.pt` loading requires explicit trust acknowledgement and a post-inspection conversion confirmation.
- Only Ultralytics YOLOv8 detection is supported in this pass; every other family/task receives an actionable error.
- Export is static FP32, batch 1, opset 17, `dynamic=False`, `simplify=False`, `nms=False`.
- Direct ONNX registration and all existing sessions remain backward compatible.
- Source checkpoints, configs, converted models, and research media remain under per-user data or ignored artifacts; never Git or release assets.
- Thesis tiling uses 1920×1344 source tiles with 40% overlap; model input size remains independent.
- Synthetic evidence proves engineering behavior only; real-model scientific acceptance remains open.
- A distributable bundle must contain pinned converter dependencies and required license texts, and must perform no runtime downloads.

## Review Focus

- A forged `.pt` extension or malformed trusted checkpoint must fail in the converter without crashing or blocking the service; covered in Task 2.
- Backend restart or converter termination must produce an interrupted/actionable job instead of an eternal converting state; covered in Task 2.
- A checkpoint reporting YOLO11, segmentation, pose, OBB, classification, missing labels, or non-contiguous labels must be identified and rejected before export; covered in Task 2.
- Tiling sources smaller than one tile, exact tile boundaries, and non-divisible right/bottom edges must cover every pixel exactly as intended and merge same-class duplicates without merging different classes; covered in Task 4.
- Bundle construction must fail closed when converter wheels, hashes, or license files are missing; covered in Task 5.

---

### Task 1: Provenance-aware model registry

**Files:**
- Modify: `src/reefvision_edge/domain.py`
- Modify: `src/reefvision_edge/model_registry.py`
- Modify: `src/reefvision_edge/api.py`
- Modify: `src/reefvision_edge/exports.py`
- Test: `tests/test_model_registry.py`
- Test: `tests/test_api.py`
- Test: `tests/test_exports.py`

**Interfaces:**
- Produces: `ConversionProvenance`, `InferenceProfile`, and optional `ModelMetadata.conversion`/`supported_profiles` fields.
- Produces: `ModelRegistry.register_converted(checkpoint_path, onnx_path, metadata, provenance, source_config_path=None) -> RegisteredModel`.
- Produces: model API fields for source format/hash, conversion summary, classes, and supported profiles.

- [ ] Write failing registry/API/export tests proving a converted model keeps unchanged `source.pt`, optional config, ONNX, canonical metadata, conversion JSON, both hashes, class order, and response/manifest provenance while legacy ONNX registration remains unchanged.
- [ ] Run the focused tests and confirm failure because provenance types and `register_converted` do not exist.
- [ ] Implement additive metadata models and atomic converted-model registration; validate that the supplied derived hash matches the ONNX bytes and never move a source before ONNX validation succeeds.
- [ ] Extend model responses and manifests without changing existing required fields or schemas.
- [ ] Run focused tests and the full Python suite.
- [ ] Commit `feat: preserve converted checkpoint provenance`.

### Task 2: Isolated converter and durable import jobs

**Files:**
- Create: `src/reefvision_edge/converter_worker.py`
- Create: `src/reefvision_edge/model_imports.py`
- Modify: `src/reefvision_edge/config.py`
- Modify: `src/reefvision_edge/app.py`
- Modify: `src/reefvision_edge/api.py`
- Test: `tests/test_converter_worker.py`
- Test: `tests/test_model_imports.py`
- Test: `tests/test_api.py`
- Create: `tests/fixtures/fake_converter.py`

**Interfaces:**
- Consumes: `ModelRegistry.register_converted(...)` from Task 1.
- Produces: `ModelImportService.inspect(stream, filename, trusted, config_stream=None, config_filename=None) -> ModelImportRecord`.
- Produces: `ModelImportService.convert(import_id, confidence_threshold, iou_threshold) -> ModelImportRecord` and `get/list`.
- Produces: routes `POST /api/model-imports`, `GET /api/model-imports`, `GET /api/model-imports/{id}`, and `POST /api/model-imports/{id}/convert`.
- Converter request/result JSON schemas are versioned `reefvision-edge-model-conversion/v1`.

- [ ] Write failing worker contract tests with a synthetic inspection payload and rejection fixtures for wrong task/family, missing labels, malformed checkpoint, and unsupported export output.
- [ ] Write failing service/API tests proving unchecked trust is rejected before staging, inspection is asynchronous, API health remains responsive, post-inspection conversion is separate, process failure is actionable, and stale running jobs become interrupted on restart.
- [ ] Run focused tests and confirm expected missing-module/route failures.
- [ ] Implement the converter CLI with lazy imports, atomic result writes, YOLOv8 detect identification, ordered label extraction, optional YAML fallback, fixed-shape FP32 export, version capture, and actionable exceptions.
- [ ] Implement durable job directories and nonblocking process supervision without torch/Ultralytics imports in the service.
- [ ] Wire settings, application lifecycle, and routes; cap checkpoint/config sizes and sanitize filenames.
- [ ] Run focused tests and the full Python suite.
- [ ] Commit `feat: add isolated trusted checkpoint conversion`.

### Task 3: Models workspace and two-phase import UI

**Files:**
- Create: `web/src/views/Models.tsx`
- Modify: `web/src/App.tsx`
- Modify: `web/src/api.ts`
- Modify: `web/src/types.ts`
- Modify: `web/src/styles/app.css`
- Modify: `web/src/App.test.tsx`
- Create: `web/src/views/Models.test.tsx`

**Interfaces:**
- Consumes: model-import endpoints and response fields from Tasks 1-2.
- Produces: Models tab with existing direct ONNX registration plus trusted `.pt` inspection/conversion.
- Produces: polling that stops at `awaiting_confirmation`, `completed`, `failed`, or `interrupted`.

- [ ] Write failing UI tests for direct ONNX compatibility, disabled inspection until trust is checked, browser confirmation before upload, detected task/class/input review, separate conversion confirmation, progress states, and actionable unsupported-checkpoint errors.
- [ ] Run focused web tests and confirm the Models view/import API is absent.
- [ ] Implement types/API and the Models view; keep conversion polling independent of Monitor refresh and preserve the existing ReefVision theme.
- [ ] Move or reuse the ONNX registration form without duplicating backend behavior.
- [ ] Run web tests, typecheck, format check, and production build.
- [ ] Commit `feat: add trusted model import workspace`.

### Task 4: Full-frame and thesis-tiled inference profiles

**Files:**
- Create: `src/reefvision_edge/tiling.py`
- Modify: `src/reefvision_edge/inference.py`
- Modify: `src/reefvision_edge/domain.py`
- Modify: `src/reefvision_edge/sessions.py`
- Modify: `src/reefvision_edge/api.py`
- Modify: `src/reefvision_edge/exports.py`
- Modify: `web/src/types.ts`
- Modify: `web/src/api.ts`
- Modify: `web/src/views/Monitor.tsx`
- Test: `tests/test_tiling.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_sessions.py`
- Test: `tests/test_exports.py`
- Modify: `web/src/App.test.tsx`

**Interfaces:**
- Consumes: `InferenceProfile` from Task 1.
- Produces: `tile_windows(frame_width, frame_height, tile_width=1920, tile_height=1344, overlap=0.4) -> tuple[TileWindow, ...]`.
- Produces: `merge_tiled_detections(detections, iou_threshold) -> tuple[Detection, ...]`.
- Produces: `YoloOnnxDetector.predict(frame, profile=InferenceProfile.FULL_FRAME)`.
- Produces: session/API/manifest `inference_profile` and recorded thesis tile settings.

- [ ] Write failing literal-coordinate tests for small, exact, and irregular frame tiling; assert complete coverage and final edge anchors.
- [ ] Write failing merge tests for same-class duplicate removal, different-class retention, and full-frame coordinate offsets.
- [ ] Write failing session/export/UI tests for profile selection/persistence and backward-compatible full-frame defaults.
- [ ] Run focused tests and confirm profile/tiling behavior is missing.
- [ ] Implement deterministic windows, offset restoration, class-aware NMS merge, and aggregate tile latency.
- [ ] Thread the profile through session creation, execution, metrics, manifests, and Monitor selection.
- [ ] Run focused tests, full Python suite, and full web checks.
- [ ] Commit `feat: add thesis tiled inference profile`.

### Task 5: Pinned converter runtime, licenses, and Windows packaging

**Files:**
- Modify: `pyproject.toml`
- Create: `packaging/windows/model-converter.lock`
- Create: `packaging/windows/model-converter-licenses.json`
- Create: `scripts/build_windows_converter_bundle.ps1`
- Modify: `scripts/build_offline_bundle.ps1`
- Modify: `scripts/build_windows_msi.ps1`
- Create: `packaging/windows/reefvision-edge-model-converter.spec`
- Modify: `packaging/windows/wix/main.wxs`
- Modify: `tests/test_windows_packaging.py`
- Modify: `tests/test_windows_msi.py`
- Modify: `README.md`

**Interfaces:**
- Produces: hashed CPython 3.12 Windows x64 converter wheel lock and offline `converter-wheels/`.
- Produces: packaged `reefvision-edge-model-converter.exe` and license directory installed beside backend payload.
- Produces: environment/config discovery of the converter executable with no download fallback.

- [ ] Write failing packaging tests that parse exact hashes/versions, require allowed license expressions and copied license files, ensure research extensions are excluded from source/MSI payloads, and fail when converter material is incomplete.
- [ ] Run packaging tests and confirm missing lock/bundle/spec behavior.
- [ ] Resolve and hash the exact converter dependency closure for Python 3.12 Windows x64; record license source and redistribution obligations from installed metadata/license files.
- [ ] Implement offline converter bundle construction and fail-closed license collection.
- [ ] Add the converter PyInstaller build and MSI/offline-bundle payload without importing converter dependencies into the backend environment.
- [ ] Run packaging tests and build the offline Windows CUDA bundle.
- [ ] Commit `build: package the offline model converter`.

### Task 6: Synthetic parity and Windows RTX 3090 acceptance

**Files:**
- Create: `scripts/build_synthetic_yolov8_checkpoint.py`
- Create: `scripts/acceptance_windows_model_import.py`
- Create: `tests/test_model_import_acceptance.py`
- Modify: `docs/verification-report.md`

**Interfaces:**
- Consumes: packaged converter, API/UI, inference profiles, and bundle from Tasks 1-5.
- Produces: ignored synthetic checkpoint/media, parity JSON, full-frame/tiled sessions, exported CSV/video, CUDA profile, and acceptance report.

- [ ] Write a failing engineering acceptance test around deterministic fixture outputs and parity tolerances; keep the expensive packaged run environment-gated.
- [ ] Generate a small YOLOv8 detect checkpoint from bundled architecture YAML without downloading weights and generate representative synthetic coral-like frames/clip outside Git.
- [ ] Build the backend/converter payload from the offline bundle and start it on loopback with an isolated data directory.
- [ ] Through the UI, inspect and confirm the trusted checkpoint, verify detected labels, convert it, select NVIDIA GPU, and run recorded full-frame and thesis-tiled sessions.
- [ ] Compare PyTorch and ONNX detections with matched `rect=False`, thresholds, external NMS, and input size; record per-class counts, confidence/box deltas, and limitations.
- [ ] Verify CUDA operator profiling, Monitor labels/boxes, saved-video overlays, detection CSV names/geometry, full-frame versus tiled FPS/latency/drop metrics, and service responsiveness during conversion.
- [ ] Run the complete Python/web/lint/type/build/package suites, validate bundle hashes, and record the exact tested commit/bundle path.
- [ ] Commit `test: verify trusted model import on Windows CUDA`.

## Final review and real-model handoff

- [ ] Review the complete diff against the spec, with special attention to trust boundaries, subprocess quoting/path handling, stale jobs, license completeness, and source-coordinate tiling.
- [ ] Fix every Critical/Important finding using a failing regression test first, then rerun the full suite.
- [ ] Ask the owner to select `best.pt`, matching `args.yaml` or `data.yaml` when available, and representative coral images/clip.
- [ ] Show detected six-class order for confirmation and record missing resize, threshold, tile-edge, and overlap-merge settings explicitly.
- [ ] Keep real-model RTX 3090 full-frame/tiled parity and scientific acceptance open until those files are tested.

