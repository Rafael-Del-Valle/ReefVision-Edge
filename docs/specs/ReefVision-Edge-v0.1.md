# ReefVision-Edge v0.1 Specification

## Intent

Build a standalone, offline-first field inference application for the Jetson station named Jarvis. A field operator can register a local ONNX object detector, run it against imported footage or a Linux UVC/V4L2 camera, inspect responsive overlays and runtime metrics, reopen durable sessions, and export machine-readable results for desktop review.

## Product boundaries

- Product, package, data, service, documentation, and repository names use `ReefVision-Edge` and remain independent of desktop ReefVision.
- The service is local Python with a same-origin React/TypeScript dashboard. It runs headlessly, binds to loopback by default, and is reachable through an SSH tunnel.
- LAN binding is opt-in and applies authentication to UI, API, and streams.
- Runtime operation is offline. Model weights, fonts, application assets, media, and data remain local.
- Training, field-mask editing, tracking, direct DJI/Sony integrations, and fleet/cloud management are outside v0.1.

## Reference baseline and theme

Desktop ReefVision is read-only reference material at commit `e94dad6a3bece7b776bf45776bc5d06cfbc2c0dc`. The import order is `apps/web/src/styles/clarity-overhaul.css`, followed by `design-system.css` and `app.css` from `App.tsx`.

ReefVision-Edge uses the same semantic token names and values in a local theme layer:

- `--rv-bg: #06171c`
- `--rv-bg-raised: #0a2228`
- `--rv-panel: rgba(7, 34, 40, 0.94)`
- `--rv-panel-soft: rgba(12, 47, 54, 0.72)`
- `--rv-border: rgba(182, 238, 228, 0.14)`
- `--rv-border-strong: rgba(114, 219, 201, 0.42)`
- `--rv-accent: #72dbc9`
- `--rv-accent-strong: #80ddce`
- `--rv-text: #eaf8f5`
- `--rv-muted: #9dbdb8`
- `--rv-muted-strong: #bad7d2`
- success `#72dbc9`, warning `#f3c96b`, danger `#ff8f87`, and info `#8fc7ff`

Browser tests assert computed colors for representative surfaces, text, buttons, selected navigation, focus, and statuses. Screenshots cover Monitor, Sessions, and Settings at laptop and compact widths. Provenance and applicable notices are documented.

## Supported model contract

v0.1 accepts `.onnx` only, with explicit JSON metadata. The first supported output contract is `ultralytics_yolo_detect_v1`: one output tensor shaped `[1, 4 + class_count, candidate_count]`, boxes in input-pixel `xywh`, and per-class confidence values without a separate objectness channel. Input is fixed NCHW RGB float32, normalized to `[0, 1]`, using centered letterbox padding. Class IDs are contiguous integers starting at zero.

Registration validates metadata, graph input/output shape, and a dry inference result; rejects checkpoint formats and unsupported/dynamic variants with structured errors; records SHA-256 hashes for the model and canonical metadata. ONNX Runtime CPU is the portable baseline. Available providers and the provider actually selected are reported, with CPU fallback visible.

Desktop ReefVision trains Ultralytics YOLO checkpoints but does not provide a direct ONNX export command at the pinned commit. The reproducible path is therefore an explicit Ultralytics CLI export from the registered/trained local checkpoint into the supported contract, followed by Edge registration validation. The project includes a tiny deterministic ONNX fixture that proves the contract without a network download.

## Sources and inference

- Imported videos are copied into Edge-managed storage without modifying the source.
- Linux cameras are addressed by V4L2 device path such as `/dev/video0`; unsupported hosts and unavailable devices return actionable errors.
- Decoder qualification records container, codec, dimensions, frame rate, and frame count when available. Synthetic fixtures exercise a locally available codec path. DJI Mini 3/RM330 and Sony A7RIII formats remain unverified without supplied media.
- Recorded sampling is deterministic: process frame zero and every Nth source frame. Each detection stores source frame index and timestamp derived as `frame_index / reported_fps` when constant-rate metadata is available.
- The frame is centered-letterboxed for inference and detections are mapped back and clipped to original pixel coordinates.
- Live queues are bounded and newest-frame biased. Preview exposes boxes, labels, confidence, per-frame count, effective inference FPS, latency, skipped/dropped frames, provider, and connection status.

## Sessions and export

SQLite stores model/source provenance, settings, state, metrics, frame references, and detections. States are `queued`, `running`, `completed`, `canceled`, `interrupted`, and `failed`. Startup converts stale `running` rows to `interrupted`. Camera removal, decoder failure, disk-full errors, cancellation, and service restart produce consistent final state and preserve committed detections.

Exports contain `manifest.json` with schema `reefvision-edge-session/v1` and `detections.csv`. Both include model/source references, class map, source dimensions, frame index/timestamp basis, original-coordinate boxes, score, confidence/IoU thresholds, sample interval, and actual execution provider. A round-trip validator checks consistency.

Desktop ReefVision at the pinned commit imports image-associated COCO/YOLO annotations and video tracks, but it has no contract for frame-level Edge detections. v0.1 therefore ships its native versioned export and documents remaining desktop import work rather than claiming compatibility.

## Dashboard and deployment

The compact dashboard has Monitor, Sessions, and Settings navigation. Source/model selection, start/stop, preview, metrics, and export stay prominent. The service serves the built dashboard from the same origin.

The repository includes local development and production commands, storage layout, an example systemd service, SSH tunnel instructions, authentication/LAN rules, offline setup guidance, and a verification matrix. Hardware-specific Jetson, decoder, execution-provider, DJI/Sony codec, and UVC results are claimed only after actual execution on Jarvis.

## Approaches considered

1. **Single Python service plus same-origin static React build (selected).** One deployable service owns SQLite, media, workers, API, preview, and static assets. This minimizes field setup and keeps authentication consistent while retaining a typed frontend build.
2. **Separate frontend server and Python API.** Easier development isolation but creates two field services, CORS, and duplicated authentication; rejected for Jarvis deployment simplicity.
3. **Native desktop shell.** Strong kiosk experience but conflicts with headless service and SSH-tunnel requirements; rejected for v0.1.

Implementation call: session work runs in bounded in-process worker threads. It is simpler and observable for one Jetson station; persisted state and startup recovery prevent a thread from being mistaken for durable state.
