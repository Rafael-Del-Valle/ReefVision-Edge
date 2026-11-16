# Windows Live View, Recording, and RTX 3090 Design

## Scope

This pass makes the Windows Monitor usable for recorded replay and Windows capture devices, adds independent data and monitor-video recording, and proves ONNX Runtime CUDA inference on the workstation's RTX 3090. DJI mapping and Jetson deployment remain out of scope. Physical-camera acceptance remains open until a camera is connected.

## Runtime profiles

The application exposes `Auto`, `NVIDIA GPU`, and `CPU` compute choices. `CPU` creates a CPU-only ONNX Runtime session. `NVIDIA GPU` preloads the bundled NVIDIA DLLs, requires `CUDAExecutionProvider`, and fails with an actionable dependency/provider error if CUDA cannot initialize. `Auto` attempts the same CUDA path and falls back to CPU while preserving a user-visible reason.

Windows x64 uses an exact, reproducible CUDA profile: `onnxruntime-gpu==1.32.0`, CUDA 13.4 redistributable libraries, and cuDNN 9. The bundle downloads these wheels once, records hashes, installs them into the frozen build environment, and includes their DLLs. The CPU and Jetson runtime profiles remain separately selectable; the Jetson profile is not built or accepted in this pass.

Every run records the requested compute choice, actual device, active provider chain, provider explanation, and ONNX Runtime operator-profile summary. Explicit NVIDIA mode is accepted only when profiling records at least one CUDA-executed operator. CPU-assigned operators are reported, not hidden.

## Monitor sources and states

Windows capture discovery probes Media Foundation and DirectShow camera indices without making a device persistent. The API returns stable source identifiers, display labels, backend, and availability. A selected recorded video provides a paced, looping raw preview; a selected camera provides live frames when it can be opened. Monitor states are `idle`, `connecting`, `receiving`, `replaying`, `stalled`, `disconnected`, and `error`, with a visible explanation.

Only one component owns a camera handle. Starting inference releases preview ownership before the session opens that camera. Recorded replay uses independent decoders and is the automated moving-frame acceptance source. No connected camera is present on the workstation, so discovery and error handling are tested while physical image acceptance stays open.

## Sessions and recording

Session creation stores:

- `record_data` and `record_monitor` independently;
- `preview_overlays` and `recording_overlays` independently;
- requested compute device and profiling flag.

Data recording writes frame timestamps, dimensions, inference status, latency, class ID/name, confidence, and box geometry. When data recording is off, aggregate session metrics still update but frame and detection rows are not retained.

Monitor recording writes a software-encoded MP4 from the rendered preview frames. The software encoder is the required baseline; NVIDIA video encoding is optional and not part of GPU-inference acceptance. The video is finalized on completion, cancellation, or failure and is copied into session export when present.

The timer, boxes, and labels overlays are supported for the current detector. Trails and masks are displayed as unavailable with explicit reasons because the current output has neither track identities nor segmentation masks. Preview and recording settings are applied separately and saved in the manifest so a reopened session can explain exactly what the saved video contains.

## Metrics and artifacts

Sessions expose input FPS, inference FPS, median and p95 inference latency, dropped frames, and peak process-attributable NVIDIA VRAM. VRAM sampling uses local `nvidia-smi`; inability to sample is reported without invalidating CPU runs. Exports keep schema-v1 compatibility while adding optional fields and artifacts: `frames.csv`, `detections.csv`, `monitor.mp4`, provider profile summary, recording flags, overlay settings, and device metrics.

GPU/CPU parity uses the same deterministic local compute model and clip. Class IDs must match exactly; confidence scores must differ by no more than `1e-4`; each box coordinate may differ by no more than `0.5` pixel. The GPU proof uses ONNX Runtime JSON profiling and requires CUDA node events. The combined acceptance run lasts at least 60 seconds with both recordings enabled, then reopens the session, parses the CSV, decodes the saved MP4, and compares overlay settings to the manifest.

## Bundle acceptance

The rebuilt local Windows x64 bundle must run without the source checkout, a developer server, runtime downloads, or a terminal. Its acceptance harness verifies bundled DLL preloading, available providers, explicit GPU failure behavior, actual CUDA profiling, CPU/GPU parity, all three recording combinations, sustained replay metrics, exported CSV, decoded MP4, and saved overlay settings. It reports the tested commit and bundle path. Physical-camera and optional NVENC checks remain open unless hardware/support is present.
