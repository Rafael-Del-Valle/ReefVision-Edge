# Trusted Ultralytics model import design

## Intent

ReefVision-Edge must let a researcher import a trusted Ultralytics YOLOv8 detection
checkpoint without weakening the existing ONNX-only inference runtime. The service remains
responsive and never deserializes PyTorch objects itself. A separately packaged converter
inspects and exports the checkpoint, while Edge continues to execute only validated ONNX.

The engineering workflow is accepted with a small locally generated synthetic YOLOv8
checkpoint. Scientific acceptance remains open until the owner's six-class thesis checkpoint,
matching training configuration, and representative coral media are supplied.

## Goals

- Add a Models workspace with direct ONNX registration and a two-phase trusted `.pt` import.
- Inspect task, architecture family, class order, training image size, and checkpoint hash in a
  separate process before conversion.
- Require an explicit trust acknowledgement and a second confirmation after inspection.
- Support Ultralytics YOLOv8 detection checkpoints first and reject other tasks/families with
  actionable compatibility errors.
- Export static-batch, static-shape FP32 ONNX with external NMS and opset 17.
- Preserve the source checkpoint and derived ONNX together in the local registry, with complete
  conversion provenance and hashes.
- Preserve direct ONNX registration and the existing full-frame inference workflow.
- Add a selectable Thesis tiling profile using source tiles up to 1920 by 1344 pixels with 40%
  horizontal and vertical overlap, independently of the ONNX input size.
- Restore tile detections to full-frame coordinates and merge duplicates by class using the
  registered model's IoU threshold.
- Package conversion dependencies and their license texts locally; perform no runtime download.
- Verify synthetic PyTorch/ONNX parity, ONNX Runtime CUDA execution, Monitor labels, saved video,
  CSV output, and throughput on Windows.

## Non-goals

- Do not claim scientific parity, accuracy, species mapping, or thesis-tile equivalence from the
  synthetic checkpoint.
- Do not support arbitrary pickled PyTorch modules, YOLO segmentation/pose/OBB/classification, or
  non-Ultralytics checkpoint families in this pass.
- Do not commit or publish research weights, media, or locally converted models.
- Do not embed NMS into ONNX or add TensorRT export.
- Do not make the converter a prerequisite for direct ONNX registration or ordinary inference.

## Approaches considered

1. **Separate packaged converter executable (selected).** The service stages a checkpoint and
   launches a short-lived local process. This isolates trusted deserialization, keeps Monitor
   responsive, and allows converter dependencies to be versioned separately from ONNX Runtime.
2. **Load Ultralytics inside the service.** Simpler plumbing, but a malformed checkpoint or a
   PyTorch failure would share memory and failure state with live capture. It also enlarges the
   always-on process.
3. **Require an external Python environment.** Smallest installer, but violates offline,
   reproducible conversion and makes exporter behavior depend on the workstation.

## Import flow

The Models workspace keeps the existing ONNX + metadata form. A separate **Import model** card
accepts `.pt` plus an optional `args.yaml` or `data.yaml`.

1. The user selects a local checkpoint and explicitly checks that it is trusted.
2. **Inspect checkpoint** shows a browser confirmation before upload. Edge stores the untouched
   bytes under its per-user data directory and starts the converter in inspection mode.
3. The UI polls a durable local job record while the normal Monitor remains usable.
4. Inspection returns architecture family, task, ordered class labels, input size, source hash,
   and converter versions. Unsupported checkpoints stop here with an actionable error.
5. The user reviews the detected information and presses **Convert model**. A second confirmation
   starts export.
6. The converter writes a static FP32 ONNX beside the staged source. Edge validates the normal
   `ultralytics_yolo_detect_v1` contract, atomically moves both files into the model registry, and
   records provenance.

Checkpoint loading is limited to this explicit trusted path. Merely selecting a file never loads
it. The service and converter bind no network listeners and conversion performs no downloads.

## Converter contract

The converter is invoked with an operation, a request JSON path, and a result JSON path. It writes
the result atomically and exits nonzero on failure. Inspection and export run with pinned:

- Python 3.12 x64
- Ultralytics 8.4.215
- PyTorch 2.15.0 CPU Windows wheel
- torchvision 0.29.2
- ONNX 1.23.2

The exporter uses `format=onnx`, batch 1, `dynamic=False`, `simplify=False`, opset 17, FP32,
`nms=False`, and CPU export. The detected or explicitly reviewed input shape must be a fixed
multiple of 32. Metadata records centered letterbox, RGB, `[0,1]` scaling, XYWH raw output, class
probabilities, and external class-aware NMS.

Inspection accepts only a loaded Ultralytics model whose task is `detect` and whose model YAML,
checkpoint arguments, or model class identifies YOLOv8. Errors name the detected task/family and
list the currently supported family: Ultralytics YOLOv8 detection.

## Provenance and storage

Each converted registry directory contains:

- `source.pt` — unchanged uploaded checkpoint
- optional `source-config.yaml` — unchanged user-supplied configuration
- `model.onnx` — validated derived model
- `metadata.json` — canonical Edge metadata
- `conversion.json` — canonical inspection/export provenance

Provenance includes source and derived SHA-256, filenames, task/family, ordered class map, input
shape, exporter and runtime versions, exact export arguments, preprocessing/output contract,
confidence/IoU thresholds, and conversion time. Direct ONNX models keep their current layout and
API compatibility.

## Inference profiles

`full_frame` keeps the current centered-letterbox inference.

`thesis_tiling` calculates deterministic covering windows. Interior windows are 1920 by 1344;
strides are 60% of each dimension; final windows are anchored to the right/bottom edge so the
whole frame is covered. Edge windows may be smaller when the source is smaller than a tile. Every
tile is independently letterboxed to the model input shape. Detections are offset back into the
source frame and class-aware greedy NMS merges overlapping duplicates using the model IoU
threshold. The session manifest records the selected profile and tile settings.

This merge is an engineering default. The real checkpoint's original overlap-merging rules,
thresholds, resize behavior, and tile-edge policy remain explicit missing thesis settings until
the owner supplies `args.yaml`, `data.yaml`, or the original inference configuration.

## Parity and acceptance

Synthetic engineering parity uses generated local images and matched `rect=False`, confidence,
IoU, image size, and external NMS settings. Class IDs must match exactly. Confidence and each box
coordinate must stay within the existing Edge tolerances (`1e-4` and `0.5` source pixel) unless a
documented exporter limitation is found. Full-frame and tiled throughput are reported separately.

The Windows acceptance bundle must import through the UI, use explicit NVIDIA GPU mode, prove
CUDA operator execution, render the imported class names/boxes in Monitor and the saved video,
and export matching detection CSV rows. The synthetic run proves engineering behavior only.

Real-model acceptance requires the owner's `best.pt`, optional `args.yaml`/`data.yaml`, several
representative coral images or a short clip, confirmation of the detected six-class order, and an
RTX 3090 PyTorch-versus-ONNX parity run for both full-frame and thesis tiling.

## Licensing and redistribution

ReefVision-Edge is AGPL-3.0-only. Ultralytics 8.4.215 is AGPL-3.0 and therefore license-compatible
with this distribution. PyTorch uses a BSD-style license with bundled third-party license terms;
ONNX is Apache-2.0. The bundle copies the installed distributions' license files and a generated
dependency/version inventory. A distributable build is blocked if required license material is
missing. Research checkpoints and media are never included in build inputs or release assets.

## Failure behavior

- Unchecked trust acknowledgement: request rejected before staging.
- Wrong extension, empty/oversized file, malformed pickle, missing converter, or converter crash:
  actionable job error; service and existing models remain usable.
- Unsupported task/family: inspection succeeds as identification but conversion is disabled with
  the detected values and current support list.
- Missing labels or invalid/non-contiguous class order: conversion blocked; optional YAML is used
  only when it agrees with a loadable detection model or supplies otherwise missing labels.
- ONNX load, shape, dry-run, or output-contract failure: derived file retained in the job evidence
  directory, not registered as runnable.
- Backend restart during a job: durable status becomes interrupted; the user can retry without
  mutating the original checkpoint.

