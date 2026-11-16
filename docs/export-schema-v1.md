# ReefVision-Edge session export v1

An export is a ZIP containing exactly `manifest.json` and `detections.csv`. The manifest schema identifier is `reefvision-edge-session/v1`; each CSV row uses `reefvision-edge-detection/v1`.

Coordinates are floating-point `xyxy` values in original source-frame pixels. Frame indices are zero-based. For constant-frame-rate recorded sources, `timestamp_ms` is derived as `frame_index / reported_fps * 1000`; the manifest records this basis. Empty timestamps are permitted for live sources.

The manifest records session state and metrics, sampling, thresholds, class map, source dimensions/codec/hash, model and metadata SHA-256 hashes, and the ONNX Runtime provider actually used. Consumers must reject unknown schema identifiers, unsafe ZIP paths, mismatched counts/classes, invalid hashes, and coordinates outside the source dimensions.

Desktop ReefVision commit `e94dad6a3bece7b776bf45776bc5d06cfbc2c0dc` does not define an import contract for frame-level Edge detections. This v1 schema is therefore a native Edge contract; desktop import remains follow-up work.
