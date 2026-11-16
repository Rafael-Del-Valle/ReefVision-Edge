# Monitor Preview Stability Design

## Intent

Make the Windows Monitor show the latest Sony a7R III frame smoothly while CUDA inference, data recording, and monitor recording run independently. The visible surface must stay mounted, remain close to real time, and never trade away CSV or saved-video integrity.

## Measured problem

- Preview-only captured 60.00 FPS but displayed 1.94 FPS.
- CUDA inference captured 56.68 FPS but displayed 1.93 FPS, with 17 dropped frames during the measured interval.
- CUDA plus both recordings captured 56.73 FPS but displayed 1.92 FPS, with 18 dropped frames during the measured interval.
- `App` polls session state every 500 ms. Monitor turns each poll into a new JPEG URL. The inference image also keys itself by `session.updated_at`, replacing the image DOM node every poll.
- A live 13.512-second recording wrote all 704 source frames into a fixed 30 FPS MP4, producing 23.467 seconds of playback. Live timestamps were `None`, so the saved timer could not advance correctly.
- The camera queues are already bounded to the newest frame. The remaining capture loss is synchronous per-frame work: inference, database updates, JPEG rendering, and preview fsync. Disk-backed preview publication measured about 2 ms per 640x480 frame on this PC.

## Goals

- Keep one visible display surface mounted for the lifetime of a selected source or session.
- Fetch/decode/draw at a bounded target of 30 displayed FPS with at most one frame request in flight.
- Show displayed FPS and frame age separately from the 500 ms session-statistics poll.
- Publish inference previews at no more than the display rate while always making the newest processed frame available.
- Give live camera frames elapsed timestamps and pace the 30 FPS software encoder from those timestamps so saved duration matches wall time within 10%.
- Preserve every inferred frame and detection row requested by Record data; preview throttling must not alter CSV contents or overlay geometry.
- Preserve explicit CUDA profiling and the existing size-one newest-frame capture queue.

## Non-goals

- No NVENC requirement, model change, DJI work, Jetson work, or new network protocol.
- No increase to the session statistics polling rate.
- No attempt to render unsupported trails or masks.

## Approaches

### Chosen: stable canvas with latest-frame fetch

A memoized canvas owns a sequential fetch/decode/draw loop. It leaves the previous pixels visible until the next JPEG is decoded, bounds work to one request, and changes only when the source/session identity changes. Response headers carry frame index and publish time for display telemetry. This reuses the current authenticated HTTP endpoints and has the smallest packaging risk.

### Rejected: multipart MJPEG stream

An MJPEG endpoint would keep an `<img>` mounted and let the browser decode continuously, but it adds a long-lived streaming transport, disconnect handling, authentication behavior, and backend encoder ownership that the local app does not otherwise need.

### Rejected: faster cache-busted `<img>` polling

Removing the React key and polling the image URL faster is smaller, but source changes still trigger image navigation/decoding, can flash between loads, and do not provide a clean one-in-flight bound or reliable display telemetry.

## Components

- Backend frame endpoints return coherent JPEG metadata (`X-Frame-Index`, `X-Frame-Published-At`).
- `PreviewStore` records publish time and camera sessions limit preview rendering/persistence to 30 FPS, forcing a final newest-frame publish at stop.
- Windows and V4L2 live frames receive elapsed capture timestamps.
- `MonitorVideoWriter` uses timestamps to select frames for its configured output FPS.
- A memoized `FrameSurface` canvas fetches the latest frame sequentially and reports throttled display telemetry to Monitor.
- Session statistics continue polling every 500 ms and update outside the canvas.

## Acceptance

- The canvas node identity remains stable across session-stat updates.
- Physical preview, CUDA preview, and CUDA plus both recordings display at least 25 FPS at 640x480 with median displayed-frame age below 100 ms.
- The rebuilt bundle reports CUDA operators on the RTX 3090.
- A 30-60 second saved MP4 decodes, has advancing timer pixels, and its duration is within 10% of the session wall duration.
- Exported frame rows match the session frame count; detection counts reconcile per frame; zero-detection inferred rows remain distinguishable from skipped/missing frames.

