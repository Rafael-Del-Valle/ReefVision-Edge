# Monitor Aspect Ratio and Overlay Preferences Design

## Intent

ReefVision Edge Monitor must show the connected Sony feed using the capture device's real landscape dimensions, without exposing the detector's square tensor shape in the preview or monitor recording. Trails and Masks remain user preferences even when this build cannot render them, so the UI explains the missing capability instead of disabling the preference.

## Constraints

- Keep the existing bounded newest-frame queue and mounted canvas from the live-preview lag fix.
- Prefer a verified Windows DirectShow 16:9 mode (1920x1080, then 1280x720) and keep the actual mode returned by the device; never resize or crop capture frames merely to force 16:9.
- Preserve source pixels through preview rendering and monitor recording. The 64x64 or 640x640 model input remains an internal centered letterbox, with boxes restored to source coordinates.
- Size the browser canvas from its decoded bitmap and CSS intrinsic ratio so resizing scales both axes uniformly and keeps the full frame visible.
- Keep Live preview and Saved recording overlay settings separate, persisted locally, and submitted unchanged in the session manifest.
- Unsupported Trails and Masks are editable. A checked unsupported preference shows `Tracking required` or `Segmentation model required` beside that preference.

## Approaches

1. **Selected: capture negotiation plus intrinsic canvas sizing.** Ask DirectShow for common 16:9 modes and validate the returned frame, then display that frame through a width-constrained, height-auto canvas. This fixes the root capture mode and the independent CSS distortion risk.
2. CSS-only presentation. This would make 640x480 look wider only by stretching or cropping, violating source fidelity.
3. Resize every captured frame to 16:9. This would invent pixels, distort detections, and waste capture/recording bandwidth.

## Capability Findings

The registered model schema only accepts `ultralytics_yolo_detect_v1`; `Detection` contains a box but no mask or track ID. There is no tracker state or tracker implementation in this build. Therefore Boxes and Labels can render now, while Trails and Masks must remain preference-only until tracking and segmentation are added.

## Acceptance

- Automated tests prove mode negotiation, 16:9 source-coordinate restoration, source-sized recording, responsive intrinsic canvas sizing, editable unsupported controls, independent persistence, and concise capability reasons.
- The rebuilt Windows CUDA bundle drives the connected Sony through explicit NVIDIA GPU inference with both recordings enabled.
- Physical evidence records capture/preview dimensions, display responsiveness, decoded saved-video dimensions, overlay alignment, and preview smoothness.
