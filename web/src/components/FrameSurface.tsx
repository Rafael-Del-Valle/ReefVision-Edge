import { memo, useEffect, useRef } from "react";

export interface FrameTelemetry {
  displayedFps: number;
  frameAgeMs: number | null;
  frameIndex: number | null;
}

interface FrameSurfaceProps {
  endpoint: string;
  streamKey: string;
  label: string;
  targetFps?: number;
  onTelemetry?: (value: FrameTelemetry) => void;
}

function waitForNextFrame(delayMs: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(resolve, delayMs);
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

export const FrameSurface = memo(function FrameSurface({
  endpoint,
  streamKey,
  label,
  targetFps = 30,
  onTelemetry,
}: FrameSurfaceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const telemetryRef = useRef(onTelemetry);

  useEffect(() => {
    telemetryRef.current = onTelemetry;
  }, [onTelemetry]);

  useEffect(() => {
    const controller = new AbortController();
    const intervalMs = 1000 / Math.max(1, targetFps);
    let telemetryStartedAt = performance.now();
    let displayedFrames = 0;

    async function pump() {
      while (!controller.signal.aborted) {
        const cycleStartedAt = performance.now();
        let bitmap: ImageBitmap | null = null;
        try {
          const response = await fetch(endpoint, {
            cache: "no-store",
            signal: controller.signal,
          });
          if (response.ok) {
            bitmap = await createImageBitmap(await response.blob());
            if (!controller.signal.aborted) {
              const canvas = canvasRef.current;
              const context = canvas?.getContext("2d");
              if (canvas && context) {
                if (
                  canvas.width !== bitmap.width ||
                  canvas.height !== bitmap.height
                ) {
                  canvas.width = bitmap.width;
                  canvas.height = bitmap.height;
                }
                context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                displayedFrames += 1;
                const now = performance.now();
                const telemetryElapsed = now - telemetryStartedAt;
                if (telemetryElapsed >= 1000) {
                  const publishedAt = Number(
                    response.headers.get("X-Frame-Published-At"),
                  );
                  const frameIndex = Number(
                    response.headers.get("X-Frame-Index"),
                  );
                  telemetryRef.current?.({
                    displayedFps: (displayedFrames * 1000) / telemetryElapsed,
                    frameAgeMs: Number.isFinite(publishedAt)
                      ? Math.max(0, Date.now() - publishedAt * 1000)
                      : null,
                    frameIndex: Number.isFinite(frameIndex) ? frameIndex : null,
                  });
                  telemetryStartedAt = now;
                  displayedFrames = 0;
                }
              }
            }
          }
        } catch (error) {
          if (controller.signal.aborted) return;
          if (error instanceof DOMException && error.name === "AbortError")
            return;
        } finally {
          bitmap?.close();
        }
        const remaining = Math.max(
          0,
          intervalMs - (performance.now() - cycleStartedAt),
        );
        await waitForNextFrame(remaining, controller.signal);
      }
    }

    void pump();
    return () => controller.abort();
  }, [endpoint, streamKey, targetFps]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={label}
      data-endpoint={endpoint}
      data-stream-key={streamKey}
    />
  );
});
