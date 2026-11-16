import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { App } from "./App";

const model = {
  id: "model-1",
  name: "Coral detector",
  format: "ultralytics_yolo_detect_v1",
  input_width: 640,
  input_height: 640,
  classes: { "0": "coral" },
  confidence_threshold: 0.5,
  iou_threshold: 0.45,
  model_sha256: "a".repeat(64),
  metadata_sha256: "b".repeat(64),
  provider: "CPUExecutionProvider",
  available_providers: ["CPUExecutionProvider"],
};

const source = {
  id: "source-1",
  kind: "video",
  name: "reef.avi",
  sha256: "c".repeat(64),
  codec: "MJPG",
  width: 160,
  height: 90,
  fps: 10,
  frame_count: 5,
};

const session = {
  id: "session-1",
  model_id: "model-1",
  source_id: "source-1",
  status: "completed",
  sample_every: 2,
  provider: "CPUExecutionProvider",
  frame_count: 3,
  detection_count: 6,
  inference_fps: 8.5,
  average_latency_ms: 13.2,
  dropped_frames: 0,
  error_message: null,
  created_at: "2026-11-16T12:00:00Z",
  updated_at: "2026-11-16T12:00:01Z",
  completed_at: "2026-11-16T12:00:01Z",
};

function jsonResponse(value: unknown): Promise<Response> {
  return Promise.resolve(
    new Response(JSON.stringify(value), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

function installFetch(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return jsonResponse([model]);
      if (url.endsWith("/api/sources")) return jsonResponse([source]);
      if (url.endsWith("/api/sessions")) return jsonResponse([session]);
      if (url.endsWith("/api/capabilities"))
        return jsonResponse({
          available_providers: ["CPUExecutionProvider"],
          camera_backend: "unavailable",
          offline_runtime: true,
        });
      if (url.endsWith("/api/settings"))
        return jsonResponse({
          bind_host: "127.0.0.1",
          bind_port: 8765,
          data_dir: "/var/lib/reefvision-edge",
          auth_required: false,
        });
      throw new Error(`Unexpected request: ${url}`);
    }),
  );
}

test("shows prominent model source sampling and run controls", async () => {
  installFetch();
  render(<App />);

  expect(await screen.findByRole("heading", { name: "Field monitor" })).toBeVisible();
  expect(screen.getByLabelText("Model")).toHaveValue("model-1");
  expect(screen.getByLabelText("Source")).toHaveValue("source-1");
  expect(screen.getByLabelText("Sample every N frames")).toHaveValue(1);
  expect(screen.getByRole("button", { name: "Start inference" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
  expect(screen.getByText("CPU baseline")).toBeVisible();
});

test("navigates among monitor sessions and settings with runtime state", async () => {
  installFetch();
  const user = userEvent.setup();
  render(<App />);
  await screen.findByRole("heading", { name: "Field monitor" });

  await user.click(screen.getByRole("tab", { name: "Sessions" }));
  expect(screen.getByRole("heading", { name: "Saved sessions" })).toBeVisible();
  expect(screen.getByText("6 detections")).toBeVisible();
  expect(screen.getByRole("status", { name: "completed" })).toBeVisible();

  await user.click(screen.getByRole("tab", { name: "Settings" }));
  expect(screen.getByRole("heading", { name: "Station settings" })).toBeVisible();
  expect(screen.getByText("CPUExecutionProvider")).toBeVisible();
  expect(screen.getByText("Loopback only")).toBeVisible();
});

test("reports an actionable station loading failure", async () => {
  vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline"))));
  render(<App />);

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not load station data",
  );
  expect(screen.getByRole("button", { name: "Retry" })).toBeVisible();
});
