import { render, screen, waitFor } from "@testing-library/react";
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
  compute_device: "auto",
  record_data: true,
  record_monitor: true,
  preview_overlays: {
    timer: true,
    trails: false,
    boxes: true,
    masks: false,
    labels: true,
  },
  recording_overlays: {
    timer: true,
    trails: false,
    boxes: true,
    masks: false,
    labels: true,
  },
  enable_profiling: true,
  provider: "CPUExecutionProvider",
  actual_device: "CPU",
  provider_reason: "CUDAExecutionProvider is not available. Using CPU.",
  frame_count: 3,
  detection_count: 6,
  input_fps: 10.0,
  inference_fps: 8.5,
  average_latency_ms: 13.2,
  median_latency_ms: 12.5,
  p95_latency_ms: 18.4,
  peak_vram_mb: null,
  dropped_frames: 0,
  operator_profile: null,
  operator_profile_path: null,
  monitor_video_path: "C:/data/recordings/session-1.mp4",
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

function installFetch({
  sources = [source],
  sessions = [session],
  monitorStatus = {
    state: "replaying",
    source_id: "source-1",
    frame_index: 4,
    detail: null,
    updated_at: 1234.5,
  },
}: {
  sources?: (typeof source)[];
  sessions?: (typeof session)[];
  monitorStatus?: {
    state: string;
    source_id: string | null;
    frame_index: number | null;
    detail: string | null;
    updated_at: number;
  };
} = {}): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/models")) return jsonResponse([model]);
      if (url.endsWith("/api/sources")) return jsonResponse(sources);
      if (url.endsWith("/api/sessions")) return jsonResponse(sessions);
      if (url.endsWith("/api/capture-devices"))
        return jsonResponse([
          {
            id: "camera:msmf:0",
            label: "Camera 1 (Media Foundation)",
            index: 0,
            backend: "Media Foundation",
            available: true,
          },
        ]);
      if (url.endsWith("/api/monitor/status"))
        return jsonResponse(monitorStatus);
      if (url.endsWith("/api/capabilities"))
        return jsonResponse({
          available_providers: ["CPUExecutionProvider"],
          camera_backend: "unavailable",
          offline_runtime: true,
          overlay_capabilities: {
            timer: { supported: true, reason: null },
            trails: {
              supported: false,
              reason: "Tracking IDs are not available for the active detector.",
            },
            boxes: { supported: true, reason: null },
            masks: {
              supported: false,
              reason: "The active detector does not output segmentation masks.",
            },
            labels: { supported: true, reason: null },
          },
        });
      if (url.endsWith("/api/settings"))
        return jsonResponse({
          bind_host: "127.0.0.1",
          bind_port: 8765,
          data_dir: "/var/lib/reefvision-edge",
          auth_required: false,
        });
      if (url.endsWith("/api/monitor/frame") || url.includes("/api/previews/"))
        return Promise.resolve(
          new Response(new Blob(["frame"], { type: "image/jpeg" }), {
            status: 200,
            headers: {
              "Content-Type": "image/jpeg",
              "X-Frame-Index": "4",
              "X-Frame-Published-At": String(Date.now() / 1000),
            },
          }),
        );
      throw new Error(`Unexpected request: ${url}`);
    }),
  );
}

test("shows prominent model source sampling and run controls", async () => {
  installFetch();
  render(<App />);

  expect(
    await screen.findByRole("heading", { name: "Field monitor" }),
  ).toBeVisible();
  expect(screen.getByLabelText("Model")).toHaveValue("model-1");
  expect(screen.getByLabelText("Source")).toHaveValue("source-1");
  expect(screen.getByLabelText("Sample every N frames")).toHaveValue(1);
  expect(screen.getByRole("button", { name: "Start inference" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
  expect(screen.getByText("CPU baseline")).toBeVisible();
});

test("does not discover cameras while a camera session owns capture", async () => {
  installFetch({
    sessions: [
      {
        ...session,
        status: "running",
        source_id: "camera-source-1",
      },
    ],
    monitorStatus: {
      state: "receiving",
      source_id: "camera-source-1",
      frame_index: 4,
      detail: null,
      updated_at: 1234.5,
    },
  });
  const fetchMock = vi.mocked(fetch);

  render(<App />);

  expect(
    await screen.findByRole("heading", { name: "Field monitor" }),
  ).toBeVisible();
  expect(
    fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/capture-devices"),
    ),
  ).toHaveLength(0);
  expect(
    screen.getByRole("button", { name: "Refresh cameras" }),
  ).toBeDisabled();
});

test("allows rediscovery after a disconnected camera releases capture", async () => {
  installFetch({
    sources: [
      source,
      {
        ...source,
        id: "camera-source-1",
        kind: "camera",
        name: "Camera 1",
      },
    ],
    sessions: [
      {
        ...session,
        status: "running",
        source_id: "camera-source-1",
      },
    ],
    monitorStatus: {
      state: "disconnected",
      source_id: "camera-source-1",
      frame_index: 4,
      detail: "Camera disconnected.",
      updated_at: 1234.5,
    },
  });
  const fetchMock = vi.mocked(fetch);

  render(<App />);

  expect(
    await screen.findByRole("heading", { name: "Field monitor" }),
  ).toBeVisible();
  expect(
    fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/capture-devices"),
    ),
  ).toHaveLength(1);
  expect(screen.getByRole("button", { name: "Refresh cameras" })).toBeEnabled();
});

test("keeps discovery disabled when another camera session is active", async () => {
  installFetch({
    sources: [
      source,
      { ...source, id: "camera-a", kind: "camera", name: "Camera A" },
      { ...source, id: "camera-b", kind: "camera", name: "Camera B" },
    ],
    sessions: [
      {
        ...session,
        status: "running",
        source_id: "camera-b",
      },
    ],
    monitorStatus: {
      state: "disconnected",
      source_id: "camera-a",
      frame_index: 4,
      detail: "Camera A disconnected.",
      updated_at: 1234.5,
    },
  });
  const fetchMock = vi.mocked(fetch);

  render(<App />);

  expect(
    await screen.findByRole("heading", { name: "Field monitor" }),
  ).toBeVisible();
  expect(
    fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/capture-devices"),
    ),
  ).toHaveLength(0);
  expect(
    screen.getByRole("button", { name: "Refresh cameras" }),
  ).toBeDisabled();
});

test("starting inference does not rediscover cameras", async () => {
  installFetch();
  const fetchMock = vi.mocked(fetch);
  fetchMock.mockImplementation(
    (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/sessions") && init?.method === "POST") {
        return jsonResponse({ ...session, id: "session-2", status: "queued" });
      }
      if (url.endsWith("/api/sessions/session-2/start"))
        return jsonResponse({});
      if (url.endsWith("/api/sources/video") && init?.method === "POST")
        return jsonResponse({ ...source, id: "source-2", name: "new.avi" });
      if (url.endsWith("/api/sources/camera") && init?.method === "POST")
        return jsonResponse({
          ...source,
          id: "camera-source-1",
          kind: "camera",
          name: "Camera 1",
        });
      if (url.endsWith("/api/models")) return jsonResponse([model]);
      if (url.endsWith("/api/sources")) return jsonResponse([source]);
      if (url.endsWith("/api/sessions")) return jsonResponse([session]);
      if (url.endsWith("/api/capture-devices"))
        return jsonResponse([
          {
            id: "camera:msmf:0",
            label: "Camera 1 (Media Foundation)",
            index: 0,
            backend: "Media Foundation",
            available: true,
          },
        ]);
      if (url.endsWith("/api/monitor/status"))
        return jsonResponse({
          state: "replaying",
          source_id: "source-1",
          frame_index: 4,
          detail: null,
          updated_at: 1234.5,
        });
      if (url.endsWith("/api/capabilities"))
        return jsonResponse({
          available_providers: ["CPUExecutionProvider"],
          camera_backend: "unavailable",
          offline_runtime: true,
          overlay_capabilities: {
            timer: { supported: true, reason: null },
            trails: { supported: false, reason: "Tracking is unavailable." },
            boxes: { supported: true, reason: null },
            masks: { supported: false, reason: "Masks are unavailable." },
            labels: { supported: true, reason: null },
          },
        });
      if (url.endsWith("/api/settings"))
        return jsonResponse({
          bind_host: "127.0.0.1",
          bind_port: 8765,
          data_dir: "/var/lib/reefvision-edge",
          auth_required: false,
        });
      throw new Error(`Unexpected request: ${url}`);
    },
  );
  const user = userEvent.setup();
  render(<App />);
  await screen.findByRole("heading", { name: "Field monitor" });
  const discoveryCallsBefore = fetchMock.mock.calls.filter(([input]) =>
    String(input).endsWith("/api/capture-devices"),
  ).length;

  await user.click(screen.getByRole("button", { name: "Start inference" }));
  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some(([input]) =>
        String(input).endsWith("/api/sessions/session-2/start"),
      ),
    ).toBe(true),
  );

  await user.upload(
    screen.getByLabelText("Video file"),
    new File(["video"], "new.avi", { type: "video/x-msvideo" }),
  );
  await user.click(screen.getByRole("button", { name: "Import video" }));
  await user.click(screen.getByRole("button", { name: "Connect camera" }));
  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some(([input]) =>
        String(input).endsWith("/api/sources/camera"),
      ),
    ).toBe(true),
  );

  expect(
    fetchMock.mock.calls.filter(([input]) =>
      String(input).endsWith("/api/capture-devices"),
    ),
  ).toHaveLength(discoveryCallsBefore);
});

test("shows live source compute recording overlays and actual runtime metrics", async () => {
  installFetch();
  render(<App />);

  await screen.findByRole("heading", { name: "Field monitor" });
  expect(screen.getByLabelText("Compute device")).toHaveValue("auto");
  expect(screen.getByLabelText("Record data")).toBeChecked();
  expect(screen.getByLabelText("Record monitor")).not.toBeChecked();
  expect(
    screen.getByRole("img", { name: "Live source preview" }),
  ).toHaveProperty("tagName", "CANVAS");
  expect(screen.getByText("Display")).toBeVisible();
  expect(screen.getByText("Frame age")).toBeVisible();
  expect(screen.getAllByText("Replay preview").length).toBeGreaterThan(0);
  expect(screen.getByLabelText("Camera device")).toHaveDisplayValue(
    "Camera 1 (Media Foundation)",
  );
  expect(
    screen.getByText("CUDAExecutionProvider is not available. Using CPU."),
  ).toBeVisible();
  expect(screen.getByText("12.5 ms")).toBeVisible();
  expect(screen.getByText("18.4 ms")).toBeVisible();
  expect(screen.getAllByText("up to last 4,096 frames")).toHaveLength(2);
  expect(screen.getByLabelText("Preview masks")).toBeDisabled();
  expect(screen.getByLabelText("Recording trails")).toBeDisabled();
  expect(
    screen.getByText("The active detector does not output segmentation masks."),
  ).toBeVisible();

  const fetchMock = vi.mocked(fetch);
  const discoveryCallsBefore = fetchMock.mock.calls.filter(([input]) =>
    String(input).endsWith("/api/capture-devices"),
  ).length;
  await userEvent
    .setup()
    .click(screen.getByRole("button", { name: "Refresh cameras" }));
  await waitFor(() =>
    expect(
      fetchMock.mock.calls.filter(([input]) =>
        String(input).endsWith("/api/capture-devices"),
      ),
    ).toHaveLength(discoveryCallsBefore + 1),
  );
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
  expect(
    screen.getByRole("heading", { name: "Station settings" }),
  ).toBeVisible();
  expect(screen.getByText("CPUExecutionProvider")).toBeVisible();
  expect(screen.getByText("Loopback only")).toBeVisible();
});

test("reports an actionable station loading failure", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.reject(new Error("offline"))),
  );
  render(<App />);

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not load station data",
  );
  expect(screen.getByRole("button", { name: "Retry" })).toBeVisible();
});
