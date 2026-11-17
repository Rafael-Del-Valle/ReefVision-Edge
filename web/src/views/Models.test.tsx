import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Models } from "./Models";

const apiMocks = vi.hoisted(() => ({
  registerModel: vi.fn(),
  inspectModelImport: vi.fn(),
  modelImport: vi.fn(),
  convertModelImport: vi.fn(),
}));

vi.mock("../api", () => ({
  api: apiMocks,
}));

const inspection = {
  task: "detect",
  architecture_family: "ultralytics_yolov8",
  class_names: ["open", "closed"],
  input_width: 640,
  input_height: 640,
  supported: true,
  source_sha256: "a".repeat(64),
  versions: {
    python: "3.12.10",
    ultralytics: "8.4.215",
    torch: "2.15.0",
    onnx: "1.23.2",
  },
};

const importRecord = {
  id: "import-1",
  status: "awaiting_confirmation",
  filename: "synthetic-yolov8.pt",
  source_sha256: "a".repeat(64),
  inspection,
  error: null,
  model_id: null,
  created_at: "2026-11-16T20:00:00Z",
  updated_at: "2026-11-16T20:00:01Z",
};

beforeEach(() => {
  apiMocks.registerModel.mockReset();
  apiMocks.inspectModelImport.mockReset();
  apiMocks.modelImport.mockReset();
  apiMocks.convertModelImport.mockReset();
  vi.stubGlobal(
    "confirm",
    vi.fn(() => true),
  );
});

test("keeps direct ONNX registration available", async () => {
  const user = userEvent.setup();
  const registered = {
    id: "model-2",
    name: "Direct ONNX",
    format: "ultralytics_yolo_detect_v1",
  };
  apiMocks.registerModel.mockResolvedValue(registered);
  const onModelRegistered = vi.fn();
  render(<Models models={[]} onModelRegistered={onModelRegistered} />);

  await user.upload(
    screen.getByLabelText("ONNX model"),
    new File(["onnx"], "model.onnx"),
  );
  await user.upload(
    screen.getByLabelText("Model metadata"),
    new File(["{}"], "metadata.json", { type: "application/json" }),
  );
  await user.click(screen.getByRole("button", { name: "Register ONNX" }));

  await waitFor(() => expect(apiMocks.registerModel).toHaveBeenCalledOnce());
  expect(onModelRegistered).toHaveBeenCalledWith(registered);
});

test("requires trust, reviews labels, then separately confirms conversion", async () => {
  const user = userEvent.setup();
  apiMocks.inspectModelImport.mockResolvedValue(importRecord);
  apiMocks.convertModelImport.mockResolvedValue({
    ...importRecord,
    status: "completed",
    model_id: "model-imported",
  });
  apiMocks.modelImport.mockResolvedValue({
    ...importRecord,
    status: "completed",
    model_id: "model-imported",
  });
  const onModelRegistered = vi.fn();
  render(<Models models={[]} onModelRegistered={onModelRegistered} />);

  await user.upload(
    screen.getByLabelText("Trusted PyTorch checkpoint"),
    new File(["checkpoint"], "synthetic-yolov8.pt"),
  );
  expect(
    screen.getByRole("button", { name: "Inspect checkpoint" }),
  ).toBeDisabled();

  await user.click(
    screen.getByLabelText(
      "I trust this local checkpoint and understand it will be loaded",
    ),
  );
  await user.click(screen.getByRole("button", { name: "Inspect checkpoint" }));

  expect(confirm).toHaveBeenCalledWith(
    expect.stringContaining("load Python objects"),
  );
  expect(await screen.findByText("Ultralytics YOLOv8 detection")).toBeVisible();
  expect(screen.getByText("open")).toBeVisible();
  expect(screen.getByText("closed")).toBeVisible();
  expect(screen.getByText("640 × 640")).toBeVisible();

  await user.clear(screen.getByLabelText("Registered model name"));
  await user.type(
    screen.getByLabelText("Registered model name"),
    "Synthetic thesis",
  );
  await user.click(screen.getByRole("button", { name: "Convert to ONNX" }));

  expect(confirm).toHaveBeenLastCalledWith(
    expect.stringContaining("static FP32 ONNX"),
  );
  await waitFor(() =>
    expect(apiMocks.convertModelImport).toHaveBeenCalledOnce(),
  );
});

test("shows actionable unsupported-checkpoint errors", async () => {
  const user = userEvent.setup();
  apiMocks.inspectModelImport.mockResolvedValue({
    ...importRecord,
    status: "failed",
    inspection: null,
    error:
      "Detected task 'segment'. ReefVision-Edge currently supports detection checkpoints only.",
  });
  render(<Models models={[]} onModelRegistered={vi.fn()} />);

  await user.upload(
    screen.getByLabelText("Trusted PyTorch checkpoint"),
    new File(["checkpoint"], "segment.pt"),
  );
  await user.click(
    screen.getByLabelText(
      "I trust this local checkpoint and understand it will be loaded",
    ),
  );
  await user.click(screen.getByRole("button", { name: "Inspect checkpoint" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "supports detection checkpoints only",
  );
});
