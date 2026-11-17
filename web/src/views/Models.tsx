import { FormEvent, useEffect, useState } from "react";

import { api } from "../api";
import type { ModelImportRecord, ModelRecord } from "../types";

interface ModelsProps {
  models: ModelRecord[];
  onModelRegistered: (model?: ModelRecord) => void | Promise<void>;
}

const pendingStatuses = new Set(["inspecting", "converting"]);

function familyLabel(record: ModelImportRecord): string {
  return record.inspection?.architecture_family === "ultralytics_yolov8"
    ? "Ultralytics YOLOv8 detection"
    : (record.inspection?.architecture_family ?? "Unknown checkpoint");
}

export function Models({ models, onModelRegistered }: ModelsProps) {
  const [checkpoint, setCheckpoint] = useState<File | null>(null);
  const [configuration, setConfiguration] = useState<File | null>(null);
  const [trusted, setTrusted] = useState(false);
  const [record, setRecord] = useState<ModelImportRecord | null>(null);
  const [modelName, setModelName] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [error, setError] = useState<string | null>(null);
  const [registering, setRegistering] = useState(false);
  const [onnxModel, setOnnxModel] = useState<File | null>(null);
  const [onnxMetadata, setOnnxMetadata] = useState<File | null>(null);

  useEffect(() => {
    if (!record || !pendingStatuses.has(record.status)) return;
    let active = true;
    const poll = async () => {
      try {
        const updated = await api.modelImport(record.id);
        if (!active) return;
        setRecord(updated);
        if (updated.status === "completed") await onModelRegistered();
      } catch (reason) {
        if (active)
          setError(
            reason instanceof Error ? reason.message : "Model import failed.",
          );
      }
    };
    const timer = window.setInterval(() => void poll(), 350);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [record?.id, record?.status, onModelRegistered]);

  const inspect = async () => {
    if (!checkpoint || !trusted) return;
    if (
      !window.confirm(
        "Trusted .pt files can load Python objects. Continue only if you trust this checkpoint?",
      )
    )
      return;
    setError(null);
    try {
      const imported = await api.inspectModelImport(
        checkpoint,
        configuration ?? undefined,
      );
      setRecord(imported);
      setModelName(checkpoint.name.replace(/\.pt$/i, ""));
      if (imported.error) setError(imported.error);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Checkpoint inspection failed.",
      );
    }
  };

  const convert = async () => {
    if (!record || !modelName.trim()) return;
    if (
      !window.confirm(
        "Export this checkpoint as static FP32 ONNX with external NMS and register it locally?",
      )
    )
      return;
    setError(null);
    try {
      const updated = await api.convertModelImport(record.id, {
        modelName: modelName.trim(),
        confidenceThreshold: confidence,
        iouThreshold: iou,
      });
      setRecord(updated);
      if (updated.status === "completed") await onModelRegistered();
      if (updated.error) setError(updated.error);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Checkpoint conversion failed.",
      );
    }
  };

  const registerOnnx = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!onnxModel?.size || !onnxMetadata?.size) {
      setError("Choose both an ONNX model and its JSON metadata.");
      return;
    }
    setRegistering(true);
    setError(null);
    try {
      await onModelRegistered(await api.registerModel(onnxModel, onnxMetadata));
      event.currentTarget.reset();
      setOnnxModel(null);
      setOnnxMetadata(null);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "ONNX registration failed.",
      );
    } finally {
      setRegistering(false);
    }
  };

  return (
    <section className="view models-view">
      <div className="view-heading">
        <div>
          <p className="eyebrow">Local model registry</p>
          <h2>Models</h2>
          <p>
            Register an Edge-compatible ONNX model or inspect and convert a
            trusted Ultralytics YOLOv8 detection checkpoint locally.
          </p>
        </div>
        <span className="quiet-chip">{models.length} registered</span>
      </div>

      {error ? (
        <p className="inline-alert" role="alert">
          {error}
        </p>
      ) : null}

      <div className="models-grid">
        <article className="panel model-import-card">
          <p className="panel-kicker">Trusted checkpoint</p>
          <h3>Import PyTorch model</h3>
          <p className="card-note">
            Supported now: Ultralytics YOLOv8 detection. Segmentation,
            classification, pose, and unknown checkpoint families are rejected
            with a compatibility error.
          </p>
          <label>
            <span>Trusted PyTorch checkpoint</span>
            <input
              type="file"
              accept=".pt"
              onChange={(event) => {
                setCheckpoint(event.target.files?.[0] ?? null);
                setRecord(null);
                setTrusted(false);
                setError(null);
              }}
            />
          </label>
          <label>
            <span>Optional args.yaml or data.yaml</span>
            <input
              type="file"
              accept=".yaml,.yml"
              onChange={(event) =>
                setConfiguration(event.target.files?.[0] ?? null)
              }
            />
          </label>
          <label className="check-row trust-check">
            <input
              type="checkbox"
              checked={trusted}
              onChange={(event) => setTrusted(event.target.checked)}
            />
            <span>
              I trust this local checkpoint and understand it will be loaded
            </span>
          </label>
          <button
            className="primary"
            disabled={
              !checkpoint || !trusted || record?.status === "inspecting"
            }
            onClick={() => void inspect()}
          >
            {record?.status === "inspecting"
              ? "Inspecting…"
              : "Inspect checkpoint"}
          </button>

          {record?.inspection ? (
            <section className="inspection-card" aria-label="Checkpoint review">
              <div className="inspection-title">
                <div>
                  <span>Detected model</span>
                  <strong>{familyLabel(record)}</strong>
                </div>
                <span className="status-badge" data-state={record.status}>
                  {record.status.replaceAll("_", " ")}
                </span>
              </div>
              <dl>
                <div>
                  <dt>Task</dt>
                  <dd>{record.inspection.task}</dd>
                </div>
                <div>
                  <dt>Model input</dt>
                  <dd>
                    {record.inspection.input_width} ×{" "}
                    {record.inspection.input_height}
                  </dd>
                </div>
                <div>
                  <dt>Classes</dt>
                  <dd>{record.inspection.class_names.length}</dd>
                </div>
              </dl>
              <ol className="class-list">
                {record.inspection.class_names.map((name, index) => (
                  <li key={`${index}-${name}`}>
                    <span>{index}</span>
                    {name}
                  </li>
                ))}
              </ol>
              {record.status === "awaiting_confirmation" ? (
                <div className="conversion-options">
                  <label>
                    <span>Registered model name</span>
                    <input
                      value={modelName}
                      onChange={(event) => setModelName(event.target.value)}
                    />
                  </label>
                  <div className="threshold-grid">
                    <label>
                      <span>Confidence threshold</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={confidence}
                        onChange={(event) =>
                          setConfidence(event.target.valueAsNumber)
                        }
                      />
                    </label>
                    <label>
                      <span>IoU threshold</span>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        value={iou}
                        onChange={(event) => setIou(event.target.valueAsNumber)}
                      />
                    </label>
                  </div>
                  <button
                    className="primary"
                    disabled={!modelName.trim()}
                    onClick={() => void convert()}
                  >
                    Convert to ONNX
                  </button>
                </div>
              ) : null}
              {record.status === "converting" ? (
                <p className="progress-note">
                  Converting in a separate local process…
                </p>
              ) : null}
              {record.status === "completed" ? (
                <p className="success-note">
                  Conversion complete. The checkpoint and derived ONNX are
                  registered together.
                </p>
              ) : null}
            </section>
          ) : null}
        </article>

        <div className="models-side">
          <article className="panel model-import-card">
            <p className="panel-kicker">Existing workflow</p>
            <h3>Register ONNX directly</h3>
            <form onSubmit={(event) => void registerOnnx(event)}>
              <label>
                <span>ONNX model</span>
                <input
                  name="onnx-model"
                  type="file"
                  accept=".onnx"
                  onChange={(event) =>
                    setOnnxModel(event.target.files?.[0] ?? null)
                  }
                />
              </label>
              <label>
                <span>Model metadata</span>
                <input
                  name="onnx-metadata"
                  type="file"
                  accept=".json"
                  onChange={(event) =>
                    setOnnxMetadata(event.target.files?.[0] ?? null)
                  }
                />
              </label>
              <button
                className="secondary"
                disabled={registering}
                type="submit"
              >
                {registering ? "Validating…" : "Register ONNX"}
              </button>
            </form>
          </article>
          <article className="panel model-list-card">
            <p className="panel-kicker">Available models</p>
            <h3>Registry</h3>
            {models.length ? (
              <ul>
                {models.map((model) => (
                  <li key={model.id}>
                    <div>
                      <strong>{model.name}</strong>
                      <span>
                        {model.source_format === "ultralytics_pt"
                          ? "Converted .pt + ONNX"
                          : "Direct ONNX"}
                      </span>
                    </div>
                    <small>
                      {Object.keys(model.classes).length} classes ·{" "}
                      {model.input_width} × {model.input_height}
                    </small>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="card-note">No models registered yet.</p>
            )}
          </article>
        </div>
      </div>
    </section>
  );
}
