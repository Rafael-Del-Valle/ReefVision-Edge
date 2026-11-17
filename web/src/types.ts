export type SessionStatus =
  "queued" | "running" | "completed" | "canceled" | "interrupted" | "failed";

export type ComputeDevice = "auto" | "nvidia" | "cpu";

export type InferenceProfile = "full_frame" | "thesis_tiling";

export interface OverlaySettings {
  timer: boolean;
  trails: boolean;
  boxes: boolean;
  masks: boolean;
  labels: boolean;
}

export type OverlayName = keyof OverlaySettings;

export interface OverlayCapability {
  supported: boolean;
  reason: string | null;
}

export interface CaptureDevice {
  id: string;
  label: string;
  index: number;
  backend: string;
  available: boolean;
}

export interface MonitorStatus {
  state:
    | "idle"
    | "connecting"
    | "receiving"
    | "replaying"
    | "stalled"
    | "disconnected"
    | "error";
  source_id: string | null;
  frame_index: number;
  detail: string | null;
  updated_at: number;
}

export interface SessionCreateOptions {
  modelId: string;
  sourceId: string;
  sampleEvery: number;
  inferenceProfile: InferenceProfile;
  computeDevice: ComputeDevice;
  recordData: boolean;
  recordMonitor: boolean;
  previewOverlays: OverlaySettings;
  recordingOverlays: OverlaySettings;
  enableProfiling: boolean;
}

export interface ModelRecord {
  id: string;
  name: string;
  format: string;
  input_width: number;
  input_height: number;
  classes: Record<string, string>;
  confidence_threshold: number;
  iou_threshold: number;
  model_sha256: string;
  metadata_sha256: string;
  provider: string;
  available_providers: string[];
  source_format?: "onnx" | "ultralytics_pt";
  source_sha256?: string;
  conversion?: Record<string, unknown> | null;
  supported_profiles?: InferenceProfile[];
}

export interface ModelImportInspection {
  task: string;
  architecture_family: string;
  class_names: string[];
  input_width: number;
  input_height: number;
  supported: boolean;
  source_sha256: string;
  versions: Record<string, string>;
}

export interface ModelImportRecord {
  id: string;
  status:
    | "inspecting"
    | "awaiting_confirmation"
    | "converting"
    | "completed"
    | "failed"
    | "interrupted";
  filename: string;
  source_sha256: string;
  inspection: ModelImportInspection | null;
  error: string | null;
  model_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface SourceRecord {
  id: string;
  kind: "video" | "camera";
  name: string;
  sha256: string | null;
  codec?: string;
  container?: string;
  width?: number;
  height?: number;
  fps?: number;
  frame_count?: number;
  device?: string;
  created_at?: string;
}

export interface SessionRecord {
  id: string;
  model_id: string;
  source_id: string;
  status: SessionStatus;
  sample_every: number;
  inference_profile: InferenceProfile;
  compute_device: ComputeDevice;
  record_data: boolean;
  record_monitor: boolean;
  preview_overlays: OverlaySettings;
  recording_overlays: OverlaySettings;
  enable_profiling: boolean;
  provider: string | null;
  actual_device: string | null;
  provider_reason: string | null;
  frame_count: number;
  detection_count: number;
  input_fps: number | null;
  inference_fps: number | null;
  average_latency_ms: number | null;
  median_latency_ms: number | null;
  p95_latency_ms: number | null;
  peak_vram_mb: number | null;
  dropped_frames: number;
  operator_profile: Record<string, unknown> | null;
  operator_profile_path: string | null;
  monitor_video_path: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface Capabilities {
  available_providers: string[];
  camera_backend: "V4L2" | "Media Foundation / DirectShow" | "unavailable";
  offline_runtime: boolean;
  overlay_capabilities: Record<OverlayName, OverlayCapability>;
}

export interface PublicSettings {
  bind_host: string;
  bind_port: number;
  data_dir: string;
  auth_required: boolean;
}
