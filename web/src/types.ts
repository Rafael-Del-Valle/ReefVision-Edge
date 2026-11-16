export type SessionStatus =
  "queued" | "running" | "completed" | "canceled" | "interrupted" | "failed";

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
  provider: string | null;
  frame_count: number;
  detection_count: number;
  inference_fps: number | null;
  average_latency_ms: number | null;
  dropped_frames: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface Capabilities {
  available_providers: string[];
  camera_backend: "V4L2" | "unavailable";
  offline_runtime: boolean;
}

export interface PublicSettings {
  bind_host: string;
  bind_port: number;
  data_dir: string;
  auth_required: boolean;
}
