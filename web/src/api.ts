import type {
  Capabilities,
  ModelRecord,
  PublicSettings,
  SessionRecord,
  SourceRecord,
} from "./types";

interface ApiErrorBody {
  error?: { code?: string; message?: string };
  detail?: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as ApiErrorBody;
      message = body.error?.message ?? body.detail ?? message;
    } catch {
      // Keep the status-derived fallback when the response is not JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const api = {
  models: () => request<ModelRecord[]>("/api/models"),
  sources: () => request<SourceRecord[]>("/api/sources"),
  sessions: () => request<SessionRecord[]>("/api/sessions"),
  capabilities: () => request<Capabilities>("/api/capabilities"),
  settings: () => request<PublicSettings>("/api/settings"),
  session: (id: string) => request<SessionRecord>(`/api/sessions/${id}`),
  registerModel: async (model: File, metadata: File): Promise<ModelRecord> => {
    const body = new FormData();
    body.append("model", model);
    body.append("metadata_json", await metadata.text());
    return request<ModelRecord>("/api/models", { method: "POST", body });
  },
  importVideo: (video: File): Promise<SourceRecord> => {
    const body = new FormData();
    body.append("video", video);
    return request<SourceRecord>("/api/sources/video", {
      method: "POST",
      body,
    });
  },
  registerCamera: (device: string) =>
    request<SourceRecord>("/api/sources/camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device }),
    }),
  createSession: (modelId: string, sourceId: string, sampleEvery: number) =>
    request<SessionRecord>("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: modelId,
        source_id: sourceId,
        sample_every: sampleEvery,
      }),
    }),
  startSession: (id: string) =>
    request<SessionRecord>(`/api/sessions/${id}/start`, { method: "POST" }),
  stopSession: (id: string) =>
    request<SessionRecord>(`/api/sessions/${id}/stop`, { method: "POST" }),
};
