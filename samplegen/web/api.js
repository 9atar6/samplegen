// Thin wrapper over the samplegen HTTP API.

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : {},
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data && data.detail;
    const message = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ")
      : `Request failed (${res.status})`;
    throw new Error(message);
  }
  return data;
}

export const api = {
  status: () => request("/api/status"),
  catalog: () => request("/api/catalog"),
  generate: (body) => request("/api/generate", { method: "POST", body }),
  job: (id) => request(`/api/jobs/${id}`),
  cancel: (id) => request(`/api/jobs/${id}/cancel`, { method: "POST" }),
  sample: (id) => request(`/api/samples/${id}`),
  samples: (params) => request(`/api/samples?${new URLSearchParams(params)}`),
  setStatus: (id, status) => request(`/api/samples/${id}/status`, { method: "POST", body: { status } }),
  setFavorite: (id, favorite) => request(`/api/samples/${id}/favorite`, { method: "POST", body: { favorite } }),
  reveal: (id) => request(`/api/samples/${id}/reveal`, { method: "POST" }),
  rename: (id, name) => request(`/api/samples/${id}/rename`, { method: "POST", body: { name } }),
  setTags: (id, tags) => request(`/api/samples/${id}/tags`, { method: "POST", body: { tags } }),
  tags: () => request("/api/tags"),
  exportPack: (body) => request("/api/packs", { method: "POST", body }),
  stems: (id, exportFormat) => request(`/api/samples/${id}/stems`, { method: "POST", body: { export: exportFormat } }),
  instrument: (body) => request("/api/instruments", { method: "POST", body }),
  styles: () => request("/api/styles"),
  training: () => request("/api/training"),
  trainingScan: (path) => request("/api/training/scan", { method: "POST", body: { path } }),
  trainingDescribe: (clips) => request("/api/training/describe", { method: "POST", body: { clips } }),
  trainingStart: (body) => request("/api/training/start", { method: "POST", body }),
  trainingStop: () => request("/api/training/stop", { method: "POST" }),
  audioUrl: (id) => `/api/samples/${id}/audio`,
  uploadSource: async (file) => {
    const res = await fetch(`/api/sources?${new URLSearchParams({ filename: file.name })}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Error((data && data.detail) || `Upload failed (${res.status})`);
    return data;
  },
  sourceFromSample: (id) => request(`/api/sources/from-sample/${id}`, { method: "POST" }),
  sourceAudioUrl: (id) => `/api/sources/${id}/audio`,
};
