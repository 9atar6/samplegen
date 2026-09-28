// A sample row: play button, waveform, metadata, keep / trash / favorite, drag-out.

import { api } from "./api.js";
import { icon, iconButton } from "./icons.js";
import * as player from "./player.js";
import { toast } from "./toast.js";
import { drawWaveform } from "./waveform.js";

const rows = new Map(); // sampleId -> { el, record, canvas, buffer }

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    observer.unobserve(entry.target);
    loadWave(entry.target.dataset.id);
  }
}, { rootMargin: "200px" });

function formatSeconds(s) {
  return s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${s.toFixed(s < 10 ? 2 : 1)} s`;
}

function describe(record) {
  const p = record.params || {};
  const tags = (record.tags || []).map((t) => `#${t}`).join(" ");
  let base;
  if (record.mode === "loop") base = `${p.bpm} BPM · ${p.key} · ${p.bars} bars · seed ${record.seed}`;
  else if (record.mode === "instrument") base = `instrument · ${p.low_note}–${p.high_note} · ${p.notes} notes · ⤴ opens it`;
  else if (record.mode === "stems") base = `${p.stem} stem · ${formatSeconds(record.duration)}`;
  else base = `${formatSeconds(record.duration)} · seed ${record.seed}`;
  return tags ? `${base} · ${tags}` : base;
}

function isLooping(record) {
  return record.mode === "loop" || Boolean(record.params && (record.params.loop || (record.mode === "stems" && record.params.bpm)));
}

function openEditor(row, onSaved) {
  const next = row.el.nextElementSibling;
  if (next && next.classList.contains("sample-editor")) {
    next.remove();
    return;
  }
  const form = document.createElement("form");
  form.className = "sample-editor";
  const name = document.createElement("input");
  name.value = row.record.name;
  name.maxLength = 120;
  name.placeholder = "Name";
  const tags = document.createElement("input");
  tags.value = (row.record.tags || []).join(", ");
  tags.placeholder = "Tags, comma separated (e.g. metal, impact, dark)";
  const save = document.createElement("button");
  save.type = "submit";
  save.className = "ghost";
  save.textContent = "Save";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "ghost";
  cancel.textContent = "Cancel";
  const error = document.createElement("span");
  error.className = "editor-error";
  form.append(name, tags, save, cancel, error);
  cancel.addEventListener("click", () => form.remove());
  form.addEventListener("keydown", (e) => { if (e.key === "Escape") form.remove(); });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      let record = row.record;
      if (name.value.trim() && name.value.trim() !== record.name) record = await api.rename(record.id, name.value.trim());
      record = await api.setTags(record.id, tags.value.split(",").map((t) => t.trim()).filter(Boolean));
      form.remove();
      onSaved(record);
      row.el.focus();
    } catch (err) {
      error.textContent = err.message;
    }
  });
  row.el.after(form);
  name.focus();
}

function fileName(record) {
  return record.rel_path.split("/").pop();
}

async function loadWave(id) {
  const row = rows.get(id);
  if (!row) return;
  try {
    row.buffer = await player.loadBuffer(id);
    redraw(id);
  } catch {
    row.el.classList.add("broken");
  }
}

export function redraw(id) {
  const row = rows.get(id);
  if (row && row.buffer && row.el.isConnected) drawWaveform(row.canvas, id, row.buffer, player.progress(id));
}

export function createSampleRow(record, { onChange } = {}) {
  const el = document.createElement("div");
  el.className = "sample";
  el.tabIndex = 0;
  el.draggable = true;
  el.dataset.id = record.id;

  const playBtn = iconButton("play", "Play / stop (Space)", "play");
  const canvas = document.createElement("canvas");
  canvas.className = "wave";
  const info = document.createElement("div");
  info.className = "info";
  const title = document.createElement("div");
  title.className = "name";
  const meta = document.createElement("div");
  meta.className = "meta";
  info.append(title, meta);

  const favBtn = iconButton("star", "Favorite (F)", "fav");
  const keepBtn = iconButton("check", "Keep (K)", "keep");
  const trashBtn = iconButton("trash", "Trash (X)", "trash");
  const revealBtn = iconButton("folder", "Show in folder", "reveal");
  const sourceBtn = iconButton("swap", "Use as source for Transform / Edit (T)", "use-source");
  const stemsBtn = iconButton("stems", "Split into stems (S)", "stems");
  const editBtn = iconButton("edit", "Rename / tags (E)", "edit");
  const actions = document.createElement("div");
  actions.className = "actions";
  actions.append(favBtn, keepBtn, trashBtn, editBtn, sourceBtn, stemsBtn, revealBtn);
  el.append(playBtn, canvas, info, actions);

  const row = { el, record, canvas, buffer: null };
  rows.set(record.id, row);

  const render = () => {
    const r = row.record;
    title.textContent = r.name;
    title.title = r.prompt;
    meta.textContent = describe(r);
    stemsBtn.hidden = r.mode === "instrument";
    el.classList.toggle("kept", r.status === "kept");
    el.classList.toggle("trashed", r.status === "trashed");
    el.classList.toggle("favorite", r.favorite);
    keepBtn.title = r.status === "kept" ? "Un-keep (back to Inbox)" : "Keep (K)";
    trashBtn.title = r.status === "trashed" ? "Restore" : "Trash (X)";
    trashBtn.innerHTML = icon(r.status === "trashed" ? "restore" : "trash");
  };

  const update = async (promise) => {
    try {
      row.record = await promise;
      render();
      if (onChange) onChange(row.record);
    } catch (err) {
      el.classList.add("error-flash");
      toast(err.message, "error");
      setTimeout(() => el.classList.remove("error-flash"), 1200);
    }
  };

  const emit = (name) => document.dispatchEvent(new CustomEvent(name, { detail: row.record }));
  const actionsApi = {
    toggle: () => player.toggle(record.id, { loop: isLooping(row.record) }),
    keep: () => update(api.setStatus(record.id, row.record.status === "kept" ? "new" : "kept")),
    trash: () => update(api.setStatus(record.id, row.record.status === "trashed" ? "new" : "trashed")),
    favorite: () => update(api.setFavorite(record.id, !row.record.favorite)),
    useAsSource: () => emit("samplegen:use-source"),
    stems: () => { if (row.record.mode !== "instrument") emit("samplegen:stems"); },
    edit: () => openEditor(row, (updated) => update(Promise.resolve(updated))),
  };
  el.sampleActions = actionsApi;
  sourceBtn.addEventListener("click", actionsApi.useAsSource);
  stemsBtn.addEventListener("click", actionsApi.stems);
  editBtn.addEventListener("click", actionsApi.edit);

  playBtn.addEventListener("click", actionsApi.toggle);
  keepBtn.addEventListener("click", actionsApi.keep);
  trashBtn.addEventListener("click", actionsApi.trash);
  favBtn.addEventListener("click", actionsApi.favorite);
  revealBtn.addEventListener("click", () => api.reveal(record.id).catch(() => {}));
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const fraction = (e.clientX - rect.left) / rect.width;
    const duration = row.buffer ? row.buffer.duration : 0;
    player.play(record.id, { offset: fraction * duration, loop: isLooping(row.record) });
    el.focus();
  });
  el.addEventListener("dragstart", (e) => {
    // Chromium turns DownloadURL into a real file drop (Explorer, most DAWs).
    const url = new URL(api.audioUrl(record.id), location.href).href;
    e.dataTransfer.setData("DownloadURL", `audio/wav:${fileName(row.record)}:${url}`);
    e.dataTransfer.setData("text/uri-list", url);
    e.dataTransfer.effectAllowed = "copy";
  });

  render();
  observer.observe(el);
  return el;
}

export function removeRow(id) {
  const row = rows.get(id);
  if (!row) return;
  if (player.isPlaying(id)) player.stop();
  const next = row.el.nextElementSibling;
  if (next && next.classList.contains("sample-editor")) next.remove();
  row.el.remove();
  rows.delete(id);
}

// Redraw the playing row every frame; update play buttons on start/stop.
let playingId = null;
player.onChange((id) => {
  for (const [rid, row] of rows) {
    row.el.classList.toggle("playing", rid === id);
    const play = row.el.querySelector(".play");
    const wanted = rid === id ? "stop" : "play";
    if (play.dataset.glyph !== wanted) {
      play.dataset.glyph = wanted;
      play.innerHTML = icon(wanted);
    }
    if (rid !== id) redraw(rid);
  }
  playingId = id;
});

function tick() {
  if (playingId) redraw(playingId);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

window.addEventListener("resize", () => {
  for (const id of rows.keys()) redraw(id);
});
