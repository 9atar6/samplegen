// A sample row: play button, waveform, metadata, keep / trash / favorite, drag-out.

import { api } from "./api.js";
import { icon, iconButton } from "./icons.js";
import * as player from "./player.js";
import { toast } from "./toast.js";
import { drawWaveform, forgetPeaks } from "./waveform.js";

// Rows don't keep their decoded audio: the waveform is drawn from cached peaks, and the
// audio itself lives in the player's small LRU cache, so memory stays flat in long sessions.
// The same sample can be on screen twice (Generate feed + Library), so an id maps to a set of rows.
const rows = new Map(); // sampleId -> Set<{ el, record, canvas, duration, loading }>
const rowOfElement = new WeakMap();

function rowsOf(id) {
  return rows.get(id) ?? new Set();
}

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    observer.unobserve(entry.target);
    const row = rowOfElement.get(entry.target);
    if (row) loadWave(row);
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
  else if (record.mode === "kit") base = `drum kit · ${p.pieces} pieces · ${p.round_robins} takes each · ⤴ opens it`;
  else if (record.mode === "layered") base = `layered hit · take ${p.take} · ${formatSeconds(record.duration)}`;
  else if (record.mode === "stems") base = `${p.stem} stem · ${formatSeconds(record.duration)}`;
  else if (record.mode === "performance" || record.mode === "recovered") base = `${record.prompt} · ${formatSeconds(record.duration)}`;
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

async function extractMidi(row, button) {
  if (button.classList.contains("busy")) return;
  button.classList.add("busy");
  button.title = "Listening for notes…";
  try {
    row.midi = await api.midi(row.record.id);
    button.classList.add("ready");
    button.title = `MIDI ready (${row.midi.notes} notes): drag ♪ into your DAW · click to redo`;
    toast(`${row.midi.notes} notes found. Drag the ♪ button into your DAW.`, "ok");
  } catch (err) {
    button.title = "Extract MIDI notes (M)";
    toast(err.message, "error");
  } finally {
    button.classList.remove("busy");
  }
}

function fileName(record) {
  return record.rel_path.split("/").pop();
}

async function loadWave(row) {
  if (row.loading) return;
  row.loading = true;
  const id = row.record.id;
  try {
    const buffer = await player.loadBuffer(id);
    row.duration = buffer.duration;
    row.el.classList.remove("broken");
    if (row.el.isConnected) drawWaveform(row.canvas, id, buffer, player.progress(id));
  } catch {
    row.el.classList.add("broken");
    row.el.title = "Couldn't load this sound — click its waveform to retry.";
  } finally {
    row.loading = false;
  }
}

export function redraw(id) {
  for (const row of rowsOf(id)) {
    if (row.duration == null || !row.el.isConnected) continue;
    // No peaks for this width yet (e.g. after a resize): fetch the audio again to compute them.
    if (!drawWaveform(row.canvas, id, null, player.progress(id))) loadWave(row);
  }
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
  const midiBtn = iconButton("midi", "Extract MIDI notes (M)", "midi");
  const similarBtn = iconButton("similar", "Find sounds like this one (L)", "similar");
  const variationsBtn = iconButton("spark", "Make 4 more like this one (V)", "variations");
  midiBtn.draggable = true;
  const actions = document.createElement("div");
  actions.className = "actions";
  actions.append(favBtn, keepBtn, trashBtn, editBtn, variationsBtn, sourceBtn, stemsBtn, midiBtn, similarBtn, revealBtn);
  el.append(playBtn, canvas, info, actions);

  const row = { el, record, canvas, duration: null, loading: false };
  if (!rows.has(record.id)) rows.set(record.id, new Set());
  rows.get(record.id).add(row);
  rowOfElement.set(el, row);

  const render = () => {
    const r = row.record;
    title.textContent = r.name;
    title.title = r.prompt;
    meta.textContent = describe(r);
    midiBtn.hidden = r.mode === "instrument" || r.mode === "kit"; // instruments already are notes
    variationsBtn.hidden = r.mode === "instrument" || r.mode === "kit";
    stemsBtn.hidden = r.mode === "instrument" || r.mode === "kit";
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
    midi: () => extractMidi(row, midiBtn),
    similar: () => emit("samplegen:similar"),
    variations: () => { if (!["instrument", "kit"].includes(row.record.mode)) emit("samplegen:variations"); },
  };
  el.sampleActions = actionsApi;
  variationsBtn.addEventListener("click", actionsApi.variations);
  similarBtn.addEventListener("click", actionsApi.similar);
  midiBtn.addEventListener("click", actionsApi.midi);
  midiBtn.addEventListener("dragstart", (e) => {
    e.stopPropagation(); // drag the MIDI file, not the row's WAV
    if (!row.midi) {
      e.preventDefault();
      toast("Click ♪ first to extract the notes, then drag it into your DAW.", "info");
      return;
    }
    const url = new URL(row.midi.url, location.href).href;
    e.dataTransfer.setData("DownloadURL", `audio/midi:${row.midi.filename}:${url}`);
    e.dataTransfer.setData("text/uri-list", url);
    e.dataTransfer.effectAllowed = "copy";
  });
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
    if (el.classList.contains("broken")) loadWave(row); // retry a sound that failed to load
    const duration = row.duration || 0;
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

// Remove the rows of `id` inside `within` (e.g. the Library list), leaving any copy elsewhere alone.
export function removeRow(id, within = document) {
  const set = rowsOf(id);
  for (const row of [...set]) {
    if (!within.contains(row.el)) continue;
    const next = row.el.nextElementSibling;
    if (next && next.classList.contains("sample-editor")) next.remove();
    if (row.el.contains(document.activeElement)) {
      // Keep the keyboard user's place: focus the neighbour instead of dropping to <body>.
      const neighbour = [row.el.nextElementSibling, row.el.previousElementSibling]
        .find((n) => n && n.classList.contains("sample"));
      neighbour?.focus({ preventScroll: false });
    }
    observer.unobserve(row.el);
    row.el.remove();
    set.delete(row);
  }
  if (set.size === 0) {
    rows.delete(id);
    if (player.isPlaying(id)) player.stop();
    player.forget(id);
    forgetPeaks(id);
  }
}

// Redraw the playing row every frame; update play buttons on start/stop.
let playingId = null;
player.onChange((id) => {
  for (const [rid, set] of rows) {
    for (const row of set) {
      row.el.classList.toggle("playing", rid === id);
      const play = row.el.querySelector(".play");
      const wanted = rid === id ? "stop" : "play";
      if (play.dataset.glyph !== wanted) {
        play.dataset.glyph = wanted;
        play.innerHTML = icon(wanted);
      }
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
