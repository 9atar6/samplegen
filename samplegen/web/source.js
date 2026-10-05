// Source panel for Transform / Edit: drop or pick a file, see it, select a region.

import { api } from "./api.js";
import { icon } from "./icons.js";
import * as player from "./player.js";
import { toast } from "./toast.js";
import { micSupported, startRecording } from "./voice.js";
import { drawWaveform } from "./waveform.js";

const $ = (sel) => document.querySelector(sel);
const MIN_SELECTION_S = 0.1;
const DRAG_START_PX = 4;

const state = { source: null, buffer: null, selection: null, selectable: false, dragging: null };
let onChangeHandler = () => {};

const key = () => (state.source ? `src:${state.source.id}` : null);

function formatTime(s) {
  return `${s.toFixed(2)} s`;
}

function describe(source) {
  const parts = [formatTime(source.duration)];
  if (source.is_loop && source.bpm) parts.push(`${source.bpm} BPM`, source.key, `${source.bars} bars`);
  else if (source.is_loop) parts.push("loop");
  return parts.filter(Boolean).join(" · ");
}

function redraw() {
  const canvas = $("#source-wave");
  if (!state.buffer || !canvas.isConnected) return;
  drawWaveform(canvas, key(), state.buffer, player.progress(key()));
  if (!state.selection) return;
  const g = canvas.getContext("2d");
  const { start, end } = state.selection;
  const x0 = (start / state.source.duration) * canvas.width;
  const x1 = (end / state.source.duration) * canvas.width;
  g.fillStyle = getComputedStyle(canvas).getPropertyValue("--selection").trim();
  g.fillRect(x0, 0, x1 - x0, canvas.height);
}

function tick() {
  if (key() && player.isPlaying(key())) redraw();
  requestAnimationFrame(tick);
}

function updateSelectionLabel() {
  const label = $("#selection-label");
  if (!state.selectable) {
    label.hidden = true;
    return;
  }
  label.hidden = false;
  label.textContent = state.selection
    ? `Regenerate ${formatTime(state.selection.start)} → ${formatTime(state.selection.end)}`
    : "Drag across the waveform to choose what to regenerate.";
}

function timeAt(event) {
  const rect = $("#source-wave").getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  return fraction * state.source.duration;
}

function bindCanvas() {
  const canvas = $("#source-wave");
  canvas.addEventListener("pointerdown", (e) => {
    if (!state.source || e.button !== 0) return;
    canvas.setPointerCapture(e.pointerId);
    state.dragging = { from: timeAt(e), x: e.clientX, moved: false, previous: state.selection };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!state.dragging || !state.selectable) return;
    // A drag starts after a few pixels, not 20 ms: on a long source 1 px of hand jitter is more than that.
    if (!state.dragging.moved && Math.abs(e.clientX - state.dragging.x) < DRAG_START_PX) return;
    const to = timeAt(e);
    state.dragging.moved = true;
    state.selection = { start: Math.min(state.dragging.from, to), end: Math.max(state.dragging.from, to) };
    redraw();
    updateSelectionLabel();
  });
  const endDrag = (cancelled) => {
    const drag = state.dragging;
    state.dragging = null;
    if (!drag) return;
    if (!drag.moved) {
      if (!cancelled) player.play(key(), { offset: drag.from, loop: state.source.is_loop, url: api.sourceAudioUrl(state.source.id) });
      return;
    }
    // Too short to regenerate (or interrupted): keep the selection you had before.
    if (cancelled || !state.selection || state.selection.end - state.selection.start < MIN_SELECTION_S) {
      state.selection = drag.previous;
    }
    redraw();
    updateSelectionLabel();
    onChangeHandler();
  };
  canvas.addEventListener("pointerup", () => endDrag(false));
  canvas.addEventListener("pointercancel", () => endDrag(true)); // touch scroll took over
}

async function loadFile(file) {
  const status = $("#source-status");
  status.hidden = false;
  status.textContent = `Loading ${file.name}…`;
  try {
    setSource(await api.uploadSource(file));
    status.hidden = true;
    return true;
  } catch (err) {
    status.textContent = err.message;
    return false;
  }
}

function bindDropzone() {
  const zone = $("#source-drop");
  const input = $("#source-file");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    if (input.files[0]) loadFile(input.files[0]);
    input.value = "";
  });
  // A file dropped anywhere else would make the browser open it and leave samplegen.
  for (const type of ["dragover", "drop"]) {
    window.addEventListener(type, (e) => {
      if ([...(e.dataTransfer?.types || [])].includes("Files")) e.preventDefault();
    });
  }
  for (const target of [zone, $("#source-panel")]) {
    target.addEventListener("dragover", (e) => {
      if (![...e.dataTransfer.types].includes("Files")) return;
      e.preventDefault();
      zone.classList.add("over");
    });
    target.addEventListener("dragleave", () => zone.classList.remove("over"));
    target.addEventListener("drop", (e) => {
      if (!e.dataTransfer.files.length) return;
      e.preventDefault();
      zone.classList.remove("over");
      loadFile(e.dataTransfer.files[0]);
    });
  }
}

export function setSource(source) {
  if (state.source) player.forget(key());
  state.source = source;
  state.buffer = null;
  state.selection = null;
  $("#source-drop").hidden = Boolean(source);
  $("#source-voice").hidden = Boolean(source);
  $("#source-loaded").hidden = !source;
  if (source) {
    $("#source-name").textContent = source.name;
    $("#source-meta").textContent = describe(source);
    player.loadBuffer(key(), api.sourceAudioUrl(source.id))
      .then((buffer) => {
        if (state.source && state.source.id === source.id) {
          state.buffer = buffer;
          redraw();
        }
      })
      .catch(() => { $("#source-meta").textContent = "Couldn't load this audio."; });
  }
  updateSelectionLabel();
  onChangeHandler();
}

export function getSource() {
  return state.source;
}

export function getSelection() {
  return state.selection;
}

export function setSelectable(selectable) {
  state.selectable = selectable;
  if (!selectable) state.selection = null;
  $("#source-wave").classList.toggle("selectable", selectable);
  redraw();
  updateSelectionLabel();
}

// ---------- voice to sound ----------
const VOICE_STRENGTH = 0.75; // high enough to replace the voice's timbre, low enough to keep its rhythm

function bindRecorder() {
  const button = $("#source-record");
  const label = $("#record-label");
  if (!micSupported()) {
    $("#source-voice").remove();
    return;
  }
  let session = null;
  const finish = async () => {
    if (!session) return;
    const current = session;
    session = null;
    button.classList.remove("recording");
    button.setAttribute("aria-pressed", "false");
    label.textContent = "Record your voice";
    try {
      const file = await current.stop();
      if ($("#transform-fields").hidden === false && Number($("#strength").value) < VOICE_STRENGTH) {
        $("#strength").value = String(VOICE_STRENGTH);
        $("#strength").dispatchEvent(new Event("input", { bubbles: true }));
      }
      if (await loadFile(file)) {
        toast("Got it. Now describe what it should become, and Generate.", "ok");
        $("#prompt").focus();
      }
    } catch (err) {
      toast(err.message, "error");
    }
  };
  button.addEventListener("click", async () => {
    if (session) return finish();
    try {
      player.stop();
      session = await startRecording({
        onTick: (s) => { label.textContent = `Recording… ${s.toFixed(1)} s — click to stop`; },
        onLimit: finish,
      });
      button.classList.add("recording");
      button.setAttribute("aria-pressed", "true");
    } catch (err) {
      session = null;
      const blocked = err.name === "NotAllowedError" || err.name === "SecurityError";
      toast(blocked ? "The microphone is blocked: allow it from the icon at the left of the address bar, then try again."
        : err.name === "NotFoundError" ? "No microphone found." : `Couldn't start recording: ${err.message}`, "error");
    }
  });
}

export function initSourcePanel({ onChange }) {
  onChangeHandler = onChange || (() => {});
  bindDropzone();
  bindCanvas();
  bindRecorder();
  $("#source-play").innerHTML = icon("play");
  $("#source-play").addEventListener("click", () => {
    if (!state.source) return;
    player.toggle(key(), { loop: state.source.is_loop, url: api.sourceAudioUrl(state.source.id) });
  });
  $("#source-clear").addEventListener("click", () => setSource(null));
  player.onChange((id) => {
    const playing = Boolean(id && id === key());
    $("#source-play").innerHTML = icon(playing ? "stop" : "play");
    $("#source-play").classList.toggle("on", playing);
    redraw();
  });
  window.addEventListener("resize", redraw);
  requestAnimationFrame(tick);
}
