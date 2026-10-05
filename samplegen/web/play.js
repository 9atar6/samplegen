// Play view: play a generated instrument from a MIDI keyboard, the computer keys or the
// screen; record takes, hum melodies into notes, load MIDI files, save takes to the library.

import { api } from "./api.js";
import { bindComputerKeys, buildPiano, connectMidi, midiSupported } from "./keyboard.js";
import { parseMidi, writeMidi } from "./midifile.js";
import { Sampler, takeLength } from "./sampler.js";
import { detectKey, foldIntoLoop, loopBars, loopSeconds, nearestLoopBpm, quantize } from "./theory.js";
import { toast } from "./toast.js";
import { startRecording } from "./voice.js";

const $ = (sel) => document.querySelector(sel);
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = (midi) => `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
const PIANO_PAD = 5; // semitones shown beyond the sampled range (they still play, repitched)
const DEFAULT_RANGE = [48, 84];

const ctx = new (window.AudioContext || window.webkitAudioContext)();
const sampler = new Sampler(ctx);
const state = {
  instruments: [],
  piano: null,
  take: [],            // [{ time, dur, midi, vel }]
  recording: null,     // { start, open: Map(midi -> { time, vel }) }
  playback: null,
  humming: null,
  midi: null,
};

const visible = () => !$("#view-play").hidden;

// ---------- notes in (from any input) ----------

function noteDown(midi, vel) {
  if (midi < 0 || midi > 127) return;
  sampler.noteOn(midi, vel);
  state.piano?.light(midi, true);
  const rec = state.recording;
  if (rec) rec.open.set(midi, { time: ctx.currentTime - rec.start, vel });
}

function noteUp(midi) {
  sampler.noteOff(midi);
  state.piano?.light(midi, false);
  const rec = state.recording;
  const started = rec?.open.get(midi);
  if (started) {
    rec.open.delete(midi);
    rec.notes.push({ ...started, midi, dur: Math.max(0.03, ctx.currentTime - rec.start - started.time) });
    drawRoll();
  }
}

// ---------- instrument ----------

async function refreshInstruments() {
  try {
    state.instruments = await api.instruments();
  } catch (err) {
    $("#play-load-status").textContent = err.message;
    return;
  }
  const select = $("#play-instrument");
  const current = select.value;
  select.replaceChildren(...state.instruments.map((i) =>
    new Option(`${i.name} · ${noteName(i.low)}–${noteName(i.high)}`, i.id)));
  $("#play-no-instruments").hidden = state.instruments.length > 0;
  select.disabled = state.instruments.length === 0;
  if (!state.instruments.length) {
    buildKeys(...DEFAULT_RANGE);
    return;
  }
  const wanted = state.instruments.some((i) => i.id === current) ? current : loadSaved();
  select.value = state.instruments.some((i) => i.id === wanted) ? wanted : state.instruments[0].id;
  if (select.value !== sampler.instrument?.id) await loadInstrument(select.value);
}

async function loadInstrument(id) {
  const status = $("#play-load-status");
  try {
    const detail = await api.instrumentDetail(id);
    status.textContent = `Loading ${detail.count} notes…`;
    await sampler.load(detail, (f) => { status.textContent = `Loading… ${Math.round(f * 100)}%`; });
    status.textContent = `${detail.name}: ${detail.count} sampled notes, ${noteName(detail.low)}–${noteName(detail.high)}.`;
    buildKeys(Math.max(21, detail.low - PIANO_PAD), Math.min(108, detail.high + PIANO_PAD));
    save(id);
  } catch (err) {
    status.textContent = err.message;
  }
}

function buildKeys(low, high) {
  // Start and end on a white key so the keyboard looks like one.
  while ([1, 3, 6, 8, 10].includes(low % 12)) low--;
  while ([1, 3, 6, 8, 10].includes(high % 12)) high++;
  state.piano = buildPiano($("#play-piano"), low, high, { onDown: noteDown, onUp: noteUp });
}

function save(id) { try { localStorage.setItem("samplegen.play.instrument", id); } catch { /* private mode */ } }
function loadSaved() { try { return localStorage.getItem("samplegen.play.instrument"); } catch { return null; } }

// ---------- MIDI keyboard ----------

async function connectKeyboard() {
  if (!midiSupported()) {
    toast("This browser can't use MIDI keyboards. Open samplegen in Chrome or Edge.", "error");
    return;
  }
  try {
    state.midi = await connectMidi({
      onDown: (m, v) => { if (visible()) noteDown(m, v); },
      onUp: noteUp,
      onSustain: (on) => sampler.setSustain(on),
      onDevices: (names) => {
        $("#midi-status").dataset.state = names.length ? "on" : "off";
        $("#midi-label").textContent = names.length ? `MIDI: ${names.join(", ")}` : "No MIDI keyboard found: plug it in (it'll appear here).";
        $("#midi-connect").hidden = true;
      },
    });
  } catch {
    toast("MIDI access was blocked: allow it from the icon at the left of the address bar.", "error");
  }
}

// ---------- takes ----------

function setTake(notes, source) {
  const first = notes.length ? Math.min(...notes.map((n) => n.time)) : 0;
  state.take = notes.map((n) => ({ ...n, time: n.time - first })); // start at the first note
  const length = state.take.length ? takeLength(state.take, 0) : 0;
  $("#take-info").textContent = state.take.length
    ? `${state.take.length} notes · ${length.toFixed(1)} s${source ? ` · ${source}` : ""}`
    : "No take yet.";
  drawRoll();
}

function toggleRecord() {
  const button = $("#take-record");
  if (state.recording) {
    const rec = state.recording;
    state.recording = null;
    for (const [midi, started] of rec.open) rec.notes.push({ ...started, midi, dur: Math.max(0.03, ctx.currentTime - rec.start - started.time) });
    button.classList.remove("recording");
    button.setAttribute("aria-pressed", "false");
    button.querySelector("span").textContent = "Record";
    if (rec.notes.length) setTake(rec.notes, "recorded");
    else toast("Nothing was played while recording.", "info");
    return;
  }
  if (!sampler.ready) return toast("Pick an instrument first.", "info");
  stopPlayback();
  if (ctx.state === "suspended") ctx.resume();
  state.recording = { start: ctx.currentTime, open: new Map(), notes: [] };
  button.classList.add("recording");
  button.setAttribute("aria-pressed", "true");
  button.querySelector("span").textContent = "Stop recording";
  toast("Recording: play your keyboard. The take starts at your first note.", "info");
}

function stopPlayback() {
  state.playback?.stop();
  state.playback = null;
  $("#take-play").querySelector("span:last-child").textContent = "Play";
}

function togglePlay() {
  if (state.playback) return stopPlayback();
  if (!state.take.length) return toast("Record, hum or load a take first.", "info");
  if (!sampler.ready) return toast("Pick an instrument first.", "info");
  if (ctx.state === "suspended") ctx.resume();
  const startedAt = ctx.currentTime + 0.08;
  const playback = sampler.playTake(state.take, (n) => {
    state.piano?.light(n.midi, true);
    setTimeout(() => state.piano?.light(n.midi, false), n.dur * 1000);
  });
  state.playback = { ...playback, startedAt };
  $("#take-play").querySelector("span:last-child").textContent = "Stop";
  playback.done.then(() => { if (state.playback?.done === playback.done) stopPlayback(); });
  requestAnimationFrame(animateRoll);
}

async function toggleHum() {
  const button = $("#take-hum");
  const label = $("#take-hum-label");
  if (state.humming) {
    const session = state.humming;
    state.humming = null;
    button.classList.remove("recording");
    label.textContent = "Listening for notes…";
    try {
      const wav = await session.stop();
      const { events } = await api.hum(wav);
      setTake(events.map(([time, end, midi, vel]) => ({ time, dur: end - time, midi, vel: Math.max(0.35, vel) })), "hummed");
      toast(`${events.length} notes heard. Press Play to hear them on your instrument.`, "ok");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      label.textContent = "Hum a melody";
    }
    return;
  }
  try {
    stopPlayback();
    state.humming = await startRecording({
      onTick: (s) => { label.textContent = `Humming… ${s.toFixed(1)} s, click to stop`; },
      onLimit: toggleHum,
    });
    button.classList.add("recording");
  } catch (err) {
    state.humming = null;
    toast(err.name === "NotAllowedError" ? "The microphone is blocked: allow it from the icon at the left of the address bar."
      : `Couldn't start the microphone: ${err.message}`, "error");
  }
}

async function loadMidiBuffer(buffer, source) {
  try {
    const { notes, bpm } = parseMidi(buffer);
    if (!notes.length) throw new Error("That MIDI file has no notes.");
    $("#take-bpm").value = String(bpm);
    setTake(notes, source);
    toast(`${notes.length} notes loaded. Press Play.`, "ok");
  } catch (err) {
    toast(err.message, "error");
  }
}

function bindDrop() {
  const zone = $("#play-drop");
  zone.addEventListener("dragover", (e) => {
    const types = [...e.dataTransfer.types];
    if (types.includes("Files") || types.includes("text/uri-list")) {
      e.preventDefault();
      zone.classList.add("over");
    }
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", async (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    const file = e.dataTransfer.files[0];
    if (file) return loadMidiBuffer(await file.arrayBuffer(), file.name);
    // A sample's ♪ button dragged from the feed/library: its URL points at our own API.
    const url = e.dataTransfer.getData("text/uri-list").split("\n")[0].trim();
    if (!url || new URL(url, location.href).origin !== location.origin || !url.includes("/midi")) {
      return toast("Drop a .mid file, or a sample's ♪ button after extracting its MIDI.", "info");
    }
    const res = await fetch(url);
    if (!res.ok) return toast("That sample has no MIDI yet: click its ♪ button first.", "error");
    loadMidiBuffer(await res.arrayBuffer(), "from a sample");
  });
}

function takeName() {
  return $("#take-name").value.trim() || `${sampler.instrument?.name || "Take"} take`;
}

function saveMidiFile() {
  if (!state.take.length) return toast("Nothing to save yet.", "info");
  const blob = new Blob([writeMidi(state.take, Number($("#take-bpm").value) || 120)], { type: "audio/midi" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${takeName().replace(/[<>:"/\\|?*]+/g, "")}.mid`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 5000);
}

async function saveToLibrary() {
  if (!state.take.length) return toast("Nothing to save yet.", "info");
  if (!sampler.ready) return toast("Pick an instrument first.", "info");
  const button = $("#take-save");
  if (button.disabled) return;
  button.disabled = true;
  try {
    const bpm = Number($("#take-bpm").value) || 120;
    const rendered = await sampler.render(state.take);
    const record = await api.savePerformance(encodeWavFloat(rendered), {
      name: takeName(), instrument: sampler.instrument?.id || "", bpm,
    });
    await api.attachMidi(record.id, writeMidi(state.take, bpm));
    toast(`“${record.name}” saved to the Library (with its MIDI).`, "ok");
  } catch (err) {
    toast(err.message, "error");
  } finally {
    button.disabled = false;
  }
}

// 32-bit float WAV of an AudioBuffer (the library's native format).
function encodeWavFloat(buffer) {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
  const frames = buffer.length;
  const bytes = frames * channels.length * 4;
  const view = new DataView(new ArrayBuffer(44 + bytes));
  const text = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, 36 + bytes, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 3, true); view.setUint16(22, channels.length, true);
  view.setUint32(24, buffer.sampleRate, true); view.setUint32(28, buffer.sampleRate * channels.length * 4, true);
  view.setUint16(32, channels.length * 4, true); view.setUint16(34, 32, true); text(36, "data"); view.setUint32(40, bytes, true);
  let o = 44;
  for (let i = 0; i < frames; i++) for (const data of channels) { view.setFloat32(o, data[i], true); o += 4; }
  return new Blob([view.buffer], { type: "audio/wav" });
}

// ---------- make it a loop ----------

// The take at a Foundation-1 tempo: stretched from the tempo it was played at, snapped to
// sixteenths, cut to 4 or 8 bars. Returns { notes, bpm, bars, key, scale }.
function takeAsLoop() {
  const playedBpm = Number($("#take-bpm").value) || 120;
  const bpm = nearestLoopBpm(playedBpm);
  const stretch = playedBpm / bpm;
  const notes = quantize(state.take.map((n) => ({ ...n, time: n.time * stretch, dur: n.dur * stretch })), bpm);
  const bars = loopBars(notes, bpm);
  const length = loopSeconds(bpm, bars);
  const fitted = notes.filter((n) => n.time < length - 1e-6).map((n) => ({ ...n, dur: Math.min(n.dur, length - n.time) }));
  return { notes: fitted, bpm, bars, ...detectKey(fitted) };
}

function loopSettings() {
  if (!state.take.length) {
    toast("Record, hum or load a take first.", "info");
    return null;
  }
  const loop = takeAsLoop();
  $("#take-key").textContent = `Sounds like ${loop.key} ${loop.scale} · ${loop.bpm} BPM · ${loop.bars} bars`;
  return loop;
}

function toLoops() {
  const loop = loopSettings();
  if (!loop) return;
  document.dispatchEvent(new CustomEvent("samplegen:loop-settings", { detail: loop }));
}

async function toNewSound() {
  const loop = loopSettings();
  if (!loop) return;
  if (!sampler.ready) return toast("Pick an instrument first: it plays your melody for the AI to hear.", "info");
  const button = $("#take-to-sound");
  if (button.disabled) return;
  button.disabled = true;
  try {
    const rendered = await sampler.render(loop.notes);
    const frames = Math.round(loopSeconds(loop.bpm, loop.bars) * rendered.sampleRate);
    const wav = encodeWavFloat(foldIntoLoop(rendered, frames, ctx));
    const name = `${takeName()} ${loop.key}${loop.scale === "minor" ? "m" : ""} ${loop.bpm}`;
    const source = await api.uploadSource(new File([wav], `${name}.wav`, { type: "audio/wav" }));
    document.dispatchEvent(new CustomEvent("samplegen:melody-source", { detail: { ...loop, source } }));
  } catch (err) {
    toast(err.message, "error");
  } finally {
    button.disabled = false;
  }
}

// ---------- piano roll ----------

function drawRoll(playhead = null) {
  const canvas = $("#play-roll");
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, width, height);
  const notes = state.recording ? state.recording.notes : state.take;
  if (!notes.length) {
    g.fillStyle = getComputedStyle(canvas).getPropertyValue("--screen-ink-3") || "#777";
    g.font = `${14 * dpr}px var(--display, serif)`;
    g.textAlign = "center";
    g.fillText(state.recording ? "Recording… play something" : "Play, record, hum, or drop a MIDI file", width / 2, height / 2);
    return;
  }
  const low = Math.min(...notes.map((n) => n.midi)) - 2;
  const high = Math.max(...notes.map((n) => n.midi)) + 2;
  const length = Math.max(1, takeLength(notes, 0));
  const rowH = height / (high - low + 1);
  const sig = getComputedStyle(canvas).getPropertyValue("--sig").trim() || "#ff5a1f";
  for (const n of notes) {
    const x = (n.time / length) * width;
    const w = Math.max(2 * dpr, (n.dur / length) * width);
    const y = (high - n.midi) * rowH;
    g.globalAlpha = 0.35 + 0.65 * n.vel;
    g.fillStyle = sig;
    g.fillRect(x, y + 1, w, Math.max(2, rowH - 2));
  }
  g.globalAlpha = 1;
  if (playhead !== null) {
    g.fillStyle = "#fff";
    g.fillRect((playhead / length) * width, 0, 1.5 * dpr, height);
  }
}

function animateRoll() {
  if (!state.playback) return drawRoll();
  drawRoll(ctx.currentTime - state.playback.startedAt);
  requestAnimationFrame(animateRoll);
}

// ---------- init ----------

export function initPlay() {
  $("#play-instrument").addEventListener("change", (e) => loadInstrument(e.target.value));
  $("#midi-connect").addEventListener("click", connectKeyboard);
  if (!midiSupported()) $("#midi-label").textContent = "MIDI keyboards need Chrome or Edge.";
  const keys = bindComputerKeys({
    isActive: visible,
    onDown: noteDown,
    onUp: noteUp,
    onOctave: (base) => { $("#play-octave").textContent = noteName(base); },
  });
  $("#play-octave").textContent = noteName(keys.base);
  $("#play-volume").addEventListener("input", (e) => {
    sampler.output.gain.value = Number(e.target.value);
    $("#play-volume-out").textContent = `${Math.round(e.target.value * 100)}%`;
  });
  $("#play-release").addEventListener("input", (e) => {
    sampler.release = Number(e.target.value);
    $("#play-release-out").textContent = `${Number(e.target.value).toFixed(2)} s`;
  });
  $("#play-hold").addEventListener("change", (e) => { sampler.hold = e.target.checked; });
  $("#take-record").addEventListener("click", toggleRecord);
  $("#take-play").addEventListener("click", togglePlay);
  $("#take-clear").addEventListener("click", () => { stopPlayback(); setTake([]); });
  $("#take-hum").addEventListener("click", toggleHum);
  $("#take-load").addEventListener("click", () => $("#take-file").click());
  $("#take-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) loadMidiBuffer(await file.arrayBuffer(), file.name);
  });
  $("#take-save-midi").addEventListener("click", saveMidiFile);
  $("#take-save").addEventListener("click", saveToLibrary);
  $("#take-to-sound").addEventListener("click", toNewSound);
  $("#take-to-loops").addEventListener("click", toLoops);
  bindDrop();
  window.addEventListener("resize", () => { if (visible()) drawRoll(); });
  buildKeys(...DEFAULT_RANGE);
}

export function showPlay() {
  refreshInstruments();
  drawRoll();
  // Reconnect quietly if the browser already allowed MIDI before (no prompt then).
  if (!state.midi && midiSupported() && navigator.permissions) {
    navigator.permissions.query({ name: "midi" }).then((p) => { if (p.state === "granted") connectKeyboard(); }).catch(() => {});
  }
}

export function hidePlay() {
  sampler.allOff();
  stopPlayback();
}
