// samplegen UI bootstrap: generate form, modes, keyboard, engine status.

import { api } from "./api.js";
import { initFeed, track } from "./feed.js";
import { icon } from "./icons.js";
import { initLibrary, setSimilar, showLibrary } from "./library.js";
import { hidePlay, initPlay, showPlay } from "./play.js";
import { toast } from "./toast.js";
import * as player from "./player.js";
import * as sourcePanel from "./source.js";
import { initTrain, showTrain } from "./train.js";

const $ = (sel) => document.querySelector(sel);
const STORAGE_KEY = "samplegen.form.v1";
const STATUS_POLL_MS = 3000;
const SOURCE_MODES = ["transform", "edit"];
const TAG_MODES = { loop: null, instrument: ["Instrument", "Timbre"] }; // null = every group
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const SECONDS_PER_PASS = 18; // Keybeds pass (6 notes) on an RTX 3070 laptop, roughly

const PROMPT_COPY = {
  sfx: ["Describe the sound", "heavy metal door slam in a concrete hallway, long reverb tail"],
  loop: ["Tags (instrument, timbre, FX, structure)", "Synth Bass, FM Bass, Gritty, Acid, Medium Reverb, Bassline"],
  free: ["Describe the sound", "dark evolving ambient drone, granular, metallic shimmer"],
  transform: ["What should it become?", "rusty metal, industrial, resonant clank"],
  edit: ["What goes in the new part?", "glass shattering, bright debris"],
  instrument: ["Describe the instrument (instrument + timbre tags)", "Grand Piano, Warm, Gritty"],
};

const STYLE_MODES = ["sfx", "free"];
const SEAMLESS_MODES = ["sfx", "free"];
const MIN_SEAMLESS_SECONDS = 6;
const VIEWS = ["generate", "library", "play", "train"];

const state = { catalog: null, mode: "sfx", scale: "minor", editOp: "inpaint", space: "Dry", styles: [] };

// ---------- chrome: icons, tab indicator, faders, finish ----------
const THEME_KEY = "samplegen.theme";

function hydrateIcons(root = document) {
  for (const el of root.querySelectorAll("[data-icon]")) {
    if (!el.firstElementChild) el.innerHTML = icon(el.dataset.icon);
  }
}

function moveTabInk() {
  const active = document.querySelector(".tab.active");
  const ink = document.querySelector(".tab-ink");
  if (!active || !ink) return;
  ink.style.left = `${active.offsetLeft}px`;
  ink.style.width = `${active.offsetWidth}px`;
}

function setFill(range) {
  const min = Number(range.min || 0);
  const max = Number(range.max || 100);
  const pct = max > min ? ((Number(range.value) - min) / (max - min)) * 100 : 0;
  range.style.setProperty("--fill", `${pct}%`);
}

function refreshRanges() {
  for (const r of document.querySelectorAll('input[type="range"]')) setFill(r);
}

function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function syncThemeButton() {
  const button = $("#theme-toggle");
  const dark = currentTheme() === "dark";
  button.innerHTML = icon(dark ? "sun" : "moon");
  button.title = dark ? "Bone finish (light)" : "Graphite finish (dark)";
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch { /* per-browser only */ }
  syncThemeButton();
}

// ---------- trained styles ----------
function syncStyleField() {
  const select = $("#style");
  const usable = STYLE_MODES.includes(state.mode) && state.styles.length > 0;
  $("#style-field").hidden = !usable;
  const style = state.styles.find((s) => s.slug === select.value);
  $("#style-strength-field").hidden = !usable || !style;
  $("#style-hint").textContent = style
    ? `Runs on the ${style.base === "sfx" ? "SFX" : "Music"} base model (50 steps: slower than stock, ~10–20 s).`
    : "";
  $("#style-strength-out").textContent = Number($("#style-strength").value).toFixed(2);
}

function setStyles(styles, keep) {
  state.styles = styles;
  const select = $("#style");
  const current = keep ?? select.value;
  select.replaceChildren(new Option("None (stock model)", ""), ...styles.map((s) => new Option(s.name, s.slug)));
  if (styles.some((s) => s.slug === current)) select.value = current;
  syncStyleField();
}

// ---------- persistence (per-browser convenience only) ----------
function saveForm() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(readForm()));
  } catch { /* storage unavailable: ignore */ }
}

function loadSavedForm() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  } catch {
    return null;
  }
}

// ---------- notes ----------
const midiName = (m) => `${NOTE_NAMES[m % 12]}${Math.floor(m / 12) - 1}`;
const nameMidi = (n) => {
  const match = /^([A-G]#?)(-?\d)$/.exec(n);
  return match ? (Number(match[2]) + 1) * 12 + NOTE_NAMES.indexOf(match[1]) : NaN;
};

function updateInstrumentEstimate() {
  const low = nameMidi($("#low-note").value);
  const high = nameMidi($("#high-note").value);
  const el = $("#instrument-estimate");
  if (!(high >= low)) {
    el.textContent = "The highest note must be above the lowest.";
    return;
  }
  const notes = high - low + 1;
  const passes = Math.ceil(notes / 6);
  const minutes = Math.max(1, Math.round((passes * SECONDS_PER_PASS) / 60));
  el.textContent = `${notes} notes · ${passes} passes · about ${minutes} min`;
}

// ---------- form ----------
function fillSelect(select, values, format = (v) => v) {
  select.replaceChildren(...values.map((v) => new Option(format(v), v)));
}

function setMode(mode, preferredModel) {
  state.mode = mode;
  for (const b of document.querySelectorAll("#mode-switch button")) b.classList.toggle("active", b.dataset.mode === mode);
  const models = state.catalog.models.filter((m) => m.modes.includes(mode));
  const select = $("#model");
  fillSelect(select, models.map((m) => m.key), (k) => models.find((m) => m.key === k).label);
  if (preferredModel && models.some((m) => m.key === preferredModel)) select.value = preferredModel;

  const usesSource = SOURCE_MODES.includes(mode);
  const isInstrument = mode === "instrument";
  $("#loop-fields").hidden = mode !== "loop";
  $("#instrument-fields").hidden = !isInstrument;
  $("#duration-field").hidden = mode === "loop" || usesSource || isInstrument;
  $("#variations-field").hidden = isInstrument;
  $("#shape-section").hidden = isInstrument;
  $("#trim-field").hidden = mode === "loop" || usesSource || isInstrument;
  $("#seamless-field").hidden = !SEAMLESS_MODES.includes(mode);
  $("#source-panel").hidden = !usesSource;
  $("#transform-fields").hidden = mode !== "transform";
  $("#edit-fields").hidden = mode !== "edit";

  const tagGroups = $("#tag-groups");
  tagGroups.hidden = !(mode in TAG_MODES);
  for (const group of tagGroups.querySelectorAll(".tag-group")) {
    const allowed = TAG_MODES[mode];
    group.hidden = Boolean(allowed) && !allowed.includes(group.dataset.group);
  }

  const [label, placeholder] = PROMPT_COPY[mode];
  $("#prompt-label").textContent = label;
  $("#prompt").placeholder = placeholder;
  sourcePanel.setSelectable(mode === "edit" && state.editOp === "inpaint");
  onModelChange();
  syncTagChips();
  syncStyleField();
  numberSections();
}

// Front-panel numbering follows whatever sections the current mode shows.
function numberSections() {
  let n = 0;
  for (const label of document.querySelectorAll("#gen-form .deck-section:not([hidden]) > .deck-label > span:first-child")) {
    n += 1;
    label.textContent = String(n).padStart(2, "0");
  }
}

function setChoice(switchId, attr, value) {
  for (const b of document.querySelectorAll(`#${switchId} button`)) b.classList.toggle("active", b.dataset[attr] === value);
}

function setEditOp(op) {
  state.editOp = op;
  setChoice("edit-op-switch", "op", op);
  $("#extend-field").hidden = op !== "extend";
  sourcePanel.setSelectable(state.mode === "edit" && op === "inpaint");
}

function setScale(scale) {
  state.scale = scale;
  setChoice("scale-switch", "scale", scale);
}

function setSpace(space) {
  state.space = space;
  setChoice("space-switch", "space", space);
}

function onSourceChange() {
  const source = sourcePanel.getSource();
  if (source) $("#source-loop").checked = source.is_loop;
  updateOutputs();
}

function onModelChange() {
  const model = state.catalog.models.find((m) => m.key === $("#model").value);
  if (!model) return;
  $("#model-desc").textContent = model.description;
  const slider = $("#duration");
  const max = Math.min(model.max_seconds, 60);
  slider.max = String(max);
  if (Number(slider.value) > max) slider.value = String(max);
  updateOutputs();
}

function updateOutputs() {
  const seconds = Number($("#duration").value);
  $("#duration-out").textContent = seconds >= 60
    ? `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")} min`
    : `${seconds.toFixed(1)} s`;
  $("#variations-out").textContent = $("#variations").value;
  $("#strength-out").textContent = Number($("#strength").value).toFixed(2);
  $("#style-strength-out").textContent = Number($("#style-strength").value).toFixed(2);
  refreshRanges();
  const source = sourcePanel.getSource();
  const extend = Number($("#extend").value);
  $("#extend-out").textContent = source
    ? `${extend.toFixed(1)} s → ${(source.duration + extend).toFixed(1)} s total`
    : `${extend.toFixed(1)} s`;
  updateInstrumentEstimate();
}

// ---------- tag chips ----------
function currentTags() {
  return $("#prompt").value.split(",").map((t) => t.trim()).filter(Boolean);
}

function syncTagChips() {
  const active = new Set(currentTags().map((t) => t.toLowerCase()));
  for (const chip of document.querySelectorAll(".chip")) chip.classList.toggle("on", active.has(chip.textContent.toLowerCase()));
}

function toggleTag(tag) {
  const tags = currentTags();
  const idx = tags.findIndex((t) => t.toLowerCase() === tag.toLowerCase());
  $("#prompt").value = (idx >= 0 ? tags.filter((_, i) => i !== idx) : [...tags, tag]).join(", ");
  syncTagChips();
  saveForm();
}

function buildTagGroups(groups) {
  const container = $("#tag-groups");
  container.replaceChildren();
  for (const [group, tags] of Object.entries(groups)) {
    const wrap = document.createElement("details");
    wrap.className = "tag-group";
    wrap.dataset.group = group;
    wrap.open = group === "Instrument";
    const summary = document.createElement("summary");
    summary.textContent = group;
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const tag of tags) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = tag;
      chip.addEventListener("click", () => toggleTag(tag));
      chips.append(chip);
    }
    wrap.append(summary, chips);
    container.append(wrap);
  }
}

// ---------- read / restore ----------
function exportFormat() {
  return { sample_rate: Number($("#export-rate").value), bit_depth: $("#export-bits").value };
}

function readForm() {
  const seed = $("#seed").value.trim();
  const source = sourcePanel.getSource();
  const selection = sourcePanel.getSelection();
  return {
    mode: state.mode,
    model: $("#model").value,
    prompt: $("#prompt").value,
    negative_prompt: $("#negative").value,
    duration: Number($("#duration").value),
    bpm: Number($("#bpm").value),
    bars: Number($("#bars").value),
    key: $("#key").value,
    scale: state.scale,
    variations: Number($("#variations").value),
    seed: seed === "" ? null : Number(seed),
    normalize_db: $("#normalize").checked ? Number($("#normalize-db").value) : null,
    trim_silence: $("#trim").checked,
    fade_in_ms: Number($("#fade-in").value) || 0,
    fade_out_ms: Number($("#fade-out").value) || 0,
    export: exportFormat(),
    source_id: SOURCE_MODES.includes(state.mode) && source ? source.id : null,
    strength: Number($("#strength").value),
    source_is_loop: $("#source-loop").checked,
    edit_op: state.editOp,
    regions: selection ? [[selection.start, selection.end]] : [],
    extend_seconds: Number($("#extend").value),
    instrument_name: $("#instrument-name").value,
    low_note: $("#low-note").value,
    high_note: $("#high-note").value,
    space: state.space,
    style: STYLE_MODES.includes(state.mode) && $("#style").value ? $("#style").value : null,
    style_strength: Number($("#style-strength").value),
    seamless: SEAMLESS_MODES.includes(state.mode) && $("#seamless").checked,
  };
}

function applyForm(saved) {
  if (!saved) return setMode("sfx");
  const set = (sel, v) => {
    if (v === undefined || v === null) return;
    const el = $(sel);
    // A <select> given a value it no longer offers ends up blank (and Generate then fails): keep its default.
    if (el.tagName === "SELECT" && ![...el.options].some((o) => o.value === String(v))) return;
    el.value = v;
  };
  const known = state.catalog.models.some((m) => m.modes.includes(saved.mode));
  setMode(known ? saved.mode : "sfx", saved.model);
  for (const [sel, key] of [["#prompt", "prompt"], ["#negative", "negative_prompt"], ["#duration", "duration"],
    ["#bpm", "bpm"], ["#bars", "bars"], ["#key", "key"], ["#variations", "variations"], ["#fade-in", "fade_in_ms"],
    ["#fade-out", "fade_out_ms"], ["#strength", "strength"], ["#extend", "extend_seconds"],
    ["#instrument-name", "instrument_name"], ["#low-note", "low_note"], ["#high-note", "high_note"]]) {
    set(sel, saved[key]);
  }
  setEditOp(saved.edit_op === "extend" ? "extend" : "inpaint");
  setSpace(saved.space === "Wet" ? "Wet" : "Dry");
  if (saved.export) {
    set("#export-rate", saved.export.sample_rate);
    set("#export-bits", saved.export.bit_depth);
  }
  $("#normalize").checked = saved.normalize_db !== null;
  if (saved.normalize_db !== null && saved.normalize_db !== undefined) set("#normalize-db", saved.normalize_db);
  $("#trim").checked = saved.trim_silence !== false;
  $("#seamless").checked = Boolean(saved.seamless);
  set("#style-strength", saved.style_strength);
  state.savedStyle = saved.style;
  setScale(saved.scale || "minor");
  updateOutputs();
  syncTagChips();
}

function showFormError(message) {
  const el = $("#form-error");
  el.textContent = message || "";
  el.hidden = !message;
}

// ---------- submitting ----------
// While a request is on its way, a second click / Ctrl+Enter is a double-click, not a new batch.
let submitting = false;
const splitting = new Set();

async function submit(body) {
  if (submitting) return;
  submitting = true;
  showFormError("");
  let job;
  try {
    job = body.mode === "instrument"
      ? await api.instrument({
        prompt: body.prompt, name: body.instrument_name, low_note: body.low_note, high_note: body.high_note,
        space: body.space, seed: body.seed, export: body.export,
      })
      : await api.generate(body);
  } catch (err) {
    showFormError(err.message);
    return;
  } finally {
    submitting = false;
  }
  track(job, () => submit({ ...body, seed: null }));
}

async function useSampleAsSource(record) {
  showView("generate");
  if (!SOURCE_MODES.includes(state.mode)) setMode("transform");
  try {
    sourcePanel.setSource(await api.sourceFromSample(record.id));
    showFormError("");
    toast(`“${record.name}” loaded as the source`, "ok");
  } catch (err) {
    showFormError(err.message);
  }
}

// From the Play tab: Loop mode in the key and tempo of a take.
function applyLoopSettings({ key, scale, bpm, bars }) {
  showView("generate");
  setMode("loop");
  $("#bpm").value = String(bpm);
  $("#bars").value = String(bars);
  $("#key").value = key;
  setScale(scale);
  updateOutputs();
  saveForm();
  $("#prompt").focus();
  toast(`Loop mode set to ${key} ${scale}, ${bpm} BPM, ${bars} bars. Pick instruments and Generate.`, "ok");
}

// From the Play tab: the take's melody as a loop source; describe the sound it should become.
function applyMelodySource({ source, key, scale, bpm, bars }) {
  showView("generate");
  setMode("transform", "f1-samples"); // Foundation-1 keeps notes, tempo and key
  $("#bpm").value = String(bpm);
  $("#bars").value = String(bars);
  $("#key").value = key;
  setScale(scale);
  sourcePanel.setSource(source);
  $("#source-loop").checked = true;
  $("#strength").value = "0.55"; // enough to change the sound, gentle enough to keep the notes
  $("#strength").dispatchEvent(new Event("input", { bubbles: true }));
  $("#prompt").value = "";
  $("#prompt").placeholder = "The sound for your melody: Supersaw Lead, Warm, Wide · Plucked Marimba · Gritty Reese Bass…";
  updateOutputs();
  saveForm();
  $("#prompt").focus();
  toast(`Your melody (${key} ${scale}, ${bpm} BPM) is the source. Describe the sound and Generate; raise Strength for wilder results.`, "ok");
}

async function splitStems(record) {
  if (splitting.has(record.id)) return;
  splitting.add(record.id);
  showView("generate");
  try {
    track(await api.stems(record.id, exportFormat()));
    toast(`Splitting “${record.name}” into stems…`);
  } catch (err) {
    toast(err.message, "error");
  } finally {
    splitting.delete(record.id);
  }
}

function showView(name) {
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.view === name);
  for (const view of VIEWS) $(`#view-${view}`).hidden = view !== name;
  moveTabInk();
  player.stop();
  if (name === "library") showLibrary();
  if (name === "train") showTrain();
  if (name === "play") showPlay();
  else hidePlay();
}

// ---------- keyboard ----------
function focusedRow() {
  return document.activeElement?.closest?.(".sample") ?? null;
}

function visibleRows() {
  return [...document.querySelectorAll(".view:not([hidden]) .sample:not(.skeleton)")];
}

function moveFocus(delta) {
  const visible = visibleRows();
  if (!visible.length) return;
  const idx = visible.indexOf(focusedRow());
  const next = visible[Math.max(0, Math.min(visible.length - 1, idx + delta))] ?? visible[0];
  next.focus();
  next.scrollIntoView({ block: "nearest" });
}

const ROW_KEYS = { k: "keep", x: "trash", f: "favorite", t: "useAsSource", s: "stems", e: "edit", m: "midi", l: "similar" };

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    if ($("#view-generate").hidden) return; // the form isn't on screen: don't generate blind
    e.preventDefault();
    $("#gen-form").requestSubmit();
    return;
  }
  if (e.target.matches?.("input, textarea, select") || e.ctrlKey || e.metaKey || e.altKey) return;
  const row = focusedRow();
  const actions = row?.sampleActions;
  // Space/Enter on a focused button (Keep, Trash...) should press that button, not play.
  const onButton = e.target.closest?.("button") && e.target !== row;
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    if (!visibleRows().length) return;
    e.preventDefault();
    moveFocus(e.key === "ArrowDown" ? 1 : -1);
  } else if (e.key === " " && actions && !onButton) {
    e.preventDefault();
    actions.toggle();
  } else if (e.key === "Escape") {
    player.stop();
  } else if (actions && ROW_KEYS[e.key.toLowerCase()]) {
    e.preventDefault();
    actions[ROW_KEYS[e.key.toLowerCase()]]();
  }
});

// ---------- engine status ----------
async function pollStatus() {
  const pill = $("#engine-pill");
  try {
    const s = await api.status();
    pill.dataset.state = s.engine;
    const labels = { ready: "Engine ready", starting: "Engine starting…", stopped: "Engine idle", error: "Engine error" };
    pill.querySelector(".label").textContent = labels[s.engine] || s.engine;
    pill.title = s.engine_error || `Library: ${s.library}`;
    const missing = s.missing || [];
    $("#setup-banner").hidden = missing.length === 0;
    $("#setup-missing").textContent = missing.length ? `Missing: ${missing.join(" · ")}.` : "";
    $("#setup-fix").textContent = s.missing_fix || "";
  } catch {
    pill.dataset.state = "error";
    pill.querySelector(".label").textContent = "App offline";
  }
  setTimeout(pollStatus, STATUS_POLL_MS);
}

// ---------- init ----------
async function init() {
  hydrateIcons();
  syncThemeButton();
  $("#theme-toggle").addEventListener("click", toggleTheme);
  moveTabInk();
  document.fonts?.ready.then(moveTabInk);
  window.addEventListener("resize", moveTabInk);
  document.addEventListener("input", (e) => { if (e.target.matches?.('input[type="range"]')) setFill(e.target); });
  state.catalog = await api.catalog();
  const { loop, instrument } = state.catalog;
  fillSelect($("#bpm"), loop.bpms);
  fillSelect($("#bars"), loop.bars);
  fillSelect($("#key"), loop.keys);
  $("#bpm").value = "128";
  const notes = [];
  for (let m = nameMidi(instrument.lowest); m <= nameMidi(instrument.highest); m++) notes.push(midiName(m));
  fillSelect($("#low-note"), notes);
  fillSelect($("#high-note"), notes);
  $("#low-note").value = "C3";
  $("#high-note").value = "B4";
  buildTagGroups(loop.tags);
  initFeed(state.catalog);
  initLibrary();
  sourcePanel.initSourcePanel({ onChange: onSourceChange });
  applyForm(loadSavedForm());

  const bind = (sel, fn) => { for (const b of document.querySelectorAll(sel)) b.addEventListener("click", () => { fn(b); saveForm(); }); };
  bind("#mode-switch button", (b) => setMode(b.dataset.mode));
  bind("#scale-switch button", (b) => setScale(b.dataset.scale));
  bind("#edit-op-switch button", (b) => setEditOp(b.dataset.op));
  bind("#space-switch button", (b) => setSpace(b.dataset.space));
  for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => showView(t.dataset.view));
  document.addEventListener("samplegen:use-source", (e) => useSampleAsSource(e.detail));
  document.addEventListener("samplegen:stems", (e) => splitStems(e.detail));
  document.addEventListener("samplegen:similar", (e) => { setSimilar(e.detail); showView("library"); });
  document.addEventListener("samplegen:loop-settings", (e) => applyLoopSettings(e.detail));
  document.addEventListener("samplegen:melody-source", (e) => applyMelodySource(e.detail));
  document.addEventListener("samplegen:styles", (e) => {
    setStyles(e.detail, state.savedStyle);
    state.savedStyle = undefined;
  });
  $("#style").addEventListener("change", () => { syncStyleField(); saveForm(); });
  $("#seamless").addEventListener("change", () => {
    // A seamless loop needs room for its 2 s crossfade: lift a one-shot length to something ambient.
    const duration = $("#duration");
    if ($("#seamless").checked && Number(duration.value) < MIN_SEAMLESS_SECONDS) {
      duration.value = String(Math.min(30, Number(duration.max)));
      setFill(duration);
      updateOutputs();
    }
  });
  initTrain();
  initPlay();
  for (const chip of document.querySelectorAll("#try-prompts .chip")) {
    chip.addEventListener("click", () => {
      if (state.mode !== "sfx") setMode("sfx");
      $("#prompt").value = chip.dataset.prompt;
      $("#prompt").dispatchEvent(new Event("input", { bubbles: true }));
      $("#prompt").focus();
    });
  }

  $("#model").addEventListener("change", () => { onModelChange(); saveForm(); });
  $("#prompt").addEventListener("input", syncTagChips);
  $("#gen-form").addEventListener("input", () => { updateOutputs(); saveForm(); });
  $("#gen-form").addEventListener("submit", (e) => {
    e.preventDefault();
    saveForm();
    submit(readForm());
  });
  refreshRanges();
  pollStatus();
}

init().catch((err) => showFormError(`Could not reach samplegen: ${err.message}`));
