// Train view: collect sounds + descriptions, start/stop training, follow progress, list styles.

import { api } from "./api.js";
import { toast } from "./toast.js";

const $ = (sel) => document.querySelector(sel);
const POLL_MS = 2000;
const MAX_CLIPS = 500; // the server's limit (training.py)
const MAX_POLL_FAILURES = 20;
const state = { base: "sfx", clips: [], polling: false, pollFailures: 0 };

const STATE_LABELS = {
  preparing: "Preparing sounds…", training: "Training…", converting: "Converting for the engine…",
  done: "Done", error: "Failed", stopped: "Stopped", idle: "",
};

function showError(message) {
  const el = $("#train-error");
  el.textContent = message || "";
  el.hidden = !message;
}

function syncEmpty() {
  const busy = state.clips.length > 0 || !$("#train-status").hidden;
  $("#train-empty").hidden = busy;
  $("#train-clips-section").hidden = state.clips.length === 0;
}

function renderCount() {
  const included = state.clips.filter((c) => c.include).length;
  $("#train-count").textContent = state.clips.length ? `${included} of ${state.clips.length} selected` : "";
}

function renderClips() {
  syncEmpty();
  const list = $("#train-clips");
  renderCount();
  if (!state.clips.length) {
    list.innerHTML = "";
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "Load a folder or a tag to list the sounds to learn from.";
    list.append(p);
    return;
  }
  list.replaceChildren(...state.clips.map((clip, index) => {
    const row = document.createElement("div");
    row.className = "train-clip";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = clip.include;
    check.title = "Use this sound";
    // Only the count and this row change: re-rendering 500 rows would also drop keyboard focus.
    check.addEventListener("change", () => {
      clip.include = check.checked;
      row.classList.toggle("excluded", !clip.include);
      renderCount();
    });
    const name = document.createElement("span");
    name.className = "clip-name";
    name.textContent = clip.name;
    name.title = clip.path || clip.name;
    const caption = document.createElement("input");
    caption.value = clip.caption;
    caption.maxLength = 400;
    caption.placeholder = "Describe this sound";
    caption.addEventListener("input", () => { clip.caption = caption.value; clip.edited = true; });
    row.append(check, name, caption);
    row.classList.toggle("excluded", !clip.include);
    row.dataset.index = String(index);
    if (clip.fresh) {
      row.classList.add("fresh");
      clip.fresh = false;
    }
    return row;
  }));
}

function addClips(items) {
  const known = new Set(state.clips.map((c) => c.path || c.sample_id));
  let skipped = 0;
  for (const item of items) {
    const key = item.path || item.sample_id;
    if (known.has(key)) continue;
    if (state.clips.length >= MAX_CLIPS) { skipped += 1; continue; }
    state.clips.push({ ...item, include: true });
    known.add(key);
  }
  renderClips();
  if (skipped) toast(`A style can learn from up to ${MAX_CLIPS} sounds: ${skipped} more were left out.`, "info");
}

async function loadFolder() {
  showError("");
  const path = $("#train-folder").value.trim();
  if (!path) return showError("Type the folder's path, e.g. E:\\SOUND\\My Foley");
  try {
    const clips = await api.trainingScan(path);
    if (!clips.length) return showError("No audio files (WAV, AIFF, FLAC, OGG, MP3) in that folder.");
    addClips(clips.map((c) => ({ path: c.path, name: c.name, caption: c.caption })));
  } catch (err) {
    showError(err.message);
  }
}

async function loadTag() {
  showError("");
  const tag = $("#train-tag").value;
  if (!tag) return showError("Choose a tag first.");
  try {
    const records = await api.samples({ tag, limit: 500 });
    addClips(records.map((r) => ({ sample_id: r.id, name: r.name, caption: r.params?.full_prompt || r.prompt })));
  } catch (err) {
    showError(err.message);
  }
}

async function autoDescribe() {
  // Descriptions you typed yourself are kept; only the others are filled in.
  const clips = state.clips.filter((c) => c.include && !c.edited);
  if (!clips.length) return showError("Select at least one sound to describe (ones you've typed a description for are kept).");
  showError("");
  const button = $("#train-describe");
  if (button.disabled) return;
  const label = button.querySelector("span:last-child");
  button.classList.add("busy");
  button.disabled = true;
  label.textContent = `Listening to ${clips.length} sounds…`;
  try {
    const result = await api.trainingDescribe(clips.map((c) => (c.sample_id ? { sample_id: c.sample_id } : { path: c.path })));
    result.captions.forEach((item, i) => {
      if (item.caption) {
        clips[i].caption = item.caption;
        clips[i].fresh = true;
      }
    });
    renderClips();
    toast(result.clap
      ? `Described ${clips.length} sounds — check them, then train`
      : "Described from measurements and file names only (run tools\\install-training.bat for sound recognition)",
    result.clap ? "ok" : "info");
  } catch (err) {
    showError(err.message);
  } finally {
    button.classList.remove("busy");
    button.disabled = false;
    label.textContent = "Auto-describe";
  }
}

async function refreshTags() {
  try {
    const tags = await api.tags();
    $("#train-tag").replaceChildren(new Option("Choose a tag", ""), ...tags.map((t) => new Option(`#${t.tag} (${t.count})`, t.tag)));
  } catch { /* leave as is */ }
}

export async function refreshStyles() {
  let styles = [];
  try {
    styles = await api.styles();
  } catch {
    return [];
  }
  const list = $("#style-list");
  if (!styles.length) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "No styles yet.";
    list.replaceChildren(p);
  } else {
    list.replaceChildren(...styles.map((s) => {
      const row = document.createElement("div");
      row.className = "style-row";
      const name = document.createElement("b");
      name.textContent = s.name;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${s.base === "sfx" ? "SFX" : "Music"} base · ${s.clips} sounds · ${s.steps} steps · ${s.created.slice(0, 10)}`;
      row.append(name, meta);
      return row;
    }));
  }
  document.dispatchEvent(new CustomEvent("samplegen:styles", { detail: styles }));
  return styles;
}

function renderStatus(status) {
  $("#train-not-installed").hidden = status.installed;
  $("#train-start").disabled = !status.installed || status.active;
  const card = $("#train-status");
  card.hidden = status.state === "idle";
  syncEmpty();
  if (card.hidden) return;
  card.dataset.state = status.state;
  $("#train-status-title").textContent = status.name || status.style || "Style";
  const elapsed = status.started ? Math.round(((status.finished || Date.now() / 1000) - status.started) / 60) : 0;
  const steps = status.total_steps ? ` · step ${status.step}/${status.total_steps}` : "";
  $("#train-status-meta").textContent = `${STATE_LABELS[status.state] || status.state}${steps} · ${elapsed} min`;
  const fraction = status.state === "done" ? 1 : status.total_steps ? status.step / status.total_steps : 0;
  $("#train-progress").style.width = `${Math.round(fraction * 100)}%`;
  $("#train-stop").hidden = !status.active;
  const lines = [...status.log_tail];
  if (status.error) lines.push(`ERROR: ${status.error}`);
  $("#train-log").textContent = lines.join("\n");
}

async function poll() {
  let status;
  try {
    status = await api.training();
    state.pollFailures = 0;
  } catch {
    // The app may be busy or restarting: keep trying for a while instead of freezing the card.
    state.pollFailures += 1;
    if (state.pollFailures >= MAX_POLL_FAILURES) {
      state.polling = false;
      $("#train-status-meta").textContent = "Lost contact with samplegen — reopen this tab to check again.";
      return;
    }
    setTimeout(poll, POLL_MS * Math.min(state.pollFailures, 5));
    return;
  }
  renderStatus(status);
  if (status.active) {
    state.sawActive = true;
    setTimeout(poll, POLL_MS);
  } else {
    state.polling = false;
    if (status.state === "done") refreshStyles();
    if (state.sawActive) {  // only announce endings we watched happen
      if (status.state === "done") toast(`Style “${status.name}” is ready — pick it in Generate`, "ok");
      if (status.state === "error") toast("Training failed — see the log in the Train tab.", "error");
    }
    state.sawActive = false;
  }
}

function startPolling() {
  if (!state.polling) {
    state.polling = true;
    poll();
  }
}

async function start(e) {
  e.preventDefault();
  showError("");
  const clips = state.clips.filter((c) => c.include);
  if (!clips.length) return showError("Select at least one sound (tick the boxes in the list).");
  if (clips.some((c) => !c.caption.trim())) return showError("Every selected sound needs a description.");
  const body = {
    name: $("#train-name").value.trim(),
    base: state.base,
    description: $("#train-desc").value.trim(),
    steps: Number($("#train-steps").value),
    rank: 16,
    clips: clips.map((c) => (c.sample_id ? { sample_id: c.sample_id, caption: c.caption } : { path: c.path, caption: c.caption })),
  };
  if (!body.name) return showError("Give the style a name.");
  const button = $("#train-start");
  if (button.disabled) return;
  button.disabled = true; // a double-click would send a second start and show a bogus error
  try {
    renderStatus(await api.trainingStart(body));
    startPolling();
  } catch (err) {
    showError(err.message);
    button.disabled = false;
  }
}

export function initTrain() {
  for (const b of document.querySelectorAll("#train-base button")) {
    b.addEventListener("click", () => {
      state.base = b.dataset.base;
      for (const o of document.querySelectorAll("#train-base button")) o.classList.toggle("active", o === b);
    });
  }
  $("#train-steps").addEventListener("input", () => { $("#train-steps-out").textContent = $("#train-steps").value; });
  $("#train-scan").addEventListener("click", loadFolder);
  $("#train-folder").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); loadFolder(); } });
  $("#train-from-tag").addEventListener("click", loadTag);
  $("#train-describe").addEventListener("click", autoDescribe);
  $("#train-stop").addEventListener("click", async () => {
    try { renderStatus(await api.trainingStop()); } catch (err) { showError(err.message); }
  });
  $("#train-form").addEventListener("submit", start);
  refreshStyles();
  startPolling();  // picks up a training that was already running
}

export function showTrain() {
  refreshTags();
  refreshStyles();
  startPolling();
}
