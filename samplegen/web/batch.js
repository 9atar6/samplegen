// Shot-list batches: a dialog to write the list, one progress card per batch in the feed,
// and an optional pack export (loudness-matched, round robins) when everything is done.

import { api } from "./api.js";
import { toast } from "./toast.js";

const $ = (sel) => document.querySelector(sel);
const POLL_MS = 3000;
const FINISHED = ["done", "error", "cancelled"];
const EXAMPLE = `# One sound per line. Options after "|": 8x (how many), 3s (length), medium (best model), loop (seamless)
door creak x8
heavy footsteps on gravel | 12x | 2s
sci-fi UI confirm beep | 20x | 0.5s
distant thunder rumble | 6x | 12s | medium
forest ambience with birds | 2x | 60s | medium | loop`;

let previewTimer = null;

async function preview() {
  const info = $("#batch-info");
  const text = $("#batch-text").value;
  if (!text.trim()) { info.textContent = ""; return; }
  try {
    const p = await api.previewBatch({ text });
    info.textContent = p.ok ? `${p.sounds} sounds from ${p.lines} lines` : p.error;
    info.classList.toggle("error", !p.ok);
  } catch { /* the app may be busy; the next keystroke tries again */ }
}

function openDialog() {
  const dialog = $("#batch-dialog");
  if (!$("#batch-text").value.trim()) $("#batch-text").value = EXAMPLE;
  dialog.showModal();
  preview();
}

async function start(e) {
  e.preventDefault();
  const button = $("#batch-start");
  if (button.disabled) return;
  button.disabled = true;
  try {
    const name = $("#batch-name").value.trim() || `batch ${new Date().toLocaleDateString()}`;
    const batch = await api.startBatch({
      name, text: $("#batch-text").value, model: $("#batch-model").value, seconds: Number($("#batch-seconds").value) || 4,
    });
    const exportWhenDone = $("#batch-export").checked ? {
      loudness: $("#batch-loudness").value ? Number($("#batch-loudness").value) : null,
      round_robin: $("#batch-rr").checked,
    } : null;
    $("#batch-dialog").close();
    follow(name, batch, exportWhenDone);
    toast(`${batch.sounds} sounds queued. They'll also be tagged #${batch.tag} in the Library.`, "ok");
  } catch (err) {
    $("#batch-info").textContent = err.message;
    $("#batch-info").classList.add("error");
  } finally {
    button.disabled = false;
  }
}

// One card for the whole batch: progress, then (optionally) the exported pack.
async function follow(name, batch, exportWhenDone) {
  $("#feed-empty").hidden = true;
  const card = document.createElement("article");
  card.className = "job batch-card";
  card.dataset.state = "running";
  const head = document.createElement("header");
  const led = document.createElement("i");
  led.className = "led";
  const title = document.createElement("div");
  title.className = "job-title";
  title.textContent = `Shot list · ${name}`;
  const meta = document.createElement("div");
  meta.className = "job-meta";
  head.append(led, title, meta);
  const bar = document.createElement("div");
  bar.className = "progress";
  const fill = document.createElement("div");
  fill.className = "progress-bar";
  bar.append(fill);
  const note = document.createElement("p");
  note.className = "hint";
  note.textContent = "The PC stays awake until the batch is done (the screen may still turn off). Results are in the Library under the batch's tag.";
  card.append(head, bar, note);
  $("#feed").prepend(card);


  const started = Date.now();
  let jobs = batch.jobs;
  let failures = 0;
  while (jobs.some((j) => !FINISHED.includes(j.state))) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    try {
      const pending = jobs.filter((j) => !FINISHED.includes(j.state)).map((j) => j.id);
      const byId = new Map((await api.jobsByIds(pending)).map((j) => [j.id, j]));
      jobs = jobs.map((j) => byId.get(j.id) ?? j);
      failures = 0;
    } catch {
      if (++failures > 40) break; // ~2 minutes without an answer: stop following, the jobs go on
    }
    const done = jobs.filter((j) => FINISHED.includes(j.state));
    const takes = jobs.reduce((n, j) => n + (j.sample_ids?.length || 0), 0);
    const minutes = Math.round((Date.now() - started) / 60000);
    meta.textContent = `${done.length}/${jobs.length} jobs · ${takes} of ${batch.sounds} sounds · ${minutes} min`;
    fill.style.width = `${Math.round((done.length / jobs.length) * 100)}%`;
  }

  const sampleIds = jobs.flatMap((j) => j.sample_ids || []);
  const failed = jobs.filter((j) => j.state === "error").length;
  card.dataset.state = failed && !sampleIds.length ? "error" : "done";
  meta.textContent = `${sampleIds.length} sounds${failed ? ` · ${failed} jobs failed` : ""}`;
  fill.style.width = "100%";
  if (!exportWhenDone || !sampleIds.length) {
    note.textContent = `Done. Find them in the Library (tag #${batch.tag}).`;
    toast(`Shot list “${name}” finished: ${sampleIds.length} sounds.`, "ok");
    return;
  }
  note.textContent = "Exporting the pack…";
  try {
    const pack = await api.exportPack({ name, sample_ids: sampleIds, export: null, reveal: true, ...exportWhenDone });
    note.textContent = `Pack exported: ${pack.folder} (${pack.exported} sounds).`;
    toast(`Shot list “${name}” done and exported as a pack.`, "ok");
  } catch (err) {
    note.textContent = `Finished, but the pack export failed: ${err.message}`;
  }
}

export function initBatch() {
  $("#batch-open").addEventListener("click", openDialog);
  $("#batch-cancel").addEventListener("click", () => $("#batch-dialog").close());
  $("#batch-form").addEventListener("submit", start);
  $("#batch-text").addEventListener("input", () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(preview, 400);
  });
  $("#batch-export").addEventListener("change", (e) => { $("#batch-export-options").hidden = !e.target.checked; });
}
