// Job feed: one card per job — status LED, ghost waveforms while it runs, then sample rows.

import { api } from "./api.js";
import { icon } from "./icons.js";
import { createSampleRow } from "./samples.js";
import { toast } from "./toast.js";

const $ = (sel) => document.querySelector(sel);
const JOB_POLL_MS = 700;
const EXTRA_LABELS = { demucs: "Stem split" };
const MAX_SKELETONS = 8;

let labelFor = (key) => EXTRA_LABELS[key] ?? key;

export function initFeed(catalog) {
  const labels = Object.fromEntries(catalog.models.map((m) => [m.key, m.label]));
  labelFor = (key) => labels[key] ?? EXTRA_LABELS[key] ?? key;
}

function ghostButton(iconName, text, title, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "ghost";
  b.innerHTML = icon(iconName, 15);
  b.append(text);
  if (title) b.title = title;
  b.addEventListener("click", onClick);
  return b;
}

function skeletonRow() {
  const row = document.createElement("div");
  row.className = "sample skeleton";
  row.setAttribute("aria-hidden", "true");
  const dot = document.createElement("div");
  dot.className = "skel-dot";
  const wave = document.createElement("div");
  wave.className = "skel-wave";
  const lines = document.createElement("div");
  lines.className = "skel-lines";
  lines.append(document.createElement("i"), document.createElement("i"));
  row.append(dot, wave, lines, document.createElement("span"));
  return row;
}

function jobCard(job, again) {
  const card = document.createElement("article");
  card.className = "job";
  card.dataset.state = job.state;
  const head = document.createElement("header");
  const led = document.createElement("i");
  led.className = "led";
  const title = document.createElement("div");
  title.className = "job-title";
  title.textContent = job.prompt;
  title.title = job.prompt;
  const meta = document.createElement("div");
  meta.className = "job-meta";
  const stopBtn = ghostButton("x", "Stop", "Stop this job", () => api.cancel(job.id).catch(() => {}));
  head.append(led, title, meta, stopBtn);
  let againBtn = null;
  if (again) {
    againBtn = ghostButton("shuffle", "More like this", "Same settings, new random seed", again);
    againBtn.hidden = true;
    head.append(againBtn);
  }
  const list = document.createElement("div");
  list.className = "sample-list";
  const count = Math.min(MAX_SKELETONS, Math.max(1, job.variations || 1));
  list.append(...Array.from({ length: job.mode === "stems" ? 4 : count }, skeletonRow));
  card.append(head, list);
  return { card, meta, list, stopBtn, againBtn };
}

// Show a job in the feed and follow it to the end. `again` re-runs it (or null).
export async function track(job, again = null) {
  $("#feed-empty").hidden = true;
  const view = jobCard(job, again);
  $("#feed").prepend(view.card);
  const started = Date.now();
  const label = labelFor(job.model);
  const what = job.mode === "instrument" ? "building" : job.mode === "stems" ? "splitting" : "generating";

  while (!["done", "error", "cancelled"].includes(job.state)) {
    const elapsed = ((Date.now() - started) / 1000).toFixed(0);
    const count = job.variations > 1 ? ` · ${job.variations}×` : "";
    view.meta.textContent = `${label}${count} · ${job.state === "queued" ? "waiting" : what} ${elapsed}s`;
    view.card.dataset.state = job.state;
    await new Promise((r) => setTimeout(r, JOB_POLL_MS));
    try {
      job = await api.job(job.id);
    } catch (err) {
      job = { ...job, state: "error", error: err.message };
    }
  }

  view.card.dataset.state = job.state;
  view.stopBtn.remove();
  if (view.againBtn) view.againBtn.hidden = false;
  const took = job.finished && job.started ? ` · ${(job.finished - job.started).toFixed(1)}s` : "";
  if (job.state === "error") {
    view.list.replaceChildren();
    view.meta.textContent = `${label} · failed`;
    const err = document.createElement("p");
    err.className = "job-error";
    err.textContent = job.error;
    view.card.append(err);
    toast("A generation failed — details in the feed.", "error");
    return;
  }
  if (job.state === "cancelled") {
    view.list.replaceChildren();
    view.meta.textContent = `${label} · stopped`;
    return;
  }
  const n = job.sample_ids.length;
  view.meta.textContent = `${label} · ${n} ${n === 1 ? "take" : "takes"}${took}`;
  if (n === 0) {
    view.list.replaceChildren();
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = job.mode === "stems" ? "Every stem was silent: this sound doesn't separate into parts." : "No audio came back.";
    view.card.append(note);
    return;
  }
  const records = await Promise.all(job.sample_ids.map((id) => api.sample(id)));
  view.list.replaceChildren(...records.map((r) => createSampleRow(r)));
  view.list.querySelector(".sample")?.focus({ preventScroll: true });
  toast(job.mode === "instrument" ? "Instrument ready — the folder icon opens it." : `${n} new ${n === 1 ? "take" : "takes"} ready`, "ok");
}
