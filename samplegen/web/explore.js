// Explore: a few words become four contrasting prompts, one take each, to choose from by
// ear. The ideas stay under the prompt: click one to make it the prompt and keep refining.

import { api } from "./api.js";
import { toast } from "./toast.js";

const $ = (sel) => document.querySelector(sel);
export const EXPLORE_MODES = ["sfx", "free", "loop"];

let deps = null;
let busy = false;

function showIdeas(prompts) {
  const list = $("#explore-list");
  list.replaceChildren(...prompts.map((prompt) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip idea";
    chip.textContent = prompt;
    chip.title = "Make this the prompt (then Generate, or Explore again from it)";
    chip.addEventListener("click", () => adopt(chip, prompt));
    return chip;
  }));
  $("#explore-ideas").hidden = false;
}

function adopt(chip, prompt) {
  for (const other of document.querySelectorAll("#explore-list .chip")) other.classList.toggle("on", other === chip);
  const field = $("#prompt");
  field.value = prompt;
  field.dispatchEvent(new Event("input", { bubbles: true })); // saves the form, syncs tag chips
  field.focus();
}

async function generateOne(form, prompt) {
  const body = { ...form, prompt, variations: 1, seed: null };
  deps.track(await api.generate(body), () => generateOne(form, prompt));
}

async function explore() {
  if (busy) return;
  const form = deps.readForm();
  if (!EXPLORE_MODES.includes(form.mode)) return;
  busy = true;
  const button = $("#explore-btn");
  button.disabled = true;
  deps.showError("");
  try {
    const { prompts } = await api.explore({ sketch: form.prompt, mode: form.mode });
    showIdeas(prompts);
    for (const prompt of prompts) await generateOne(form, prompt);
    const what = form.prompt.trim() ? `“${form.prompt.trim()}”` : "something new";
    toast(`${prompts.length} takes on ${what} are on their way. Like one? Click its idea to refine it, or V for more.`, "ok");
  } catch (err) {
    deps.showError(err.message);
  } finally {
    busy = false;
    button.disabled = false;
  }
}

// `readForm()` -> the generate body; `track(job, again)` follows a job in the feed.
export function initExplore({ readForm, track, showError }) {
  deps = { readForm, track, showError };
  $("#explore-btn").addEventListener("click", explore);
  $("#explore-close").addEventListener("click", () => { $("#explore-ideas").hidden = true; });
}

export function syncExplore(mode) {
  $("#explore-btn").hidden = !EXPLORE_MODES.includes(mode);
  $("#explore-ideas").hidden = true; // ideas are written for the mode they came from
}
