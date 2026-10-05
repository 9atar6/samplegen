// Library view: status filters, search, tag filter, sample-pack export.

import { api } from "./api.js";
import { toast } from "./toast.js";
import { createSampleRow, removeRow } from "./samples.js";

const $ = (sel) => document.querySelector(sel);
const LIST_LIMIT = 2000;
const state = { filter: "new", shown: [], bySound: false, similar: null };
let searchTimer = null;
let indexTimer = null;

async function refreshTags() {
  const select = $("#lib-tag");
  const current = select.value;
  try {
    const tags = await api.tags();
    select.replaceChildren(new Option("All tags", ""), ...tags.map((t) => new Option(`#${t.tag} (${t.count})`, t.tag)));
    if (tags.some((t) => t.tag === current)) select.value = current;
  } catch { /* keep whatever is there */ }
}

let latestLoad = 0; // only the newest request may render (filters can change mid-request)

export async function loadLibrary() {
  const ticket = ++latestLoad;
  const filter = state.filter;
  const params = filter === "favorites" ? { favorite: true } : { status: filter };
  const q = $("#lib-search").value.trim();
  const tag = $("#lib-tag").value;
  if (q) params.q = q;
  if (tag) params.tag = tag;
  params.limit = LIST_LIMIT;
  const list = $("#lib-list");
  let records;
  const soundQuery = state.bySound && q;
  $("#lib-similar").hidden = !state.similar;
  try {
    if (state.similar) {
      records = await api.similar(state.similar.id);
      $("#lib-similar-label").textContent = `Sounds most like “${state.similar.name}”`;
    } else if (soundQuery) {
      records = await api.searchSound(q); // ranked by how much they sound like the words
    } else {
      records = await api.samples(params);
    }
  } catch (err) {
    if (ticket !== latestLoad) return;
    $("#lib-empty").textContent = err.message;
    $("#lib-empty").hidden = false;
    return;
  }
  if (ticket !== latestLoad) return; // a newer search/filter is on its way
  for (const el of list.querySelectorAll(".sample")) removeRow(el.dataset.id, list);
  const ranked = Boolean(state.similar || soundQuery); // ranked results span the whole library (but trash)
  const matches = (r) => (ranked ? r.status !== "trashed"
    : (filter === "favorites" ? r.favorite && r.status !== "trashed" : r.status === filter)
      && (!tag || (r.tags || []).includes(tag)));
  state.shown = records.map((r) => r.id);
  list.replaceChildren(...records.map((r) => createSampleRow(r, {
    onChange: (updated) => {
      if (!matches(updated)) {
        removeRow(updated.id, list);
        state.shown = state.shown.filter((id) => id !== updated.id);
      }
      refreshTags();
      updatePackCount();
    },
  })));
  $("#lib-empty").textContent = "Nothing here yet.";
  $("#lib-empty").hidden = records.length > 0;
  updatePackCount();
}

function updatePackCount() {
  $("#pack-count").textContent = `${state.shown.length} sample${state.shown.length === 1 ? "" : "s"}`;
}

function bindPackDialog() {
  const dialog = $("#pack-dialog");
  const result = $("#pack-result");
  $("#pack-open").addEventListener("click", () => {
    dialog.hidden = !dialog.hidden;
    result.hidden = true;
    if (!dialog.hidden) $("#pack-name").focus();
  });
  $("#pack-cancel").addEventListener("click", () => { dialog.hidden = true; });
  dialog.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.shown.length) {
      result.textContent = "Nothing to export: the list below is empty.";
      result.hidden = false;
      return;
    }
    const format = $("#pack-format").value;
    const [rate, bits] = format ? format.split(":") : [];
    const body = {
      name: $("#pack-name").value,
      sample_ids: state.shown,
      export: format ? { sample_rate: Number(rate), bit_depth: bits } : null,
      loudness: $("#pack-loudness").value ? Number($("#pack-loudness").value) : null,
      round_robin: $("#pack-rr").checked,
      reveal: true,
    };
    $("#pack-export").disabled = true;
    result.hidden = false;
    result.textContent = `Exporting ${state.shown.length} samples…`;
    try {
      const pack = await api.exportPack(body);
      const missing = pack.missing ? ` (${pack.missing} missing files skipped)` : "";
      result.textContent = `Exported ${pack.exported} samples to ${pack.folder}${missing}`;
      toast(`Pack “${pack.name}” exported — ${pack.exported} samples`, "ok");
    } catch (err) {
      result.textContent = err.message;
    } finally {
      $("#pack-export").disabled = false;
    }
  });
}

export function initLibrary() {
  for (const b of document.querySelectorAll("#lib-filter button")) {
    b.addEventListener("click", () => {
      state.filter = b.dataset.filter;
      for (const o of document.querySelectorAll("#lib-filter button")) o.classList.toggle("active", o === b);
      loadLibrary();
    });
  }
  $("#lib-search").addEventListener("input", () => {
    state.similar = null; // typing starts a new search
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadLibrary, 250);
  });
  $("#lib-tag").addEventListener("change", loadLibrary);
  $("#lib-by-sound").addEventListener("click", () => setBySound(!state.bySound));
  $("#lib-similar-clear").addEventListener("click", () => { state.similar = null; loadLibrary(); });
  bindPackDialog();
}

function setBySound(on) {
  state.bySound = on;
  const button = $("#lib-by-sound");
  button.classList.toggle("active", on);
  button.setAttribute("aria-pressed", String(on));
  $("#lib-search").placeholder = on ? "Describe how it sounds: metallic scrape, warm pad, distant thunder…" : "Search names, prompts, tags…";
  pollIndex();
  if ($("#lib-search").value.trim()) loadLibrary();
}

// "Listening to your library… 120 / 800" while the fingerprints are being made.
async function pollIndex() {
  clearTimeout(indexTimer);
  const hint = $("#lib-index-status");
  if (!(state.bySound || state.similar) || $("#view-library").hidden) {
    hint.hidden = true;
    return;
  }
  try {
    const s = await api.searchStatus();
    hint.hidden = false;
    if (!s.available) {
      hint.textContent = "Search by sound needs the CLAP model: double-click tools\\install-search.bat (or install style training), then restart samplegen.";
      return;
    }
    if (s.error) hint.textContent = `Search by sound is paused: ${s.error}`;
    else if (s.indexed < s.total) hint.textContent = `Listening to your library… ${s.indexed} / ${s.total} sounds (newest first). Results get better as it goes.`;
    else { hint.hidden = true; return; }
  } catch {
    hint.hidden = true;
  }
  indexTimer = setTimeout(pollIndex, 4000);
}

// "Sounds like this" on any sample row: the next Library load lists its closest matches.
export function setSimilar(record) {
  state.similar = { id: record.id, name: record.name };
  $("#lib-search").value = "";
}

export function showLibrary() {
  refreshTags();
  pollIndex();
  return loadLibrary();
}
