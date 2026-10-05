// Library view: status filters, search, tag filter, sample-pack export.

import { api } from "./api.js";
import { toast } from "./toast.js";
import { createSampleRow, removeRow } from "./samples.js";

const $ = (sel) => document.querySelector(sel);
const LIST_LIMIT = 2000;
const state = { filter: "new", shown: [] };
let searchTimer = null;

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
  try {
    records = await api.samples(params);
  } catch (err) {
    if (ticket !== latestLoad) return;
    $("#lib-empty").textContent = err.message;
    $("#lib-empty").hidden = false;
    return;
  }
  if (ticket !== latestLoad) return; // a newer search/filter is on its way
  for (const el of list.querySelectorAll(".sample")) removeRow(el.dataset.id, list);
  const matches = (r) => (filter === "favorites" ? r.favorite && r.status !== "trashed" : r.status === filter)
    && (!tag || (r.tags || []).includes(tag));
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
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadLibrary, 250);
  });
  $("#lib-tag").addEventListener("change", loadLibrary);
  bindPackDialog();
}

export function showLibrary() {
  refreshTags();
  return loadLibrary();
}
