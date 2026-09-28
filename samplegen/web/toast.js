// Small transient notifications, bottom-right.

import { icon } from "./icons.js";

const DURATION_MS = 3200;

function host() {
  let el = document.getElementById("toasts");
  if (!el) {
    el = document.createElement("div");
    el.id = "toasts";
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");
    document.body.append(el);
  }
  return el;
}

export function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  const glyph = { ok: "check", error: "alert", info: "spark" }[kind] ?? "spark";
  el.innerHTML = icon(glyph, 16);
  const text = document.createElement("span");
  text.textContent = message;
  el.append(text);
  host().append(el);
  requestAnimationFrame(() => el.classList.add("in"));
  setTimeout(() => {
    el.classList.remove("in");
    el.addEventListener("transitionend", () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 600);
  }, kind === "error" ? DURATION_MS * 2 : DURATION_MS);
}
