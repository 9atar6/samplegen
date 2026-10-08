// Inline SVG icons (24x24, stroke = currentColor). Use: el.innerHTML = icon("play").

const PATHS = {
  play: '<path d="M7 4.5v15l12.5-7.5z" fill="currentColor" stroke="none"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
  star: '<path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"/>',
  check: '<path d="M4.5 12.5l4.8 4.8L19.5 7"/>',
  trash: '<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 12.5h9l1-12.5M10 11v5M14 11v5"/>',
  restore: '<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4"/>',
  swap: '<path d="M4 8h13l-3.5-3.5M20 16H7l3.5 3.5"/>',
  stems: '<path d="M12 3l9 4.5-9 4.5-9-4.5zM3 12l9 4.5 9-4.5M3 16.5L12 21l9-4.5"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
  spark: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"/>',
  loop: '<path d="M17 3l3 3-3 3M4 11V9a3 3 0 0 1 3-3h13M7 21l-3-3 3-3M20 13v2a3 3 0 0 1-3 3H4"/>',
  tag: '<path d="M3.5 12.5V4.5a1 1 0 0 1 1-1h8l8 8-9 9zM8 8h.01"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>',
  wave: '<path d="M2 12h2M6 8v8M10 4v16M14 7v10M18 10v4M22 12h0"/>',
  keys: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 4v10M12 4v10M16 4v10M8 14v6M16 14v6"/>',
  shuffle: '<path d="M4 7h3.5c5 0 5 10 10 10H20M17 14l3 3-3 3M4 17h3.5c1.4 0 2.4-.8 3.2-1.9M13.3 8.9C14.1 7.8 15.1 7 16.5 7H20M17 4l3 3-3 3"/>',
  scissors: '<circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><path d="M8 7.5L20 18M8 16.5L20 6"/>',
  drone: '<path d="M2 12c2.5-6 4.5-6 7 0s4.5 6 7 0 4.5-6 6 0"/>',
  bolt: '<path d="M13 2.5L4.5 13.5H12l-1 8 8.5-11H12z"/>',
  alert: '<path d="M12 3.5l9.5 16.5h-19zM12 10v4.5M12 17.5h.01"/>',
  tune: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  similar: '<path d="M2 9c2-4 4-4 6 0s4 4 6 0 4-4 6 0M2 16c2-4 4-4 6 0s4 4 6 0 4-4 6 0" opacity=".9"/>',
  midi:'<path d="M9 18V5.5l11-2V16"/><circle cx="6.5" cy="18" r="2.5"/><circle cx="17.5" cy="16" r="2.5"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
  grip:'<path d="M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01" stroke-width="3"/>',
};

export function icon(name, size = 18) {
  const body = PATHS[name] ?? PATHS.spark;
  return `<svg class="ico" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}

export function iconButton(name, title, className = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `icon ${className}`.trim();
  b.title = title;
  b.setAttribute("aria-label", title);
  b.innerHTML = icon(name);
  return b;
}
