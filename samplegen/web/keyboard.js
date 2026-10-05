// Everything that plays notes: an on-screen piano, the computer keyboard, and any MIDI
// keyboard plugged in (Web MIDI, Chrome/Edge). All three call the same handlers.

const BLACK = new Set([1, 3, 6, 8, 10]);
// Two rows like a piano: A W S E D F T G Y H U J K O L P = C ... E (one octave and a bit).
const COMPUTER_KEYS = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14, p: 15 };
const COMPUTER_VELOCITY = 0.75;

// ---------- on-screen piano ----------

export function buildPiano(container, low, high, { onDown, onUp }) {
  container.replaceChildren();
  const keys = new Map();
  const whites = [];
  for (let midi = low; midi <= high; midi++) if (!BLACK.has(midi % 12)) whites.push(midi);
  const width = 100 / whites.length;
  let whiteIndex = 0;
  for (let midi = low; midi <= high; midi++) {
    const key = document.createElement("button");
    key.type = "button";
    key.tabIndex = -1;
    key.dataset.midi = String(midi);
    const black = BLACK.has(midi % 12);
    key.className = black ? "pkey black" : "pkey white";
    if (black) {
      key.style.left = `calc(${whiteIndex * width}% - ${width * 0.3}%)`;
      key.style.width = `${width * 0.6}%`;
    } else {
      key.style.left = `${whiteIndex * width}%`;
      key.style.width = `${width}%`;
      if (midi % 12 === 0) key.dataset.label = `C${midi / 12 - 1}`;
      whiteIndex++;
    }
    keys.set(midi, key);
    container.append(key);
  }
  // Pointer: press, glide across keys while held, release anywhere.
  let held = null;
  const press = (midi, e) => {
    if (held === midi) return;
    if (held !== null) onUp(held);
    held = midi;
    const rect = keys.get(midi).getBoundingClientRect();
    const depth = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)); // lower on the key = louder
    onDown(midi, 0.45 + 0.55 * depth);
  };
  container.onpointerdown = (e) => {
    const key = e.target.closest(".pkey");
    if (!key) return;
    container.setPointerCapture(e.pointerId);
    press(Number(key.dataset.midi), e);
  };
  container.onpointermove = (e) => {
    if (held === null) return;
    const key = document.elementFromPoint(e.clientX, e.clientY)?.closest?.(".pkey");
    if (key && container.contains(key)) press(Number(key.dataset.midi), e);
  };
  const release = () => {
    if (held !== null) onUp(held);
    held = null;
  };
  container.onpointerup = release;
  container.onpointercancel = release;
  return {
    light(midi, on) { keys.get(midi)?.classList.toggle("down", on); },
    has(midi) { return keys.has(midi); },
  };
}

// ---------- computer keyboard ----------

export function bindComputerKeys({ isActive, onDown, onUp, onOctave }) {
  let base = 60; // C4
  const down = new Map(); // key -> midi (so a note always ends even if the octave changed meanwhile)
  document.addEventListener("keydown", (e) => {
    if (!isActive() || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.matches?.("input, textarea, select, [contenteditable]")) return;
    const k = e.key.toLowerCase();
    if (k === "z" || k === "x") {
      base = Math.max(24, Math.min(96, base + (k === "z" ? -12 : 12)));
      onOctave?.(base);
      e.preventDefault();
      return;
    }
    if (!(k in COMPUTER_KEYS) || down.has(k)) return;
    e.preventDefault();
    e.stopPropagation();
    const midi = base + COMPUTER_KEYS[k];
    down.set(k, midi);
    onDown(midi, COMPUTER_VELOCITY);
  }, true);
  document.addEventListener("keyup", (e) => {
    const k = e.key.toLowerCase();
    const midi = down.get(k);
    if (midi === undefined) return;
    down.delete(k);
    onUp(midi);
  }, true);
  window.addEventListener("blur", () => { for (const midi of down.values()) onUp(midi); down.clear(); });
  return { get base() { return base; } };
}

// ---------- MIDI keyboards (Web MIDI) ----------

export function midiSupported() {
  return typeof navigator.requestMIDIAccess === "function";
}

// Listens to every connected MIDI input. Resolves to { inputs: [names] } or throws (denied).
export async function connectMidi({ onDown, onUp, onSustain, onDevices }) {
  const access = await navigator.requestMIDIAccess({ sysex: false });
  const attach = () => {
    const names = [];
    for (const input of access.inputs.values()) {
      names.push(input.name);
      input.onmidimessage = ({ data }) => {
        const [status, a, b] = data;
        const type = status & 0xf0;
        if (type === 0x90 && b > 0) onDown(a, b / 127);
        else if (type === 0x80 || (type === 0x90 && b === 0)) onUp(a);
        else if (type === 0xb0 && a === 64) onSustain(b >= 64); // sustain pedal
        else if (type === 0xb0 && (a === 123 || a === 120)) onSustain(false); // all notes off
      };
    }
    onDevices?.(names);
  };
  access.onstatechange = attach; // keyboards plugged in or out while the app is open
  attach();
  return access;
}
