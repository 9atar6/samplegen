// Standard MIDI Files, read and written in the browser.
// A "take" is a list of notes: { time, dur } in seconds, { midi } 0-127, { vel } 0-1.

const PPQ = 480;

// ---------- write ----------

function varLen(value) {
  const bytes = [value & 0x7f];
  while ((value >>= 7)) bytes.unshift((value & 0x7f) | 0x80);
  return bytes;
}

// Type 0 file with a tempo, so DAWs line the notes up on their grid at `bpm`.
export function writeMidi(notes, bpm = 120) {
  const ticksPerSecond = (PPQ * bpm) / 60;
  const events = [];
  for (const n of notes) {
    const on = Math.round(n.time * ticksPerSecond);
    const off = Math.max(on + 1, Math.round((n.time + n.dur) * ticksPerSecond));
    const vel = Math.max(1, Math.min(127, Math.round(n.vel * 127)));
    events.push({ tick: on, bytes: [0x90, n.midi, vel] }, { tick: off, bytes: [0x80, n.midi, 0] });
  }
  events.sort((a, b) => a.tick - b.tick || a.bytes[0] - b.bytes[0]); // note-offs before ons at the same tick
  const usPerBeat = Math.round(60_000_000 / bpm);
  const track = [0, 0xff, 0x51, 0x03, (usPerBeat >> 16) & 0xff, (usPerBeat >> 8) & 0xff, usPerBeat & 0xff];
  let last = 0;
  for (const e of events) {
    track.push(...varLen(e.tick - last), ...e.bytes);
    last = e.tick;
  }
  track.push(0, 0xff, 0x2f, 0x00);
  const header = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, PPQ >> 8, PPQ & 0xff];
  const len = track.length;
  return new Uint8Array([...header, 0x4d, 0x54, 0x72, 0x6b, (len >>> 24) & 0xff, (len >> 16) & 0xff,
    (len >> 8) & 0xff, len & 0xff, ...track]);
}

// ---------- read ----------

// Notes from every track, timed through the file's tempo changes. Returns { notes, bpm }.
export function parseMidi(buffer) {
  const data = new DataView(buffer);
  const text = (at) => String.fromCharCode(data.getUint8(at), data.getUint8(at + 1), data.getUint8(at + 2), data.getUint8(at + 3));
  if (buffer.byteLength < 14 || text(0) !== "MThd") throw new Error("That isn't a MIDI file.");
  const tracks = data.getUint16(10);
  const division = data.getUint16(12);
  if (division & 0x8000) throw new Error("SMPTE-timed MIDI files aren't supported.");

  const raw = []; // { tick, kind: "on"|"off"|"tempo", midi, vel, usPerBeat }
  let pos = 8 + data.getUint32(4);
  for (let t = 0; t < tracks && pos + 8 <= buffer.byteLength; t++) {
    const end = pos + 8 + data.getUint32(pos + 4);
    if (text(pos) !== "MTrk") { pos = end; continue; }
    let p = pos + 8;
    let tick = 0;
    let status = 0;
    const readVar = () => {
      let value = 0;
      let byte;
      do { byte = data.getUint8(p++); value = (value << 7) | (byte & 0x7f); } while (byte & 0x80);
      return value;
    };
    while (p < end) {
      tick += readVar();
      let byte = data.getUint8(p);
      if (byte & 0x80) { status = byte; p++; } // otherwise: running status
      const type = status & 0xf0;
      if (status === 0xff) {
        const meta = data.getUint8(p++);
        const len = readVar();
        if (meta === 0x51 && len === 3) {
          raw.push({ tick, kind: "tempo", usPerBeat: (data.getUint8(p) << 16) | (data.getUint8(p + 1) << 8) | data.getUint8(p + 2) });
        }
        p += len;
      } else if (status === 0xf0 || status === 0xf7) {
        p += readVar();
      } else if (type === 0x90 || type === 0x80) {
        const midi = data.getUint8(p);
        const vel = data.getUint8(p + 1);
        p += 2;
        raw.push({ tick, kind: type === 0x90 && vel > 0 ? "on" : "off", midi, vel: vel / 127 });
      } else {
        p += type === 0xc0 || type === 0xd0 ? 1 : 2;
      }
    }
    pos = end;
  }

  raw.sort((a, b) => a.tick - b.tick || (a.kind === "tempo" ? -1 : 0));
  let usPerBeat = 500000;
  let firstTempo = null;
  let lastTick = 0;
  let seconds = 0;
  const open = new Map();
  const notes = [];
  for (const e of raw) {
    seconds += ((e.tick - lastTick) * usPerBeat) / division / 1e6;
    lastTick = e.tick;
    if (e.kind === "tempo") {
      usPerBeat = e.usPerBeat;
      firstTempo ??= e.usPerBeat;
    } else if (e.kind === "on") {
      if (!open.has(e.midi)) open.set(e.midi, []);
      open.get(e.midi).push({ time: seconds, vel: e.vel });
    } else {
      const started = open.get(e.midi)?.shift();
      if (started) notes.push({ time: started.time, dur: Math.max(0.02, seconds - started.time), midi: e.midi, vel: started.vel });
    }
  }
  notes.sort((a, b) => a.time - b.time);
  return { notes, bpm: firstTempo ? Math.round(60_000_000 / firstTempo) : 120 };
}
