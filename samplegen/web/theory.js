// Music helpers for takes: which key a melody is in, snapping it to the beat,
// and fitting it to a loop of whole bars.

const KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
// Krumhansl-Kessler key profiles: how strongly each scale degree belongs to a key.
const MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
export const LOOP_BPMS = [100, 110, 120, 128, 130, 140, 150]; // what Foundation-1 was trained on
const BEATS_PER_BAR = 4;
const GRID = 4; // sixteenth notes

function correlation(a, b) {
  const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

// { key: "A", scale: "minor", confidence } from how long each pitch class sounds.
export function detectKey(notes) {
  const weight = new Array(12).fill(0);
  for (const n of notes) weight[n.midi % 12] += n.dur * (0.5 + n.vel);
  let best = { key: "C", scale: "major", confidence: 0 };
  for (let tonic = 0; tonic < 12; tonic++) {
    const rotated = weight.map((_, i) => weight[(i + tonic) % 12]);
    for (const [scale, profile] of [["major", MAJOR], ["minor", MINOR]]) {
      const r = correlation(rotated, profile);
      if (r > best.confidence) best = { key: KEYS[tonic], scale, confidence: r };
    }
  }
  return best;
}

export function nearestLoopBpm(bpm) {
  return LOOP_BPMS.reduce((a, b) => (Math.abs(b - bpm) < Math.abs(a - bpm) ? b : a));
}

// Snap starts and lengths to sixteenths at `bpm`; returns new notes (times in seconds).
export function quantize(notes, bpm) {
  const step = 60 / bpm / GRID;
  return notes.map((n) => {
    const time = Math.round(n.time / step) * step;
    const dur = Math.max(step, Math.round(n.dur / step) * step);
    return { ...n, time, dur };
  });
}

// 4 or 8 bars, whichever holds the take (longer takes are cut at 8 bars).
export function loopBars(notes, bpm) {
  const barSeconds = (60 / bpm) * BEATS_PER_BAR;
  const end = notes.reduce((m, n) => Math.max(m, n.time + n.dur), 0);
  return end <= 4 * barSeconds + 1e-6 ? 4 : 8;
}

export function loopSeconds(bpm, bars) {
  return (60 / bpm) * BEATS_PER_BAR * bars;
}

// Fold a rendered take into exactly `frames`: what rings past the end wraps onto the
// start, so the loop is seamless (like a sampler's loop recording).
export function foldIntoLoop(buffer, frames, ctx) {
  const out = ctx.createBuffer(buffer.numberOfChannels, frames, buffer.sampleRate);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const src = buffer.getChannelData(c);
    const dst = out.getChannelData(c);
    for (let i = 0; i < src.length; i++) dst[i % frames] += src[i];
  }
  return out;
}
