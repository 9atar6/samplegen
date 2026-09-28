// Gapless WebAudio player: one sound at a time, loops loop sample-accurately.

import { api } from "./api.js";

const ctx = new (window.AudioContext || window.webkitAudioContext)();
const buffers = new Map(); // sampleId -> Promise<AudioBuffer>
const listeners = new Set();

let current = null; // { id, source, startedAt, offset, loop, duration }

// `id` is a cache key: a sample id, or e.g. "src:<id>" with an explicit url.
export function loadBuffer(id, url = api.audioUrl(id)) {
  if (!buffers.has(id)) {
    const promise = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`audio ${r.status}`);
        return r.arrayBuffer();
      })
      .then((data) => ctx.decodeAudioData(data));
    promise.catch(() => buffers.delete(id)); // allow retry after a failure
    buffers.set(id, promise);
  }
  return buffers.get(id);
}

export function forget(id) {
  buffers.delete(id);
}

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit() {
  for (const fn of listeners) fn(current ? current.id : null);
}

export function stop() {
  if (current) {
    current.source.onended = null;
    current.source.stop();
    current = null;
    emit();
  }
}

export async function play(id, { offset = 0, loop = false, url } = {}) {
  if (ctx.state === "suspended") await ctx.resume();
  const buffer = await loadBuffer(id, url);
  stop();
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = loop;
  source.connect(ctx.destination);
  const start = Math.max(0, Math.min(offset, buffer.duration - 0.001));
  source.start(0, start);
  current = { id, source, startedAt: ctx.currentTime, offset: start, loop, duration: buffer.duration };
  source.onended = () => {
    if (current && current.source === source) {
      current = null;
      emit();
    }
  };
  emit();
}

export async function toggle(id, opts) {
  if (current && current.id === id) stop();
  else await play(id, opts);
}

export function isPlaying(id) {
  return Boolean(current && current.id === id);
}

// 0..1 position of the playing sample, or null.
export function progress(id) {
  if (!current || current.id !== id) return null;
  const elapsed = ctx.currentTime - current.startedAt + current.offset;
  const pos = current.loop ? elapsed % current.duration : Math.min(elapsed, current.duration);
  return pos / current.duration;
}
