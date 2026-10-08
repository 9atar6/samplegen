// Plays a generated instrument like a sampler: the nearest recorded note, repitched,
// shaped by velocity and a release envelope. Also renders a take offline to a WAV-ready buffer.

const MIN_GAIN = 0.0001;
function velocityGain(vel) {
  return Math.pow(0.12 + 0.88 * Math.max(0, Math.min(1, vel)), 2); // soft notes stay audible
}

export class Sampler {
  constructor(ctx) {
    this.ctx = ctx;
    this.output = ctx.createGain();
    this.output.gain.value = 0.8;
    this.output.connect(ctx.destination);
    this.samples = new Map(); // midi -> AudioBuffer
    this.loops = new Map(); // midi -> { start, end } in seconds (sustain loop, crossfade baked in)
    this.hold = true; // held keys keep sounding through the loop
    this.roots = [];
    this.voices = new Map(); // midi -> voice
    this.sustained = new Set();
    this.sustain = false;
    this.release = 0.6;
    this.instrument = null;
  }

  // `detail` is /api/instruments/{id}: { id, name, notes: [{ midi, url }] }.
  async load(detail, onProgress) {
    this.allOff();
    const samples = new Map();
    const loops = new Map();
    let done = 0;
    await Promise.all(detail.notes.map(async (note) => {
      const res = await fetch(note.url);
      if (!res.ok) throw new Error(`Couldn't load note ${note.midi} (${res.status})`);
      const decoded = await this.ctx.decodeAudioData(await res.arrayBuffer());
      if (note.loop) {
        const { buffer, start, end } = withLoopCrossfade(this.ctx, decoded, note.loop);
        samples.set(note.midi, buffer);
        loops.set(note.midi, { start, end });
      } else {
        samples.set(note.midi, decoded);
      }
      onProgress?.(++done / detail.notes.length);
    }));
    this.samples = samples;
    this.loops = loops;
    this.exact = detail.kind === "kit"; // a drum kit plays its own keys only: no repitched snares
    this.roots = [...samples.keys()].sort((a, b) => a - b);
    this.instrument = detail;
  }

  get ready() {
    return this.roots.length > 0;
  }

  nearest(midi) {
    let best = this.roots[0];
    for (const root of this.roots) if (Math.abs(root - midi) < Math.abs(best - midi)) best = root;
    return best;
  }

  // One note, on any context (live or offline). Returns a voice to release later.
  voice(ctx, destination, midi, vel, when) {
    if (this.exact && !this.samples.has(midi)) return null;
    const root = this.nearest(midi);
    const src = ctx.createBufferSource();
    src.buffer = this.samples.get(root);
    src.playbackRate.value = Math.pow(2, (midi - root) / 12);
    const loop = this.hold && this.loops.get(root);
    if (loop) {
      src.loop = true;
      src.loopStart = loop.start;
      src.loopEnd = loop.end;
    }
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(velocityGain(vel), when);
    src.connect(gain).connect(destination);
    src.start(when);
    return { src, gain };
  }

  // `force`: stop even a drum hit (retrigger, Stop); otherwise kit hits ring out like one-shots.
  stopVoice(voice, when, release = this.release, force = false) {
    if (!voice || (this.exact && !force)) return;
    voice.gain.gain.cancelScheduledValues(when);
    voice.gain.gain.setTargetAtTime(MIN_GAIN, when, Math.max(0.01, release / 4));
    try { voice.src.stop(when + release * 1.5 + 0.05); } catch { /* already stopped */ }
  }

  noteOn(midi, vel = 0.8) {
    if (!this.ready) return;
    if (this.ctx.state === "suspended") this.ctx.resume();
    const old = this.voices.get(midi);
    if (old) this.stopVoice(old, this.ctx.currentTime, 0.03, true); // retrigger: quick fade, no click
    this.sustained.delete(midi);
    this.voices.set(midi, this.voice(this.ctx, this.output, midi, vel, this.ctx.currentTime));
  }

  noteOff(midi) {
    if (this.sustain) {
      this.sustained.add(midi);
      return;
    }
    const voice = this.voices.get(midi);
    if (!voice) return;
    this.stopVoice(voice, this.ctx.currentTime);
    this.voices.delete(midi);
  }

  setSustain(on) {
    this.sustain = on;
    if (on) return;
    for (const midi of this.sustained) this.noteOff(midi);
    this.sustained.clear();
  }

  allOff() {
    for (const voice of this.voices.values()) this.stopVoice(voice, this.ctx.currentTime, 0.05, true);
    this.voices.clear();
    this.sustained.clear();
  }

  // Play a whole take now. Returns { stop, done } (done resolves when it has finished).
  playTake(notes, onNote) {
    const start = this.ctx.currentTime + 0.08;
    const voices = notes.map((n) => {
      const v = this.voice(this.ctx, this.output, n.midi, n.vel, start + n.time);
      this.stopVoice(v, start + n.time + n.dur);
      return v;
    });
    const timers = onNote ? notes.map((n) => setTimeout(() => onNote(n), (n.time + 0.08) * 1000)) : [];
    const length = takeLength(notes, this.release);
    let finish;
    const done = new Promise((resolve) => { finish = resolve; });
    const endTimer = setTimeout(() => finish(), length * 1000 + 100);
    return {
      done,
      stop: () => {
        clearTimeout(endTimer);
        timers.forEach(clearTimeout);
        for (const v of voices) this.stopVoice(v, this.ctx.currentTime, 0.05, true);
        finish();
      },
    };
  }

  // The take as an AudioBuffer (stereo, 44.1 kHz), for saving to the library.
  async render(notes, sampleRate = 44100) {
    const length = takeLength(notes, this.release) + 0.2;
    const offline = new OfflineAudioContext(2, Math.ceil(length * sampleRate), sampleRate);
    const out = offline.createGain();
    out.gain.value = this.output.gain.value;
    out.connect(offline.destination);
    for (const n of notes) {
      const v = this.voice(offline, out, n.midi, n.vel, n.time);
      this.stopVoice(v, n.time + n.dur);
    }
    return offline.startRendering();
  }
}

// A copy of `decoded` whose loop end already crossfades into the audio just before the loop
// start, so jumping from end back to start is seamless. `loop` is in the file's frames.
function withLoopCrossfade(ctx, decoded, loop) {
  const ratio = decoded.sampleRate / loop.rate; // the browser may have resampled the file
  const start = Math.round(loop.start * ratio);
  const end = Math.min(decoded.length, Math.round(loop.end * ratio));
  const fade = Math.min(Math.round(loop.crossfade * ratio), start, end - start);
  const buffer = ctx.createBuffer(decoded.numberOfChannels, decoded.length, decoded.sampleRate);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const src = decoded.getChannelData(c);
    const out = buffer.getChannelData(c);
    out.set(src);
    for (let i = 0; i < fade; i++) {
      const r = (i + 1) / fade; // 0 -> 1 across the last `fade` frames before the loop end
      out[end - fade + i] = src[end - fade + i] * (1 - r) + src[start - fade + i] * r;
    }
  }
  return { buffer, start: start / decoded.sampleRate, end: end / decoded.sampleRate };
}

export function takeLength(notes, release = 0.6) {
  return notes.reduce((end, n) => Math.max(end, n.time + n.dur), 0) + release * 1.5;
}
