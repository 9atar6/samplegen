// Voice to sound: record an imitation from the mic and hand it back as a WAV file,
// ready to be the source of a Transform ("pshhh" -> a real steam burst).

const MAX_SECONDS = 30;
const EDGE_THRESHOLD = 0.02; // relative to the take's peak: trims the click of pressing the button
const EDGE_KEEP_S = 0.03;

export function micSupported() {
  return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
}

// Starts recording at once. Returns { stop } where stop() resolves to a WAV File.
export async function startRecording({ onTick, onLimit } = {}) {
  // Processing off: noise suppression and auto gain would smear exactly the transients we want.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const started = performance.now();
  const timer = setInterval(() => {
    const seconds = (performance.now() - started) / 1000;
    onTick?.(seconds);
    if (seconds >= MAX_SECONDS) onLimit?.();
  }, 100);
  recorder.start();

  let stopping = null;
  const stop = () => {
    stopping ??= new Promise((resolve, reject) => {
      clearInterval(timer);
      recorder.onstop = async () => {
        for (const track of stream.getTracks()) track.stop();
        try {
          resolve(await toWavFile(new Blob(chunks, { type: recorder.mimeType })));
        } catch (err) {
          reject(err);
        }
      };
      recorder.stop();
    });
    return stopping;
  };
  return { stop };
}

async function toWavFile(blob) {
  const ctx = new AudioContext();
  try {
    const buffer = await ctx.decodeAudioData(await blob.arrayBuffer());
    const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
    const [from, to] = soundBounds(channels, buffer.sampleRate);
    if (to - from < buffer.sampleRate * 0.1) throw new Error("Nothing was recorded — check the microphone and try again.");
    const trimmed = channels.map((data) => data.subarray(from, to));
    const stamp = new Date().toTimeString().slice(0, 5).replace(":", "h");
    return new File([encodeWav(trimmed, buffer.sampleRate)], `voice ${stamp}.wav`, { type: "audio/wav" });
  } finally {
    ctx.close();
  }
}

// First/last sample above a small fraction of the peak, with a little air kept around them.
function soundBounds(channels, sampleRate) {
  const frames = channels[0].length;
  let peak = 0;
  for (const data of channels) for (let i = 0; i < frames; i++) peak = Math.max(peak, Math.abs(data[i]));
  if (peak === 0) return [0, 0];
  const loud = (i) => channels.some((data) => Math.abs(data[i]) > peak * EDGE_THRESHOLD);
  let first = 0;
  while (first < frames && !loud(first)) first++;
  let last = frames - 1;
  while (last > first && !loud(last)) last--;
  const keep = Math.round(EDGE_KEEP_S * sampleRate);
  return [Math.max(0, first - keep), Math.min(frames, last + 1 + keep)];
}

// 16-bit PCM WAV: universally readable, and plenty for a voice sketch.
function encodeWav(channels, sampleRate) {
  const count = channels.length;
  const frames = channels[0].length;
  const bytes = frames * count * 2;
  const view = new DataView(new ArrayBuffer(44 + bytes));
  const text = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  text(0, "RIFF");
  view.setUint32(4, 36 + bytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, count, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * count * 2, true);
  view.setUint16(32, count * 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, bytes, true);
  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (const data of channels) {
      const v = Math.max(-1, Math.min(1, data[i]));
      view.setInt16(offset, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      offset += 2;
    }
  }
  return view.buffer;
}
