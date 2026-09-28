// Waveform drawing: mirrored rounded bars (min/max per bar), phosphor gradient, playhead.

const BAR = 2;   // bar width in CSS px
const GAP = 1;   // gap between bars in CSS px
const peakCache = new Map(); // `${id}:${bars}` -> Float32Array [min,max,...], oldest first
const MAX_PEAK_ENTRIES = 1500; // a few KB each

export function forgetPeaks(id) {
  for (const key of [...peakCache.keys()]) if (key.startsWith(`${id}:`)) peakCache.delete(key);
}

export function computePeaks(buffer, columns) {
  const channels = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) channels.push(buffer.getChannelData(c));
  const frames = buffer.length;
  const step = frames / columns;
  const peaks = new Float32Array(columns * 2);
  for (let col = 0; col < columns; col++) {
    const from = Math.floor(col * step);
    const to = Math.max(from + 1, Math.floor((col + 1) * step));
    let min = 0;
    let max = 0;
    for (const data of channels) {
      for (let i = from; i < to && i < frames; i++) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    peaks[col * 2] = min;
    peaks[col * 2 + 1] = max;
  }
  return peaks;
}

function cssVar(el, name) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

// Returns false (and draws nothing) when `buffer` is null and no peaks are cached for this size.
export function drawWaveform(canvas, id, buffer, progress) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const bar = BAR * dpr;
  const pitch = (BAR + GAP) * dpr;
  const bars = Math.max(1, Math.floor(width / pitch));
  const key = `${id}:${bars}`;
  if (!peakCache.has(key)) {
    if (!buffer) return false;
    peakCache.set(key, computePeaks(buffer, bars));
    if (peakCache.size > MAX_PEAK_ENTRIES) peakCache.delete(peakCache.keys().next().value);
  }
  const peaks = peakCache.get(key);

  // Normalise the drawing so quiet sounds are still readable (display only).
  let loudest = 0;
  for (let i = 0; i < peaks.length; i++) loudest = Math.max(loudest, Math.abs(peaks[i]));
  const scale = loudest > 0 ? 0.92 / Math.max(loudest, 0.05) : 1;

  const g = canvas.getContext("2d");
  g.clearRect(0, 0, width, height);
  const mid = height / 2;
  const played = progress == null ? -1 : progress * width;
  const hot = g.createLinearGradient(0, 0, 0, height);
  hot.addColorStop(0, cssVar(canvas, "--wave-played-2") || "#ffb23f");
  hot.addColorStop(0.5, cssVar(canvas, "--wave-played") || "#ff6a26");
  hot.addColorStop(1, cssVar(canvas, "--wave-played-2") || "#ffb23f");
  const dim = cssVar(canvas, "--wave") || "rgba(255,150,90,.34)";

  for (let b = 0; b < bars; b++) {
    const x = b * pitch;
    const top = mid - Math.max(peaks[b * 2 + 1] * scale, 0.012) * mid;
    const bottom = mid - Math.min(peaks[b * 2] * scale, -0.012) * mid;
    g.fillStyle = x <= played ? hot : dim;
    const h = Math.max(dpr, bottom - top);
    if (g.roundRect) {
      g.beginPath();
      g.roundRect(x, top, bar, h, bar / 2);
      g.fill();
    } else {
      g.fillRect(x, top, bar, h);
    }
  }
  if (progress != null) {
    g.fillStyle = cssVar(canvas, "--playhead") || "#fff";
    g.shadowColor = cssVar(canvas, "--sig") || "#ff5a1f";
    g.shadowBlur = 10 * dpr;
    g.fillRect(Math.min(played, width - dpr), 0, Math.max(1, 1.5 * dpr), height);
    g.shadowBlur = 0;
  }
  return true;
}
