"""samplegen ComfyUI nodes.

- SamplegenSaveWav: ComfyUI's built-in audio savers only write FLAC/MP3/Opus.
  This writes the raw model output as 32-bit float WAV so nothing is quantised
  before the samplegen app post-processes it.
- SamplegenLoadWav: loads a float WAV that the app staged in input/samplegen/.
  (The stock LoadAudio node only accepts files listed in the top input folder.)
- SamplegenInpaintConditioning: Stable Audio 3 inpainting. Attaches the source
  latent plus a regenerate-mask to the conditioning and returns an empty latent
  to sample into.
"""

import os
import struct

import numpy as np
import torch

import folder_paths
import node_helpers

WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
IO_SUBDIR = "samplegen"
MAX_REGIONS = 8


def write_float_wav(path, samples, sample_rate):
    """Write a (frames, channels) float32 array as an IEEE-float WAV file."""
    data = np.ascontiguousarray(samples, dtype="<f4")
    frames, channels = data.shape
    block_align = channels * 4
    data_bytes = data.tobytes()

    fmt_chunk = struct.pack(
        "<4sIHHIIHHH",
        b"fmt ", 18, WAVE_FORMAT_IEEE_FLOAT, channels, sample_rate,
        sample_rate * block_align, block_align, 32, 0,
    )
    fact_chunk = struct.pack("<4sII", b"fact", 4, frames)
    data_header = struct.pack("<4sI", b"data", len(data_bytes))
    riff_size = 4 + len(fmt_chunk) + len(fact_chunk) + len(data_header) + len(data_bytes)

    tmp_path = path + ".part"
    with open(tmp_path, "wb") as f:
        f.write(struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE"))
        f.write(fmt_chunk)
        f.write(fact_chunk)
        f.write(data_header)
        f.write(data_bytes)
    os.replace(tmp_path, path)


def read_float_wav(path):
    """Read a 32-bit float WAV -> (channels, frames) float32 array, sample rate."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a WAV file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(blob):
        chunk_id, size = struct.unpack("<4sI", blob[pos:pos + 8])
        body = blob[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path} is missing fmt or data chunk")
    tag, channels, sample_rate, _, _, bits = fmt
    if tag not in (WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_EXTENSIBLE) or bits != 32:
        raise ValueError(f"{path} must be 32-bit float WAV (got format {tag}, {bits} bits)")
    samples = np.frombuffer(data, dtype="<f4").reshape(-1, channels).T.copy()
    return samples, sample_rate


def parse_regions(text):
    """'4.0-8.0,12-14.5' -> [(4.0, 8.0), (12.0, 14.5)]"""
    regions = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        start, end = (float(v) for v in part.split("-", 1))
        if end <= start or start < 0:
            raise ValueError(f"bad region '{part}'")
        regions.append((start, end))
    if not regions or len(regions) > MAX_REGIONS:
        raise ValueError(f"need 1 to {MAX_REGIONS} regions, got '{text}'")
    return regions


class SamplegenSaveWav:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "job_id": ("STRING", {"default": "job"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "samplegen"

    def save(self, audio, job_id):
        if audio is None:  # e.g. the guitar/piano outputs of a 4-stem separator
            return {"ui": {"samplegen_wavs": []}}
        safe_job = "".join(c for c in job_id if c.isalnum() or c in "-_") or "job"
        out_dir = os.path.join(folder_paths.get_output_directory(), IO_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)

        waveform = audio["waveform"].detach().float().cpu().numpy()  # (batch, channels, frames)
        sample_rate = int(audio["sample_rate"])

        results = []
        for index, item in enumerate(waveform):
            filename = f"{safe_job}_{index:02d}.wav"
            write_float_wav(os.path.join(out_dir, filename), item.T, sample_rate)
            results.append({"filename": filename, "subfolder": IO_SUBDIR, "type": "output"})

        return {"ui": {"samplegen_wavs": results}}


class SamplegenLoadWav:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"
    CATEGORY = "samplegen"

    def load(self, filename):
        name = os.path.basename(filename)  # never leave input/samplegen/
        if not name or name != filename or name in (".", ".."):
            raise ValueError(f"invalid source file name: {filename!r}")
        path = os.path.join(folder_paths.get_input_directory(), IO_SUBDIR, name)
        samples, sample_rate = read_float_wav(path)
        waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, channels, frames)
        return ({"waveform": waveform, "sample_rate": sample_rate},)


class SamplegenInpaintConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "samples": ("LATENT",),
            "source_seconds": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
            "regions": ("STRING", {"default": "0-1"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "apply"
    CATEGORY = "samplegen"

    def apply(self, positive, negative, samples, source_seconds, regions):
        latent = samples["samples"]
        frames = latent.shape[-1]
        frames_per_second = frames / source_seconds
        mask = torch.zeros((1, 1, frames), dtype=latent.dtype, device=latent.device)
        for start, end in parse_regions(regions):
            first = max(0, int(np.floor(start * frames_per_second)))
            last = min(frames, int(np.ceil(end * frames_per_second)))
            mask[..., first:last] = 1.0  # 1 = regenerate
        values = {"concat_latent_image": latent, "concat_mask": mask}
        out_latent = {"samples": torch.zeros_like(latent)}
        return (
            node_helpers.conditioning_set_values(positive, values),
            node_helpers.conditioning_set_values(negative, values),
            out_latent,
        )


NODE_CLASS_MAPPINGS = {
    "SamplegenSaveWav": SamplegenSaveWav,
    "SamplegenLoadWav": SamplegenLoadWav,
    "SamplegenInpaintConditioning": SamplegenInpaintConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplegenSaveWav": "Samplegen Save WAV (32-bit float)",
    "SamplegenLoadWav": "Samplegen Load WAV (staged source)",
    "SamplegenInpaintConditioning": "Samplegen Inpaint Conditioning (Stable Audio 3)",
}
