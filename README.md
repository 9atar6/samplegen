# samplegen

Local AI sample generator for sound design and music. Everything runs on your own PC: no account,
no subscription, no upload.

## Install

**You need:** Windows 10/11, an **NVIDIA GPU with 8 GB+ VRAM**, about **40 GB** free disk, internet for
the first install.

1. Get this folder (GitHub → **Code → Download ZIP**, then unzip it somewhere with space, e.g. `C:\AI\samplegen`).
2. Double-click **`install.bat`**. It downloads the engine (ComfyUI), the models (~20 GB) and everything else
   into this folder, and asks where to save your samples. It's safe to run again at any time: finished steps
   are skipped, so it also works as a *repair*.
3. Double-click **`samplegen.bat`**. The app opens in your browser (http://127.0.0.1:8190). Close the black
   window to quit.

**Style training** (optional, teach samplegen your own sounds) also needs a free
[Hugging Face](https://huggingface.co) account. `install.bat` offers to set it up; you can also run
`tools\install-training.bat` later.

## What it does

| Mode | Model | Good for |
|---|---|---|
| **SFX** | Stable Audio 3 Small SFX (fast) or Medium (best quality) | impacts, foley, whooshes, UI, ambiences |
| **Loop** | Foundation-1.2 Samples | tempo- and key-locked musical loops (100–150 BPM, 4 or 8 bars) |
| **Free** | Stable Audio 3 Medium | textures, drones, long musical material (up to 6 min) |
| **Transform** | any of the three | turn a sound you have into something new; **Strength** from subtle to total |
| **Edit** | Stable Audio 3 (SFX or Medium) | **Regenerate selection** (drag across the waveform) or **Extend** a sound |
| **Instrument** | Foundation-1.2 Keybeds | a playable sampler instrument across a note range, as Decent Sampler (`.dspreset`) + SFZ |

On any sample: **Split into stems** (`S`, Demucs htdemucs_ft), **rename / tags** (`E`), **use as source** (`T`).
Library: filter by tag, search, and **Export pack** (copies what's listed into `Packs\<name>\`, optionally
converted, with a `samples.csv` of prompts, seeds, tempo, key and tags).

### Styles: train samplegen on your sounds

In **Train**: name the style, pick SFX or Music as the base, load 20–50 sounds that belong together (from a
folder, or library samples with a tag), press **Auto-describe** (or write the descriptions), and start.
When it's done, the style appears in the **Style** menu of the SFX and Free modes, with a strength slider.

A style is a **LoRA**: a small add-on (~20 MB) that nudges an existing model toward your sound. It isn't a
model on its own; it always runs on top of the Stable Audio 3 *base* model it was trained on (50 steps, so
styled generations are slower than stock). Styles live in
`engine\ComfyUI_windows_portable\ComfyUI\models\loras\samplegen_<name>.safetensors` (+ a `.json` description).

**Auto-describe**: CLAP (LAION `larger_clap_general`) names what the sound is; measurements add what's
reliably measurable (attack, decay, pitch sweep, tuning note, saturation, sub weight, stereo width); musical
words from the file name are kept (909, grime, house…).

## Use

- Pick a mode, describe the sound, choose how many variations, **Generate** (`Ctrl+Enter`).
- Audition: `Space` play/stop, `↑`/`↓` move, click the waveform to seek. Loops play looped, gapless.
- `K` keep, `X` trash, `F` favorite. **More like this** re-runs the same settings with a new seed.
- Drag any sample into Explorer or your DAW.
- Top-right button: bone (light) or graphite (dark) finish.

## Where things go

Your library folder (chosen during install, stored in `samplegen.local.json`; override with the
`SAMPLEGEN_LIBRARY` environment variable):

```
Inbox\<date>\          new generations
Kept\<type>\           what you kept (SFX, Loops, Music, Transformed, Edited, Stems, Instruments)
Instruments\<name>\    playable instruments (Samples\, .dspreset, .sfz)
Packs\<name>\          exported sample packs
_sources\              copies of the sounds used for Transform / Edit / stems (originals never change)
_training\             datasets and logs of style trainings
_trash\                what you trashed (never auto-deleted)
samplegen.db           prompt, seed, tags and settings of every sample
```

Every WAV also carries its prompt and seed in its metadata.

## Audio pipeline

The models output 44.1 kHz stereo float. samplegen then removes DC offset; for **loops** cuts to the exact
sample length for the BPM/bars and crossfades the model's own continuation into the loop start (seamless
wrap-around); for **one-shots** cuts to length, optionally trims silence and fades; peak-normalizes (default
−1 dBFS); exports 32-bit float (default), 24-bit or 16-bit (TPDF-dithered) at 44.1/48/96 kHz (soxr VHQ).
In **Edit**, everything outside the selection stays bit-identical to the source.

## Folder layout

```
samplegen.bat        start the app
install.bat          install / repair everything
samplegen/           the app (Python backend + web UI in samplegen/web)
engine_nodes/        samplegen's own ComfyUI nodes (float WAV in/out, Stable Audio 3 inpainting)
trainer/             style-training helpers (LoRA converter, CLAP tagger)
tools/               install-training.bat (optional part of install), run-tests.bat (developers)
tests/               test suite (engine faked, no GPU needed)
engine/              ComfyUI + models              (created by install.bat, not in git)
trainer/stable-audio-3/  official trainer            (created by install, not in git)
```

## Licenses

samplegen's own code is yours to use and share with friends. The things it downloads keep their own licenses:
Stable Audio 3 and Foundation-1 (Stability AI Community License: free, including commercial use under
$1M annual revenue), ComfyUI (GPL-3.0), AudioSeparation (GPL-3.0), CLAP (Apache-2.0). The fonts in
`samplegen/web/fonts` are under the SIL Open Font License.

## Development

Double-click **`tools\run-tests.bat`** (results also go to `logs\test-results.txt`).

> Don't run pytest through an AI agent's sandboxed shell on this PC: directory scans inside that
> sandbox trigger a Windows `bindflt.sys` blue screen. Outside the sandbox it's fine.
