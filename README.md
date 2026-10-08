# samplegen

Local AI sample generator for sound design and music. Everything runs on your own PC: no account,
no subscription, no upload.

## Install

**You need:** Windows 10/11, an **NVIDIA GPU with 8 GB+ VRAM**, about **40 GB** free disk, internet for
the first install.

1. Get this folder (GitHub → **Code → Download ZIP**, then right-click the ZIP → **Extract All…**) somewhere
   with space, e.g. `C:\AI\samplegen`.
2. Double-click **`install.bat`** and follow what it says. It checks your disk space and GPU, downloads the
   engine (ComfyUI) and the models (~20 GB, 30–90 min), asks where to save your samples, and offers a desktop
   shortcut. Safe to run again at any time: finished steps are skipped, so it also works as a *repair*.
3. Double-click **`samplegen.bat`** (or the shortcut). The app opens in your browser
   (http://127.0.0.1:8190); the first start takes about a minute. Close the black window to quit.

**Style training** (optional, teach samplegen your own sounds) also needs a free
[Hugging Face](https://huggingface.co) account. `install.bat` offers to set it up; you can also run
`tools\install-training.bat` later. It walks you through accepting the license and logging in.

### If something goes wrong

| What you see | What to do |
|---|---|
| A Windows security warning when opening a `.bat` | Files from a downloaded ZIP get flagged: click **Run** (or **More info → Run anyway**). |
| The installer says **Something went wrong** | Check internet and disk space, run `install.bat` again: it resumes. Details in `logs\install.log`. |
| A red **Install not finished** bar in the app | A model or extension is missing: run `install.bat` again. |
| **The sample library location is not available** | Your library drive is unplugged: plug it in, or delete `samplegen.local.json` and run `install.bat` to pick another folder. |
| The top-right light stays red (**Engine error**) | Hover it for the reason; the full log is `logs\engine.log`. Restarting samplegen usually fixes it. |
| **Out of memory** errors | Shorter sounds, fewer variations, or the Small SFX model. Close games / other GPU apps. |

## What it does

| Mode | Model | Good for |
|---|---|---|
| **SFX** | Stable Audio 3 Small SFX (fast) or Medium (best quality) | impacts, foley, whooshes, UI, ambiences |
| **Loop** | Foundation-1.2 Samples | tempo- and key-locked musical loops (100–150 BPM, 4 or 8 bars) |
| **Free** | Stable Audio 3 Medium | textures, drones, long musical material (up to 6 min) |
| **Transform** | any of the three | turn a sound you have into something new; **Strength** from subtle to total |
| **Edit** | Stable Audio 3 (SFX or Medium) | **Regenerate selection** (drag across the waveform) or **Extend** a sound |
| **Instrument** | Foundation-1.2 Keybeds | a playable sampler instrument across a note range, as Decent Sampler (`.dspreset`) + SFZ |

On any sample: **Split into stems** (`S`, Demucs htdemucs_ft), **rename / tags** (`E`), **use as source** (`T`),
**MIDI** (`M`, Spotify basic-pitch: the notes of a loop, bass line or melody as a `.mid`, at the loop's tempo; drag the
♪ button into your DAW. The `.mid` lives next to the WAV and goes into packs too).

- **Seamless loop** (SFX / Free): ambiences, drones and beds whose end flows back into their start — the model plays
  on past the end and that continuation is crossfaded (2 s, equal-power) into the beginning. Up to ~6 min with Medium.
- **Play** tab: play your generated instruments from a **MIDI keyboard** (Chrome/Edge, velocity + sustain pedal),
  the computer keys (`A`–`K`, `Z`/`X` octave) or the screen. Held notes sustain (each note gets a crossfaded sustain
  loop, also written into the Decent Sampler / SFZ presets). Record takes, **hum a melody** and hear it on the
  instrument, drop a `.mid` (or a sample's ♪), export MIDI, save takes to the library.
- **Shot list** (button under Generate): one sound per line (`door creak x8`, `footsteps on gravel | 12x | 3s`,
  `thunder | 20s | medium | loop`); samplegen queues them all, tags the results with the list's name, keeps the PC
  awake until it's done, and can export everything as a loudness-matched, round-robin pack at the end.
- **More like this** (`V` on any sample): 4 gentle variations of it, same name and tags (they export as round robins).
- **Drum kit** mode: one style prompt → 12 pieces (General MIDI map) × 1–4 round robins, open hat choked by the closed
  hat; Decent Sampler + SFZ in `Kits\<name>\`, playable in the Play tab.
- **Layers** mode: an impact from a **transient**, a **body** and a **tail**, each generated on its own, lined up on
  their first transient, offset and balanced; saves the blended hit and the three layers.
- **Make it a loop** (Play): **Melody → new sound** snaps a take to the beat, fits it to 4/8 bars at a Foundation-1
  tempo, renders it as a seamless loop and opens Transform with it (Foundation-1, loop kept seamless): describe a
  sound and the AI plays your melody with it. **Loops in this key & tempo** detects the take's key
  (Krumhansl-Kessler) and opens Loop mode set to it.
- **Pack export** options: match **loudness** (−14 / −18 / −23 LUFS, ITU-R BS.1770, never past −1 dBFS peak) and name
  takes of one generation as **round robins** (`name_01`, `name_02`… for Wwise / FMOD / samplers); `samples.csv`
  gets a `lufs` column.
- **Search by sound** (Library → *by sound*): type what it sounds like ("metallic scrape", "warm pad"), or press
  `L` on any sample for **sounds like this**. Uses CLAP from style training, or `tools\install-search.bat` alone;
  samplegen fingerprints your library in the background (CPU, newest first).
- **Repair**: at start, samples moved by an interrupted operation are re-linked, finished takes that were never
  saved come back as "Recovered take".
- **Voice to sound** (Transform / Edit): press **Record your voice**, imitate the sound ("pshhh", "brrrm", "tk-tk"),
  then describe what it should become. The rhythm and envelope of your voice drive the result.
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
tools/               install-training.bat, install-search.bat (optional), install-midi.bat (part of install), run-tests.bat
midi/                MIDI extraction script (its own Python in midi/.venv, created by install)
tests/               test suite (engine faked, no GPU needed)
engine/              ComfyUI + models              (created by install.bat, not in git)
trainer/stable-audio-3/  official trainer            (created by install, not in git)
```

## Licenses

samplegen's own code is yours to use and share with friends. The things it downloads keep their own licenses:
Stable Audio 3 and Foundation-1 (Stability AI Community License: free, including commercial use under
$1M annual revenue), ComfyUI (GPL-3.0), AudioSeparation (GPL-3.0), CLAP (Apache-2.0), basic-pitch (Apache-2.0). The fonts in
`samplegen/web/fonts` are under the SIL Open Font License.

## Development

Double-click **`tools\run-tests.bat`** (results also go to `logs\test-results.txt`).
(pytest comes with the `dev` dependency group, which `uv sync` / `install.bat` install by default.)

> Don't run pytest through an AI agent's sandboxed shell on this PC: directory scans inside that
> sandbox trigger a Windows `bindflt.sys` blue screen. Outside the sandbox it's fine.
