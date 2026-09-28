"""One generation request -> engine run -> post-processed samples in the library."""

import math
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import (
    PostOptions, apply_fades, normalize_peak, postprocess, read_wav, remove_dc, splice_regions, write_wav,
)
from .catalog import MODELS, SAMPLE_RATE
from .comfy import ComfyClient, EngineError
from .generation_request import MAX_SEED, GenerationRequest, sample_name, wav_comment
from .instruments import (
    INSTRUMENT_DIR, InstrumentRequest, chunk_notes, chunk_seconds, keybed_prompt, normalize_together,
    preview_run, slice_chunk, write_instrument,
)
from .library import Library, NewSample, SampleRecord, slugify
from .packs import clean_pack_name, free_folder
from .sources import NATIVE, SourceStore
from .styles import Style, StyleStore
from .stems import DEMUCS_NODE, NOT_INSTALLED, SILENT_PEAK, StemRequest, build_stems_graph, stem_from_filename
from .timing import GENERATION_PAD_SECONDS, plan_generation
from .workflows import build_audio_to_audio, build_inpaint, build_text_to_audio

GENERATION_TIMEOUT_SECONDS = 900
EDIT_CROSSFADE_MS = 20.0
INPUT_SUBDIR = "samplegen"

__all__ = ["GenerationRequest", "Generator", "sample_name"]

Finisher = Callable[[np.ndarray, int], np.ndarray]


def read_engine_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read what the engine wrote; refuse broken output (NaN/inf) instead of saving a silent file."""
    audio, sr = read_wav(path)
    if not np.isfinite(audio).all():
        path.unlink(missing_ok=True)
        raise EngineError("The engine produced broken audio (NaN). Try again, or with another seed.")
    return audio, sr


@dataclass(frozen=True)
class PreparedJob:
    graph: dict
    finish: Finisher
    params: dict
    staged_input: Path | None = None


class Generator:
    def __init__(self, engine, client: ComfyClient, library: Library, comfy_output_dir: Path,
                 sources: SourceStore | None = None, comfy_input_dir: Path | None = None,
                 timeout: float = GENERATION_TIMEOUT_SECONDS, styles: StyleStore | None = None):
        self.engine = engine
        self.client = client
        self.library = library
        self.comfy_output_dir = Path(comfy_output_dir)
        self.comfy_input_dir = Path(comfy_input_dir) if comfy_input_dir else self.comfy_output_dir.parent / "input"
        self.sources = sources or SourceStore(library.root)
        self.styles = styles or StyleStore(self.comfy_output_dir.parent / "models" / "loras")
        self.staging_dir = library.root / "_staging"
        self.timeout = timeout

    def check(self, request: GenerationRequest) -> None:
        """Validate, including the source's length for Transform / Edit and the chosen style."""
        request.validate(self._source_seconds(request))
        if getattr(request, "style", None):
            self._style(request.style)

    def _style(self, slug: str) -> Style:
        try:
            return self.styles.get(slug)
        except KeyError:
            raise ValueError("That style no longer exists; pick another one.") from None

    def _source_seconds(self, request: GenerationRequest) -> float | None:
        if not request.uses_source or not request.source_id:
            return None
        try:
            return self.sources.get(request.source_id).duration
        except KeyError:
            raise ValueError("That source sound no longer exists; load it again.") from None

    def _raw_output_path(self, output) -> Path:
        root = self.comfy_output_dir.resolve()
        path = (root / output.subfolder / output.filename).resolve()
        if not path.is_relative_to(root):  # we delete this file later; stay inside
            raise ValueError(f"engine returned a file outside its output folder: {output.filename}")
        return path

    def run(self, request, job_id: str) -> list[SampleRecord]:
        if isinstance(request, StemRequest):
            return self._run_stems(request, job_id)
        if isinstance(request, InstrumentRequest):
            return self._run_instrument(request, job_id)
        self.check(request)
        seed = request.seed if request.seed is not None else random.randint(0, MAX_SEED)
        prepared = self._prepare(request, seed, job_id)
        try:
            self.engine.ensure_running()
            outputs = self.client.wait(self.client.submit(prepared.graph), timeout=self.timeout)
        finally:
            if prepared.staged_input:
                prepared.staged_input.unlink(missing_ok=True)

        batch_id = uuid.uuid4().hex[:8]
        source_name = self.sources.get(request.source_id).name if request.uses_source else None
        name = sample_name(request, source_name)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for variation, output in enumerate(outputs, start=1):
            raw_path = self._raw_output_path(output)
            records.append(self._finish(request, prepared, raw_path, name, seed, variation, batch_id, job_id))
        return records

    # ---------- instruments ----------

    def _run_instrument(self, request: InstrumentRequest, job_id: str) -> list[SampleRecord]:
        self.check(request)
        spec = MODELS[request.model]
        seed = request.seed if request.seed is not None else random.randint(0, MAX_SEED)
        low, high = request.midi_range()
        self.engine.ensure_running()

        # Queue every pass up front (same seed for timbral consistency), then collect in order.
        queued = []
        for index, notes in enumerate(chunk_notes(low, high)):
            plan = plan_generation(chunk_seconds(len(notes)), SAMPLE_RATE)
            prompt = keybed_prompt(request.prompt, request.space, notes)
            graph = build_text_to_audio(spec, prompt, "", plan, seed, 1, f"{job_id}c{index:02d}")
            queued.append((notes, self.client.submit(graph)))

        notes_audio, sample_rate = {}, SAMPLE_RATE
        for notes, prompt_id in queued:
            outputs = self.client.wait(prompt_id, timeout=self.timeout)
            if not outputs:
                raise EngineError("The engine returned no audio for part of the keyboard.")
            raw_path = self._raw_output_path(outputs[0])
            raw, sample_rate = read_engine_wav(raw_path)
            raw_path.unlink(missing_ok=True)
            notes_audio.update(slice_chunk(remove_dc(raw), notes, sample_rate))
        notes_audio = normalize_together(notes_audio)

        title = re.sub(r"\s+", " ", request.name or request.prompt).strip()[:80]
        folder = free_folder(self.library.root / INSTRUMENT_DIR, clean_pack_name(title))
        write_instrument(folder, title, slugify(title) or "instrument", notes_audio, sample_rate, request.export)

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        staged = self.staging_dir / f"{job_id}_preview.wav"
        preview = preview_run(notes_audio, sample_rate)
        write_wav(staged, preview, sample_rate, request.export, title=title,
                  comment=f"samplegen | instrument preview | {request.full_prompt()}"[:1000])
        params = {"instrument_folder": folder.relative_to(self.library.root).as_posix(),
                  "low_note": request.low_note, "high_note": request.high_note, "notes": len(notes_audio),
                  "space": request.space, "duration": len(preview) / sample_rate}
        return [self.library.add(staged, NewSample(
            mode="instrument", model=request.model, prompt=request.prompt.strip(), negative_prompt="", seed=seed,
            params=params, duration=len(preview) / sample_rate, sample_rate=request.export.sample_rate,
            batch_id=uuid.uuid4().hex[:8], name=title,
        ))]

    # ---------- stems ----------

    def _run_stems(self, request: StemRequest, job_id: str) -> list[SampleRecord]:
        self.check(request)
        self.engine.ensure_running()
        node_info = self.client.object_info(DEMUCS_NODE)
        if not node_info:
            raise EngineError(NOT_INSTALLED)
        source = self.sources.get(request.source_id)
        staged = self._stage_input(self.sources.load(request.source_id), job_id)
        try:
            graph, separator_model = build_stems_graph(node_info, staged.name, job_id)
            outputs = self.client.wait(self.client.submit(graph), timeout=self.timeout)
        finally:
            staged.unlink(missing_ok=True)

        batch_id = uuid.uuid4().hex[:8]
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for output in outputs:
            raw_path = self._raw_output_path(output)
            stem = stem_from_filename(output.filename, job_id)
            audio, sr = read_wav(raw_path)
            raw_path.unlink(missing_ok=True)
            if audio.size == 0 or np.abs(audio).max() < SILENT_PEAK:
                continue  # nothing in this stem (common for sound effects)
            name = f"{source.name} ({stem})"
            staged_out = self.staging_dir / f"{job_id}_{stem}.wav"
            # No normalizing: stems keep their relative levels so they still sum to the source.
            write_wav(staged_out, audio, sr, request.export, title=name, comment=f"samplegen | stem {stem} of {source.name}")
            params = {"stem": stem, "source_id": source.id, "separator_model": separator_model,
                      "duration": len(audio) / sr, "bpm": source.bpm, "bars": source.bars, "key": source.key}
            records.append(self.library.add(staged_out, NewSample(
                mode="stems", model="demucs", prompt=source.name, negative_prompt="", seed=0, params=params,
                duration=len(audio) / sr, sample_rate=request.export.sample_rate, batch_id=batch_id, name=name,
            )))
        return records

    # ---------- job preparation per mode ----------

    def _prepare(self, request: GenerationRequest, seed: int, job_id: str) -> PreparedJob:
        spec = MODELS[request.model]
        prompt, negative = request.full_prompt(), request.negative_prompt.strip()
        if request.mode == "transform":
            return self._prepare_transform(request, spec, prompt, negative, seed, job_id)
        if request.mode == "edit":
            return self._prepare_edit(request, spec, prompt, negative, seed, job_id)

        lora, params = None, {}
        if request.style:
            # Styles are trained on a base model, so they must run on that base model.
            style = self._style(request.style)
            spec = MODELS[style.model_key]
            lora = (style.lora_file, request.style_strength)
            params = {"style": style.slug, "style_name": style.name, "style_strength": request.style_strength}
        plan = plan_generation(request.target_seconds(), SAMPLE_RATE)
        graph = build_text_to_audio(spec, prompt, negative, plan, seed, request.variations, job_id, lora=lora)
        options = PostOptions(
            target_samples=plan.target_samples, is_loop=request.is_loop, normalize_db=request.normalize_db,
            trim_silence=request.trim_silence and not request.is_loop,
            fade_in_ms=request.fade_in_ms, fade_out_ms=request.fade_out_ms,
        )
        return PreparedJob(graph, lambda raw, sr: postprocess(raw, sr, options), params)

    def _prepare_transform(self, request, spec, prompt, negative, seed, job_id) -> PreparedJob:
        source = self.sources.load(request.source_id)
        if request.source_is_loop:
            # Let the model hear the loop wrap around so the result can loop seamlessly too.
            pad = int(GENERATION_PAD_SECONDS * SAMPLE_RATE)
            model_input = np.concatenate([source, source[:pad]])
        else:
            model_input = source
        staged = self._stage_input(model_input, job_id)
        graph = build_audio_to_audio(spec, prompt, negative, staged.name, math.ceil(len(model_input) / SAMPLE_RATE),
                                     request.strength, seed, request.variations, job_id)
        options = PostOptions(
            target_samples=len(source), is_loop=request.source_is_loop, normalize_db=request.normalize_db,
            trim_silence=False, fade_in_ms=request.fade_in_ms, fade_out_ms=request.fade_out_ms,
        )
        params = {"source_id": request.source_id, "strength": request.strength, "loop": request.source_is_loop}
        return PreparedJob(graph, lambda raw, sr: postprocess(raw, sr, options), params, staged)

    def _prepare_edit(self, request, spec, prompt, negative, seed, job_id) -> PreparedJob:
        source = self.sources.load(request.source_id)
        source_seconds = len(source) / SAMPLE_RATE
        if request.edit_op == "extend":
            extra = np.zeros((int(request.extend_seconds * SAMPLE_RATE), source.shape[1]))
            model_input = np.concatenate([source, extra])
            # Start regenerating a crossfade-length before the end, so the splice
            # happens where both the original and the continuation exist. (The
            # last ~20 ms of the original are handed over to the continuation;
            # there is nothing after the end to crossfade with.)
            regions = [(max(0.0, source_seconds - EDIT_CROSSFADE_MS / 1000 * 1.5), len(model_input) / SAMPLE_RATE)]
        else:
            model_input = source
            regions = [tuple(r) for r in request.regions]
        staged = self._stage_input(model_input, job_id)
        graph = build_inpaint(spec, prompt, negative, staged.name, len(model_input) / SAMPLE_RATE,
                              regions, seed, request.variations, job_id)

        def finish(raw: np.ndarray, sr: int) -> np.ndarray:
            audio = splice_regions(model_input, raw, regions, sr, EDIT_CROSSFADE_MS)
            if request.edit_op == "extend":
                audio = apply_fades(audio, sr, fade_out_ms=request.fade_out_ms)
            return normalize_peak(audio, request.normalize_db) if request.normalize_db is not None else audio

        params = {"source_id": request.source_id, "edit_op": request.edit_op, "regions": regions,
                  "extend_seconds": request.extend_seconds if request.edit_op == "extend" else None}
        return PreparedJob(graph, finish, params, staged)

    def _stage_input(self, audio: np.ndarray, job_id: str) -> Path:
        folder = self.comfy_input_dir / INPUT_SUBDIR
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{job_id}.wav"
        write_wav(path, audio, SAMPLE_RATE, NATIVE)
        return path

    # ---------- results ----------

    def _finish(self, request: GenerationRequest, prepared: PreparedJob, raw_path: Path, name: str, seed: int,
                variation: int, batch_id: str, job_id: str) -> SampleRecord:
        raw, sr = read_wav(raw_path)
        audio = prepared.finish(raw, sr)
        staged = self.staging_dir / f"{job_id}_{variation:02d}.wav"
        write_wav(staged, audio, sr, request.export, title=name, comment=wav_comment(request, seed, variation))
        raw_path.unlink(missing_ok=True)  # engine scratch output, now processed
        params = {
            "duration": len(audio) / sr, "variation": variation, "batch_size": request.variations,
            "bpm": request.bpm if request.is_loop else None, "bars": request.bars if request.is_loop else None,
            "key": f"{request.key} {request.scale}" if request.is_loop else None,
            "full_prompt": request.full_prompt(), "normalize_db": request.normalize_db,
            "export": {"sample_rate": request.export.sample_rate, "bit_depth": request.export.bit_depth},
            **prepared.params,
        }
        return self.library.add(staged, NewSample(
            mode=request.mode, model=request.model, prompt=request.prompt.strip(),
            negative_prompt=request.negative_prompt.strip(), seed=seed, params=params,
            duration=len(audio) / sr, sample_rate=request.export.sample_rate, batch_id=batch_id, name=name,
        ))
