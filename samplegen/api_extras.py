"""API routes for library extras (rename, tags, packs), stems and instruments."""

import subprocess
import sys
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .audio import ExportFormat
from .instruments import InstrumentRequest
from .library import SampleNotFound
from .midi import MidiError
from .packs import MAX_PACK_SAMPLES, export_pack
from .stems import StemRequest


class ExportFormatIn(BaseModel):
    sample_rate: int = 44100
    bit_depth: Literal["32f", "24", "16"] = "32f"

    def to_format(self) -> ExportFormat:
        try:
            return ExportFormat(sample_rate=self.sample_rate, bit_depth=self.bit_depth)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


class RenameIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class TagsIn(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=20)


class PackIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    sample_ids: list[str] = Field(..., min_length=1, max_length=MAX_PACK_SAMPLES)
    export: ExportFormatIn | None = None  # None: copy files as they are
    loudness: float | None = None  # LUFS target, None: keep levels
    round_robin: bool = False
    reveal: bool = True


class StemsIn(BaseModel):
    export: ExportFormatIn = ExportFormatIn()


class InstrumentIn(BaseModel):
    prompt: str = Field(..., max_length=500)
    name: str = Field("", max_length=80)
    low_note: str = Field("C3", max_length=4)
    high_note: str = Field("B4", max_length=4)
    space: Literal["Dry", "Wet"] = "Dry"
    seed: int | None = None
    export: ExportFormatIn = ExportFormatIn(bit_depth="24")


def open_in_explorer(path, select: bool = False) -> None:
    if sys.platform == "win32" and path.exists():
        subprocess.Popen(["explorer", f"/select,{path}"] if select else ["explorer", str(path)])


def add_extra_routes(app: FastAPI, ctx) -> None:
    def sample_or_404(sample_id: str):
        try:
            return ctx.library.get(sample_id)
        except SampleNotFound:
            raise HTTPException(404, "Sample not found") from None

    def submit(request):
        try:
            return ctx.jobs.submit(request).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # ---------- MIDI ----------

    @app.post("/api/samples/{sample_id}/midi")
    def make_midi(sample_id: str):
        """Transcribe the sample's notes into a .mid next to its WAV (tempo-matched for loops)."""
        record = sample_or_404(sample_id)
        if not ctx.library.path_of(record).exists():
            raise HTTPException(410, "The file was moved or deleted outside samplegen")
        out = ctx.library.midi_path(record)
        try:
            notes = ctx.midi.transcribe(ctx.library.path_of(record), out, bpm=record.params.get("bpm"))
        except MidiError as exc:
            raise HTTPException(409 if not ctx.midi.available else 500, str(exc)) from exc
        if notes == 0:
            out.unlink(missing_ok=True)
            raise HTTPException(422, "No notes found: MIDI works on pitched sounds (melodies, chords, bass).")
        return {"notes": notes, "url": f"/api/samples/{sample_id}/midi", "filename": out.name}

    @app.get("/api/samples/{sample_id}/midi")
    def get_midi(sample_id: str):
        path = ctx.library.midi_path(sample_or_404(sample_id))
        if not path.exists():
            raise HTTPException(404, "No MIDI for this sample yet")
        return FileResponse(path, media_type="audio/midi", filename=path.name)

    # ---------- library extras ----------

    @app.post("/api/samples/{sample_id}/rename")
    def rename(sample_id: str, body: RenameIn):
        sample_or_404(sample_id)
        try:
            return ctx.library.rename(sample_id, body.name).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/samples/{sample_id}/tags")
    def set_tags(sample_id: str, body: TagsIn):
        sample_or_404(sample_id)
        try:
            return ctx.library.set_tags(sample_id, body.tags).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/tags")
    def tags():
        return [{"tag": tag, "count": count} for tag, count in ctx.library.all_tags()]

    @app.post("/api/packs", status_code=201)
    def make_pack(body: PackIn):
        records = [sample_or_404(sid) for sid in dict.fromkeys(body.sample_ids)]
        fmt = body.export.to_format() if body.export else None
        try:
            result = export_pack(ctx.library, records, body.name, fmt, loudness=body.loudness,
                                 round_robin=body.round_robin)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:  # disk full, drive unplugged, same pack exported twice at once...
            raise HTTPException(500, f"Could not write the pack: {exc}") from exc
        if body.reveal:
            open_in_explorer(result.folder)
        return result.to_dict()

    # ---------- stems ----------

    @app.post("/api/samples/{sample_id}/stems", status_code=202)
    def split_stems(sample_id: str, body: StemsIn | None = None):
        record = sample_or_404(sample_id)
        path = ctx.library.path_of(record)
        if not path.exists():
            raise HTTPException(410, "The file was moved or deleted outside samplegen")
        p = record.params
        source = ctx.sources.import_file(path, record.name[:80], is_loop=record.mode == "loop",
                                         bpm=p.get("bpm"), bars=p.get("bars"), key=p.get("key"))
        fmt = (body or StemsIn()).export.to_format()
        return submit(StemRequest(source_id=source.id, export=fmt))

    # ---------- instruments ----------

    @app.post("/api/instruments", status_code=202)
    def make_instrument(body: InstrumentIn):
        return submit(InstrumentRequest(
            prompt=body.prompt, name=body.name, low_note=body.low_note, high_note=body.high_note,
            space=body.space, seed=body.seed, export=body.export.to_format(),
        ))
