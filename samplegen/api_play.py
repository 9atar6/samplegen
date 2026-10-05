"""API routes for the Play tab: instruments to play, hum-to-notes, saving performances."""

import io
import re
import tempfile
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .library import NewSample, SampleNotFound
from .midi import MidiError
from .play import instrument_info, list_instruments

MAX_HUM_BYTES = 30 * 1024 * 1024
MAX_TAKE_BYTES = 400 * 1024 * 1024  # 10 min of 44.1 kHz stereo float
MAX_TAKE_SECONDS = 600
MAX_MIDI_BYTES = 2 * 1024 * 1024


async def read_body(request: Request, limit: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, f"That's too big (limit {limit // (1024 * 1024)} MB).")
    if not body:
        raise HTTPException(400, "Nothing was sent.")
    return bytes(body)


def add_play_routes(app: FastAPI, ctx) -> None:
    def instrument_or_404(instrument_id: str):
        try:
            info = instrument_info(ctx.library, ctx.library.get(instrument_id))
        except SampleNotFound:
            info = None
        if info is None:
            raise HTTPException(404, "Instrument not found (moved or deleted outside samplegen?)")
        return info

    @app.get("/api/instruments")
    def instruments():
        return [i.to_dict() for i in list_instruments(ctx.library)]

    @app.get("/api/instruments/{instrument_id}")
    def instrument(instrument_id: str):
        info = instrument_or_404(instrument_id)
        notes = [{"midi": m, "url": f"/api/instruments/{instrument_id}/notes/{m}"} for m in sorted(info.notes)]
        return {**info.to_dict(), "notes": notes}

    @app.get("/api/instruments/{instrument_id}/notes/{midi}")
    def instrument_note(instrument_id: str, midi: int):
        path = instrument_or_404(instrument_id).notes.get(midi)
        if path is None or not path.exists():
            raise HTTPException(404, "No sample for that note")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/midi/hum")
    async def hum(request: Request):
        """A sung / hummed / whistled melody (WAV) -> its notes, to play on an instrument."""
        data = await read_body(request, MAX_HUM_BYTES)
        if not ctx.midi.available:
            raise HTTPException(409, "MIDI extraction isn't installed yet: double-click tools\\install-midi.bat.")

        def transcribe():
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "hum.wav"
                wav.write_bytes(data)
                return ctx.midi.notes(wav, Path(tmp) / "hum.mid")

        try:
            events = await run_in_threadpool(transcribe)
        except MidiError as exc:
            raise HTTPException(500, str(exc)) from exc
        if not events:
            raise HTTPException(422, "No notes heard. Hum or whistle clearly, one note at a time, close to the mic.")
        return {"events": events}

    @app.post("/api/performances", status_code=201)
    async def save_performance(request: Request, name: str = Query("Performance", max_length=120),
                               instrument: str = Query("", max_length=40), bpm: float = Query(120, ge=20, le=300)):
        """A take rendered in the browser (WAV) -> a library sample (Kept/Performances once kept)."""
        data = await read_body(request, MAX_TAKE_BYTES)
        try:
            info = sf.info(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001 - whatever libsndfile says, it isn't audio we can keep
            raise HTTPException(422, "That isn't a WAV file samplegen can read.") from exc
        duration = info.frames / info.samplerate if info.samplerate else 0
        if not 0 < duration <= MAX_TAKE_SECONDS:
            raise HTTPException(422, f"Takes can be up to {MAX_TAKE_SECONDS // 60} minutes.")
        ctx.library.root.joinpath("_staging").mkdir(parents=True, exist_ok=True)
        staged = ctx.library.root / "_staging" / f"take_{uuid.uuid4().hex[:10]}.wav"
        staged.write_bytes(data)
        title = re.sub(r"\s+", " ", name).strip() or "Performance"
        played_on = ""
        if instrument:
            try:
                played_on = ctx.library.get(instrument).name
            except SampleNotFound:
                pass
        record = ctx.library.add(staged, NewSample(
            mode="performance", model="play", prompt=f"played on {played_on}" if played_on else "played live",
            negative_prompt="", seed=0, params={"bpm": bpm, "instrument": instrument or None, "duration": duration},
            duration=duration, sample_rate=info.samplerate, batch_id=uuid.uuid4().hex[:8], name=title,
        ))
        return record.to_dict()

    @app.put("/api/samples/{sample_id}/midi", status_code=204)
    async def attach_midi(sample_id: str, request: Request):
        """Store a .mid next to a sample (the notes of a performance)."""
        try:
            record = ctx.library.get(sample_id)
        except SampleNotFound:
            raise HTTPException(404, "Sample not found") from None
        data = await read_body(request, MAX_MIDI_BYTES)
        if not data.startswith(b"MThd"):
            raise HTTPException(422, "That isn't a MIDI file.")
        ctx.library.midi_path(record).write_bytes(data)
