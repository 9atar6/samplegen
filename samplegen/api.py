"""HTTP API + static web UI."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .api_extras import add_extra_routes, open_in_explorer
from .api_training import add_training_routes
from .audio import ExportFormat
from .autodescribe import Describer
from .catalog import KEYS, LOOP_BARS, LOOP_BPMS, LOOP_TAGS, MODELS, SCALES
from .generation_request import GenerationRequest
from .jobs import JobManager
from .library import FileInUse, Library, SampleNotFound
from .setup_check import FIX as SETUP_FIX
from .setup_check import missing_parts
from .sources import MAX_UPLOAD_BYTES, SourceNotFound, SourceStore
from .styles import StyleStore
from .training import TrainingManager

WEB_DIR = Path(__file__).parent / "web"


class ExportIn(BaseModel):
    sample_rate: int = 44100
    bit_depth: Literal["32f", "24", "16"] = "32f"


class GenerateIn(BaseModel):
    mode: Literal["sfx", "loop", "free", "transform", "edit"]
    model: str
    prompt: str = Field("", max_length=2000)
    negative_prompt: str = Field("", max_length=2000)
    duration: float = 4.0
    bpm: int = 120
    bars: int = 4
    key: str = "C"
    scale: str = "minor"
    variations: int = 4
    seed: int | None = None
    normalize_db: float | None = -1.0
    trim_silence: bool = True
    fade_in_ms: float = 0.0
    fade_out_ms: float = 5.0
    export: ExportIn = ExportIn()
    source_id: str | None = Field(None, max_length=20)
    strength: float = 0.6
    source_is_loop: bool = False
    edit_op: Literal["inpaint", "extend"] = "inpaint"
    regions: list[tuple[float, float]] = Field(default_factory=list, max_length=8)
    extend_seconds: float = 4.0
    style: str | None = Field(None, max_length=40)
    style_strength: float = 1.0


class StatusIn(BaseModel):
    status: Literal["new", "kept", "trashed"]


class FavoriteIn(BaseModel):
    favorite: bool


@dataclass
class AppContext:
    library: Library
    jobs: JobManager
    engine: object  # samplegen.engine.Engine (or a test double with .state/.error)
    sources: SourceStore | None = None
    styles: StyleStore | None = None
    training: TrainingManager | None = None
    describer: Describer | None = None

    def __post_init__(self):
        root = self.library.root
        if self.sources is None:
            self.sources = SourceStore(root)
        if self.styles is None:
            self.styles = StyleStore(root / "_styles")
        if self.training is None:  # not installed: status() reports it, start() explains how
            self.training = TrainingManager(root / "_no_trainer", root, root / "_no_engine", self.styles)
        if self.describer is None:  # without the trainer: measurements + file names only
            self.describer = Describer(root / "_no_trainer" / "python.exe", root / "_no_trainer" / "clap_describe.py")


def to_request(body: GenerateIn) -> GenerationRequest:
    data = body.model_dump()
    export = data.pop("export")
    data["regions"] = tuple(tuple(r) for r in data["regions"])
    try:
        return GenerationRequest(**data, export=ExportFormat(**export))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def create_app(ctx: AppContext) -> FastAPI:
    app = FastAPI(title="samplegen", docs_url=None, redoc_url=None)
    # Only answer to this machine's own names: blocks DNS-rebinding pages from reading the API.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        # A web page you visit could otherwise POST to localhost (trash samples, open Explorer...).
        # Browsers always send Origin on cross-site POSTs; samplegen's own page is same-origin.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            host = urlsplit(origin).hostname
            if host not in ("127.0.0.1", "localhost", "testserver"):
                return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(FileInUse)
    async def file_in_use(_request: Request, exc: FileInUse):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(sqlite3.Error)
    async def library_unavailable(_request: Request, exc: sqlite3.Error):
        return JSONResponse({"detail": f"The sample library isn't reachable ({exc}). "
                                       "Is its drive still plugged in?"}, status_code=503)

    @app.middleware("http")
    async def revalidate_ui(request: Request, call_next):
        # Browsers cache ES modules hard; "no-cache" makes them check (cheap ETag round-trip on
        # localhost) so an updated samplegen never runs a stale mix of old and new scripts.
        response = await call_next(request)
        if request.url.path.startswith("/ui/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response

    def sample_or_404(sample_id: str):
        try:
            return ctx.library.get(sample_id)
        except SampleNotFound:
            raise HTTPException(404, "Sample not found") from None

    @app.get("/api/status")
    def status():
        return {"engine": ctx.engine.state, "engine_error": ctx.engine.error, "library": str(ctx.library.root),
                "missing": missing_parts(ctx.engine.engine_dir), "missing_fix": SETUP_FIX}

    @app.get("/api/catalog")
    def catalog():
        return {
            "models": [
                {"key": m.key, "label": m.label, "description": m.description, "modes": list(m.modes),
                 "max_seconds": m.max_seconds, "default_seconds": m.default_seconds}
                for m in MODELS.values()
            ],
            "loop": {"bpms": LOOP_BPMS, "bars": LOOP_BARS, "keys": KEYS, "scales": SCALES, "tags": LOOP_TAGS},
            "instrument": {"lowest": "C1", "highest": "C7", "max_notes": 61, "tags": {
                k: LOOP_TAGS[k] for k in ("Instrument", "Timbre")}},
        }

    @app.post("/api/generate", status_code=202)
    def generate(body: GenerateIn):
        request = to_request(body)
        try:
            job = ctx.jobs.submit(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return job.to_dict()

    @app.get("/api/jobs")
    def jobs():
        return [j.to_dict() for j in ctx.jobs.recent()[:30]]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        found = ctx.jobs.get(job_id)
        if found is None:
            raise HTTPException(404, "Job not found")
        return found.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        found = ctx.jobs.cancel(job_id)
        if found is None:
            raise HTTPException(404, "Job not found")
        return found.to_dict()

    @app.get("/api/samples")
    def samples(status: Literal["new", "kept", "trashed"] | None = None, favorite: bool | None = None,
                q: str | None = Query(None, max_length=200), batch: str | None = None,
                tag: str | None = Query(None, max_length=32), limit: int = 200, offset: int = 0):
        limit = max(1, min(limit, 2000))
        records = ctx.library.list(status=status, favorite=favorite, query=q, batch_id=batch, tag=tag,
                                   limit=limit, offset=max(0, offset))
        return [r.to_dict() for r in records]

    @app.get("/api/samples/{sample_id}")
    def sample(sample_id: str):
        return sample_or_404(sample_id).to_dict()

    @app.get("/api/samples/{sample_id}/audio")
    def audio(sample_id: str):
        record = sample_or_404(sample_id)
        path = ctx.library.path_of(record)
        if not path.exists():
            raise HTTPException(410, "The file was moved or deleted outside samplegen")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @app.post("/api/samples/{sample_id}/status")
    def set_status(sample_id: str, body: StatusIn):
        sample_or_404(sample_id)
        return ctx.library.set_status(sample_id, body.status).to_dict()

    @app.post("/api/samples/{sample_id}/favorite")
    def set_favorite(sample_id: str, body: FavoriteIn):
        sample_or_404(sample_id)
        return ctx.library.set_favorite(sample_id, body.favorite).to_dict()

    @app.post("/api/samples/{sample_id}/reveal", status_code=204)
    def reveal(sample_id: str):
        record = sample_or_404(sample_id)
        folder = record.params.get("instrument_folder")
        if record.mode == "instrument" and folder:
            open_in_explorer(ctx.library.root / folder)  # the playable instrument, not just its preview
        else:
            open_in_explorer(ctx.library.path_of(record), select=True)

    add_extra_routes(app, ctx)
    add_training_routes(app, ctx)

    # ---------- sources (Transform / Edit) ----------

    def source_or_404(source_id: str):
        try:
            return ctx.sources.get(source_id)
        except SourceNotFound:
            raise HTTPException(404, "Source not found") from None

    @app.post("/api/sources", status_code=201)
    async def upload_source(request: Request, filename: str = Query("source", max_length=200)):
        too_big = HTTPException(413, f"Files are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        declared = request.headers.get("content-length") or "0"
        if not declared.isdigit() or int(declared) > MAX_UPLOAD_BYTES:
            raise too_big if declared.isdigit() else HTTPException(400, "Bad Content-Length")
        body, received = bytearray(), 0
        async for chunk in request.stream():  # enforce the cap even without a Content-Length
            received += len(chunk)
            if received > MAX_UPLOAD_BYTES:
                raise too_big
            body += chunk
        try:
            # Decoding can take seconds: keep it off the event loop so the UI stays responsive.
            source = await run_in_threadpool(ctx.sources.import_bytes, bytes(body), filename)
            return source.to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/sources/from-sample/{sample_id}", status_code=201)
    def source_from_sample(sample_id: str):
        record = sample_or_404(sample_id)
        path = ctx.library.path_of(record)
        if not path.exists():
            raise HTTPException(410, "The file was moved or deleted outside samplegen")
        p = record.params
        source = ctx.sources.import_file(path, record.name[:80], is_loop=record.mode == "loop" or bool(p.get("loop")),
                                         bpm=p.get("bpm"), bars=p.get("bars"), key=p.get("key"))
        return source.to_dict()

    @app.get("/api/sources/{source_id}")
    def source(source_id: str):
        return source_or_404(source_id).to_dict()

    @app.get("/api/sources/{source_id}/audio")
    def source_audio(source_id: str):
        source_or_404(source_id)
        return FileResponse(ctx.sources.audio_path(source_id), media_type="audio/wav")

    @app.get("/")
    def index():
        return RedirectResponse("/ui/index.html")

    app.mount("/ui", StaticFiles(directory=WEB_DIR), name="ui")
    return app
