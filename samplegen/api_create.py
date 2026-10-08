"""API routes for making many sounds: variations of a sample, shot-list batches."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .api_extras import ExportFormatIn
from .batch import parse_shot_list, shot_requests, variation_request
from .catalog import MODELS
from .kits import MAX_ROUND_ROBINS, KitRequest
from .layers import LayerRequest
from .library import SampleNotFound


class VariationsIn(BaseModel):
    count: int = Field(4, ge=1, le=8)
    strength: float = Field(0.35, ge=0.05, le=1.0)


class BatchIn(BaseModel):
    name: str = Field("", max_length=60)
    text: str = Field(..., min_length=1, max_length=50_000)
    model: str = "sa3-sfx"
    seconds: float = Field(4.0, ge=0.5, le=380)


class KitIn(BaseModel):
    prompt: str = Field(..., max_length=400)
    name: str = Field("", max_length=80)
    round_robins: int = Field(3, ge=1, le=MAX_ROUND_ROBINS)
    seed: int | None = None
    export: ExportFormatIn = ExportFormatIn(bit_depth="24")


class LayersIn(BaseModel):
    character: str = Field("", max_length=400)
    transient: str = Field("", max_length=400)
    body: str = Field("", max_length=400)
    tail: str = Field("", max_length=400)
    tail_seconds: float = 4.0
    body_offset_ms: float = 0.0
    tail_offset_ms: float = 30.0
    gains_db: tuple[float, float, float] = (0.0, 0.0, -3.0)
    variations: int = 2
    name: str = Field("", max_length=80)
    model: str = "sa3-sfx"
    seed: int | None = None
    export: ExportFormatIn = ExportFormatIn()


def add_create_routes(app: FastAPI, ctx) -> None:
    @app.post("/api/layers", status_code=202)
    def make_layers(body: LayersIn):
        if body.model not in ("sa3-sfx", "sa3-medium"):
            raise HTTPException(422, "Layers use Stable Audio 3 (Small SFX or Medium).")
        data = body.model_dump()
        data["export"] = body.export.to_format()
        try:
            return ctx.jobs.submit(LayerRequest(**data)).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/kits", status_code=202)
    def make_kit(body: KitIn):
        request = KitRequest(prompt=body.prompt, name=body.name, round_robins=body.round_robins, seed=body.seed,
                             export=body.export.to_format())
        try:
            return ctx.jobs.submit(request).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/samples/{sample_id}/variations", status_code=202)
    def variations(sample_id: str, body: VariationsIn):
        try:
            record = ctx.library.get(sample_id)
        except SampleNotFound:
            raise HTTPException(404, "Sample not found") from None
        if record.mode in ("instrument", "kit"):
            raise HTTPException(422, "Instruments and kits are made of many notes: generate a new one instead.")
        path = ctx.library.path_of(record)
        if not path.exists():
            raise HTTPException(410, "The file was moved or deleted outside samplegen")
        looping = record.mode == "loop" or bool(record.params.get("loop"))
        source = ctx.sources.import_file(path, record.name[:80], is_loop=looping)
        try:
            return ctx.jobs.submit(variation_request(record, source.id, body.count, body.strength)).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/batches/preview")
    def preview_batch(body: BatchIn):
        try:
            shots = parse_shot_list(body.text)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "sounds": 0, "lines": 0}
        return {"ok": True, "lines": len(shots), "sounds": sum(s.count for s in shots)}

    @app.post("/api/batches", status_code=202)
    def start_batch(body: BatchIn):
        if body.model not in ("sa3-sfx", "sa3-medium"):
            raise HTTPException(422, "Batches use Stable Audio 3 (Small SFX or Medium).")
        try:
            shots = parse_shot_list(body.text)
            tag = body.name.strip() or "batch"
            requests = shot_requests(shots, tag, body.model, min(body.seconds, MODELS[body.model].max_seconds))
            for request in requests:  # check everything before queueing anything
                ctx.jobs.generator.check(request)
            jobs = [ctx.jobs.submit(r).to_dict() for r in requests]
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"tag": requests[0].tags[0] if requests and requests[0].tags else "", "jobs": jobs,
                "sounds": sum(s.count for s in shots)}
