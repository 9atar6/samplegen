"""API routes for style training and trained styles."""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .library import SampleNotFound
from .training import MAX_CLIPS, Clip, clean_caption, scan_folder


class ScanIn(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)


class ClipIn(BaseModel):
    path: str | None = Field(None, max_length=1000)  # a file on disk...
    sample_id: str | None = Field(None, max_length=20)  # ...or a library sample
    caption: str = Field(..., min_length=1, max_length=400)


class DescribeItemIn(BaseModel):
    path: str | None = Field(None, max_length=1000)
    sample_id: str | None = Field(None, max_length=20)


class DescribeIn(BaseModel):
    clips: list[DescribeItemIn] = Field(..., min_length=1, max_length=MAX_CLIPS)


class TrainIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    base: Literal["sfx", "music"] = "sfx"
    description: str = Field("", max_length=300)
    steps: int = 1500
    rank: int = 16
    clips: list[ClipIn] = Field(..., min_length=1, max_length=MAX_CLIPS)


def add_training_routes(app: FastAPI, ctx) -> None:
    def resolve(item) -> Path:
        if item.sample_id:
            try:
                record = ctx.library.get(item.sample_id)
            except SampleNotFound:
                raise HTTPException(404, f"Sample {item.sample_id} not found") from None
            path = ctx.library.path_of(record)
        elif item.path:
            path = Path(item.path)
        else:
            raise HTTPException(422, "Each sound needs a file path or a library sample.")
        if not path.is_file():
            raise HTTPException(422, f"File not found: {path}")
        return path

    def to_clip(item: ClipIn) -> Clip:
        return Clip(resolve(item), clean_caption(item.caption))

    @app.post("/api/training/describe")
    def auto_describe(body: DescribeIn):
        paths = [resolve(item) for item in body.clips]
        # CLAP moves to the CPU while a style is training, so it never fights the trainer for VRAM.
        captions = ctx.describer.describe(paths, cpu=ctx.training.busy_reason() is not None)
        return {
            "clap": ctx.describer.clap_available,
            "captions": [
                {"path": item.path, "sample_id": item.sample_id, "caption": captions[str(p.resolve())]}
                for item, p in zip(body.clips, paths)
            ],
        }

    @app.get("/api/training")
    def training_status():
        return ctx.training.status()

    @app.post("/api/training/scan")
    def scan(body: ScanIn):
        try:
            clips = scan_folder(Path(body.path.strip().strip('"')))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return [{"path": str(c.path), "name": c.path.name, "caption": c.caption} for c in clips]

    @app.post("/api/training/start", status_code=202)
    def start(body: TrainIn):
        clips = [to_clip(item) for item in body.clips]
        try:
            ctx.training.start(body.name, body.base, clips, steps=body.steps, rank=body.rank,
                               description=body.description)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return ctx.training.status()

    @app.post("/api/training/stop")
    def stop():
        ctx.training.stop()
        return ctx.training.status()

    @app.get("/api/styles")
    def styles():
        return [s.to_dict() for s in ctx.styles.list()]
