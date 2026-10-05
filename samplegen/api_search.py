"""API routes for searching the library by sound."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from .library import SampleNotFound
from .search import SearchUnavailable


def add_search_routes(app: FastAPI, ctx) -> None:
    def results(found):
        return [{**record.to_dict(), "score": round(score, 4)} for record, score in found]

    @app.get("/api/search/status")
    def search_status():
        return ctx.search.status()

    @app.get("/api/search")
    async def search(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(60, ge=1, le=500)):
        try:
            # the first query may wait for the worker to load the model: keep it off the event loop
            return results(await run_in_threadpool(ctx.search.search_text, q.strip(), limit))
        except SearchUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/samples/{sample_id}/similar")
    def similar(sample_id: str, limit: int = Query(60, ge=1, le=500)):
        try:
            ctx.library.get(sample_id)
            return results(ctx.search.similar(sample_id, limit))
        except SampleNotFound:
            raise HTTPException(404, "Sample not found") from None
        except SearchUnavailable as exc:
            raise HTTPException(409, str(exc)) from exc
