"""
main.py - FastAPI server that exposes the Quantum RAG pipeline to the Next.js UI.

    uvicorn main:app --port 8000          (add --reload while developing)
    python main.py                        (same thing, bound to 127.0.0.1)

Endpoints
    GET    /             health + model names + index sizes (the UI's status badge)
    POST   /query        streams the answer as newline-delimited JSON events
    POST   /upload       multipart file -> chunked, embedded, added to `user_uploads`
    GET    /documents    indexed documents (knowledge base + uploads)
    DELETE /uploads      remove everything the user uploaded

/query event stream (one JSON object per line, Content-Type: application/x-ndjson)
    {"type": "meta",  "mode": "grounded"|"general", "best_score": 0.71, "sources": [...]}
    {"type": "token", "text": "..."}                       (many)
    {"type": "done",  "mode": "...", "cited": [1, 3], "best_score": 0.71}
    {"type": "error", "message": "..."}                    (instead of done on failure)

`meta` always arrives first (retrieval happens before generation), so the UI can
show sources while the answer is still streaming. The server binds to localhost
only: it has no authentication and must not be exposed to a network as is.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import BASE_COLLECTION, SUPPORTED_EXTENSIONS, UPLOAD_COLLECTION, settings
from embeddings import GeminiError
from ingest import ingest_uploaded_file
from loaders import LoaderError
from rag import RagPipeline, Source
from store import VectorStore


MAX_SOURCE_TEXT = 1500  # characters of each retrieved chunk sent to the UI


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore()
    app.state.store = store
    app.state.pipeline = RagPipeline(store)
    yield


app = FastAPI(title="Quanta RAG API", version="2.0.0", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=12000)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    hybrid: Optional[bool] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _public_source(source: Source) -> dict:
    return {
        "index": source.index,
        "title": source.title,
        "section": source.section,
        "page": source.page,
        "doc_type": source.doc_type,
        "origin": source.origin,
        "collection": source.collection,
        "score": round(source.score, 4),
        "text": source.text[:MAX_SOURCE_TEXT],
        "url": source.url,
    }


def _line(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


class _ClientGone(Exception):
    """Raised inside the token callback to stop generating once the browser disconnected."""


def _stream_answer(pipeline: RagPipeline, request: QueryRequest) -> Iterator[bytes]:
    """Run the (blocking) pipeline in a worker thread and relay its events as NDJSON."""
    events: "queue.Queue[object]" = queue.Queue()
    finished = object()
    cancelled = threading.Event()

    def on_token(text: str) -> None:
        if cancelled.is_set():
            raise _ClientGone()
        events.put({"type": "token", "text": text})

    def on_sources(mode: str, sources: list[Source], best: float) -> None:
        events.put({"type": "meta", "mode": mode, "best_score": round(best, 4),
                    "sources": [_public_source(s) for s in sources]})

    def work() -> None:
        try:
            answer = pipeline.ask(
                request.query,
                history=[turn.model_dump() for turn in request.history],
                on_token=on_token,
                on_sources=on_sources,
                hybrid=request.hybrid,
            )
            events.put({"type": "done", "mode": answer.mode, "cited": answer.cited,
                        "best_score": round(answer.best_score, 4)})
        except _ClientGone:
            pass
        except (GeminiError, RuntimeError, LoaderError) as error:
            events.put({"type": "error", "message": str(error)})
        except Exception as error:  # noqa: BLE001 - never leave the stream hanging
            events.put({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            events.put(finished)

    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            item = events.get()
            if item is finished:
                break
            yield _line(item)  # type: ignore[arg-type]
    finally:
        cancelled.set()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def health() -> dict:
    store: VectorStore = app.state.store
    return {
        "status": "ok",
        "service": "Quanta RAG Engine",
        "ready": bool(settings.gemini_api_key),
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
        "knowledge_base_chunks": store.count(BASE_COLLECTION),
        "upload_chunks": store.count(UPLOAD_COLLECTION),
        "hybrid_default": settings.hybrid_enabled,
        "relevance_threshold": settings.relevance_threshold,
        "max_upload_mb": settings.max_upload_mb,
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
    }


@app.post("/query")
def query(request: QueryRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return StreamingResponse(
        _stream_answer(app.state.pipeline, request),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "").name
    extension = Path(name).suffix.lower()
    if not name or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    folder = settings.data_dir / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / name
    limit = settings.max_upload_mb * 1024 * 1024
    written, too_large = 0, False
    with destination.open("wb") as handle:
        while True:
            block = file.file.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > limit:
                too_large = True
                break
            handle.write(block)
    if too_large:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=f"File is larger than {settings.max_upload_mb} MB.")

    try:
        report = ingest_uploaded_file(app.state.store, destination)
    except LoaderError as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error))
    except (GeminiError, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=str(error))
    return {"ok": True, "name": name, "chunks": report.total, "embedded": report.embedded, "skipped": report.skipped}


@app.get("/documents")
def documents() -> dict:
    store: VectorStore = app.state.store
    return {
        "knowledge_base": store.list_sources(BASE_COLLECTION),
        "uploads": store.list_sources(UPLOAD_COLLECTION),
    }


@app.delete("/uploads")
def clear_uploads() -> dict:
    app.state.store.reset(UPLOAD_COLLECTION)
    folder = settings.data_dir / "uploads"
    removed = 0
    if folder.is_dir():
        for path in folder.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
    return {"ok": True, "files_removed": removed}

@app.get("/status")
def status() -> dict:
    return {"online": True, "hybrid_default": settings.hybrid_enabled}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


