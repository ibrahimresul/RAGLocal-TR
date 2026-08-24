from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    source: str | None = None


class FilterRequest(BaseModel):
    source: str | None = None


class ExportRequest(BaseModel):
    format: Literal["markdown", "json"] = "markdown"


class BenchmarkRequest(BaseModel):
    models: list[str] = Field(default_factory=list)

from fastapi import APIRouter

router = APIRouter()



import json
import queue
import threading
from dataclasses import asdict
from contextlib import contextmanager

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app import database
from app.documents import DOCS_DIR, get_index_freshness
from app.models import LocalLLM
from app.rag import EmptyIndexError, EmptyQuestionError, RAGService
from app.retrieval import get_top_chunks

from api import state



def _sse(event, data):
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _source_payload(source):
    payload = asdict(source)
    return payload


@contextmanager
def _stage(events, stage_name):
    events.put(_sse("stage", {"stage": stage_name, "state": "start"}))
    try:
        yield
    finally:
        events.put(_sse("stage", {"stage": stage_name, "state": "end"}))


@router.post("/ask")
async def ask(request: Request, body: AskRequest):
    events: "queue.Queue[dict | None]" = queue.Queue()
    cancelled = threading.Event()

    active_source = body.source if body.source is not None else state.get_source_filter()

    def worker():
        try:
            freshness = get_index_freshness(DOCS_DIR, database.DB_PATH)
            if freshness.status == "stale":
                events.put(_sse("index_warning", {
                    "status": freshness.status,
                    "message": f"The index is stale. {freshness.change_summary()}",
                }))
            elif freshness.status == "untracked":
                events.put(_sse("index_warning", {
                    "status": freshness.status,
                    "message": "The index source manifest is unavailable.",
                }))
            elif freshness.status == "error":
                events.put(_sse("index_warning", {
                    "status": freshness.status,
                    "message": f"Document changes could not be checked: {freshness.error}",
                }))

            resolved_question, carried_terms = state.follow_up_context.resolve(
                body.question,
            )
            if carried_terms:
                events.put(_sse("query_rewrite", {
                    "added_terms": list(carried_terms),
                }))

            def stream_callback(preview):
                if not cancelled.is_set():
                    events.put(_sse("token", {"text": preview}))

            service = RAGService(retrieval_func=get_top_chunks, llm_factory=LocalLLM)
            result = service.answer(
                resolved_question,
                source_name=active_source,
                activity_factory=lambda stage_name: _stage(events, stage_name),
                stream_callback=stream_callback,
            )

            if cancelled.is_set():
                return

            state.session_history.add_result(result)
            state.follow_up_context.remember(result.question)

            events.put(_sse("done", {
                "question": result.question,
                "answer": result.answer,
                "mode": result.mode,
                "best_score": result.best_score,
                "source_filter": result.source_filter,
                "warning": result.warning,
                "warning_solution": result.warning_solution,
                "sources": [_source_payload(source) for source in result.sources],
                "timings": {
                    "retrieval_seconds": result.timings.retrieval_seconds,
                    "generation_seconds": result.timings.generation_seconds,
                    "total_seconds": result.timings.total_seconds,
                },
            }))
        except EmptyQuestionError:
            events.put(_sse("error", {"message": "Question cannot be empty."}))
        except EmptyIndexError:
            if active_source:
                events.put(_sse("error", {
                    "message": (
                        f"{active_source} contains no searchable chunks."
                    ),
                }))
            else:
                events.put(_sse("error", {
                    "message": "No searchable index was found. Run /reindex.",
                }))
        except Exception as error:
            events.put(_sse("error", {
                "message": f"Document search failed: {error}",
            }))
        finally:
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            if await request.is_disconnected():
                cancelled.set()
                break
            try:
                item = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

    return EventSourceResponse(event_stream())



import json
import queue
import threading

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.evaluation import (
    BenchmarkPreparationError,
    benchmark_model,
    load_benchmark_cases,
    normalize_model_aliases,
    prepare_benchmark_cases,
    write_benchmark_report,
)
from app.models import LocalLLM
from app.retrieval import get_top_chunks




def _sse(event, data):
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@router.post("/benchmark")
async def run_benchmark(body: BenchmarkRequest):
    events: "queue.Queue[dict | None]" = queue.Queue()

    def worker():
        try:
            aliases = normalize_model_aliases(body.models or None)
            cases = load_benchmark_cases()
            prepared_cases = prepare_benchmark_cases(
                cases,
                retrieval_func=get_top_chunks,
            )
            events.put(_sse("prepared", {
                "model_count": len(aliases),
                "case_count": len(prepared_cases),
            }))

            model_results = []
            for alias in aliases:
                events.put(_sse("model_start", {"model": alias}))
                result = benchmark_model(
                    alias,
                    prepared_cases,
                    llm_factory=LocalLLM,
                )
                model_results.append(result)
                events.put(_sse("model_done", result))

            from datetime import datetime, timezone

            report = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "case_count": len(prepared_cases),
                "models": model_results,
            }
            path = write_benchmark_report(report)
            events.put(_sse("done", {"report_path": str(path), "report": report}))
        except BenchmarkPreparationError as error:
            events.put(_sse("error", {"message": str(error)}))
        except Exception as error:
            events.put(_sse("error", {"message": f"Benchmark failed: {error}"}))
        finally:
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            try:
                item = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            yield item

    return EventSourceResponse(event_stream())



import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app import database
from app.documents import (
    DOCS_DIR,
    DocumentManagementError,
    add_document,
    ingest_documents,
    remove_document,
)



@router.post("/documents")
async def upload_document(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")





    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / file.filename
        with temp_path.open("wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        try:
            destination = add_document(temp_path, DOCS_DIR)
        except DocumentManagementError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return {"source_name": destination.name}


@router.delete("/documents/{source_name}")
async def delete_document(source_name: str):
    try:
        remove_document(source_name, DOCS_DIR)
    except DocumentManagementError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"source_name": source_name}


@router.post("/reindex")
async def reindex():
    try:
        chunk_count = ingest_documents()
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"chunk_count": chunk_count}


@router.get("/stats")
async def stats():
    return database.get_chunk_stats()


@router.get("/sources")
async def sources():
    return {"sources": database.get_indexed_sources()}


@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: int):
    chunk = database.get_chunk_by_id(chunk_id)

    if chunk is None:
        raise HTTPException(status_code=404, detail=f"chunk_id={chunk_id} was not found.")

    return chunk



from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.settings import get_project_paths
from app.conversation import SessionExportError

from api import state



@router.get("/history")
async def history():
    return {"entries": [asdict(entry) for entry in state.session_history.entries]}


@router.get("/history/{entry_id}/repeat")
async def repeat(entry_id: int):
    entry = state.session_history.get(entry_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"History entry {entry_id} was not found.")

    return {"question": entry.question, "source": entry.source_filter}


@router.get("/filter")
async def get_filter():
    return {"source": state.get_source_filter()}


@router.post("/filter")
async def set_filter(body: FilterRequest):
    state.set_source_filter(body.source)
    return {"source": state.get_source_filter()}


@router.post("/export")
async def export_session(body: ExportRequest):
    export_dir = get_project_paths().session_export_dir

    try:
        destination = state.session_history.export(body.format, export_dir)
    except SessionExportError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    media_type = "text/markdown" if body.format == "markdown" else "application/json"
    return FileResponse(
        destination,
        media_type=media_type,
        filename=destination.name,
    )



from dataclasses import asdict

from fastapi import APIRouter

from app import settings as config
from app.database import DB_PATH
from app.documents import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from app.health import run_health_checks
from app.models import DEFAULT_MODEL_ALIAS, MODEL_ALIAS, get_model_alias_source
from app.settings import get_project_paths



@router.get("/health")
async def health():
    checks = [asdict(check) for check in run_health_checks()]
    statuses = {check["status"] for check in checks}

    if "error" in statuses:
        overall = "error"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "ok"

    return {"status": overall, "checks": checks}


@router.get("/model")
async def model():
    return {
        "alias": MODEL_ALIAS,
        "default_alias": DEFAULT_MODEL_ALIAS,
        "source": get_model_alias_source(),
    }


CONFIG_FIELDS = [
    ("TOP_K", config.TOP_K, "Number of top retrieval chunks to retain"),
    ("SIMILARITY_THRESHOLD", config.SIMILARITY_THRESHOLD, "Questions below this threshold are treated as out of scope"),
    ("CONTEXT_SCORE_THRESHOLD", config.CONTEXT_SCORE_THRESHOLD, "Minimum chunk score included in the LLM context"),
    ("CONTEXT_RELATIVE_SCORE_MARGIN", config.CONTEXT_RELATIVE_SCORE_MARGIN, "Maximum score gap allowed from the best result"),
    ("NEIGHBOR_CHUNK_RADIUS", config.NEIGHBOR_CHUNK_RADIUS, "Neighbor radius around a match for generative answers"),
    ("MAX_CONTEXT_CHUNKS", config.MAX_CONTEXT_CHUNKS, "Maximum total matches and neighbors sent to the model"),
    ("CONTEXT_TERM_EVIDENCE_MIN", config.CONTEXT_TERM_EVIDENCE_MIN, "Term evidence required for lower-ranked chunks to enter context"),
    ("USE_HYBRID_SEARCH", config.USE_HYBRID_SEARCH, "Run BM25 lexical search alongside semantic search"),
    ("RRF_K", config.RRF_K, "Smoothing constant used to combine search rankings"),
    ("TERM_EVIDENCE_THRESHOLD", config.TERM_EVIDENCE_THRESHOLD, "Minimum weighted question-term coverage required in context"),
    ("TERM_EVIDENCE_MIN_PREFIX", config.TERM_EVIDENCE_MIN_PREFIX, "Shortest common root required for Turkish suffix matching"),
    ("TERM_EVIDENCE_MIN_SHORT_ROOT", config.TERM_EVIDENCE_MIN_SHORT_ROOT, "Minimum length for exact coverage of short roots"),
    ("EXTRACTIVE_SCORE_THRESHOLD", config.EXTRACTIVE_SCORE_THRESHOLD, "Minimum score for a direct source answer"),
    ("USE_EXTRACTIVE_FALLBACK", config.USE_EXTRACTIVE_FALLBACK, "Enable safe extractive and fallback answers"),
    ("MAX_EXTRACTIVE_CHARS", config.MAX_EXTRACTIVE_CHARS, "Maximum source-text length shown directly"),
    ("GROUNDEDNESS_THRESHOLD", config.GROUNDEDNESS_THRESHOLD, "Minimum support ratio for a grounded generative answer"),
    ("MIN_GENERATIVE_ANSWER_CHARS", config.MIN_GENERATIVE_ANSWER_CHARS, "Shorter LLM answers are rejected"),
    ("USE_RERANKER", config.USE_RERANKER, "Cross-encoder reranking (evaluated and disabled)"),
    ("CHUNK_SIZE", CHUNK_SIZE, "Maximum tokens including special tokens"),
    ("CHUNK_OVERLAP", CHUNK_OVERLAP, "Token overlap between adjacent chunks"),
]


@router.get("/config")
async def get_config():
    project_paths = get_project_paths()
    fields = [
        {"name": name, "value": value, "description": description}
        for name, value, description in CONFIG_FIELDS
    ]
    fields.append({
        "name": "LOCAL_RAG_MODEL",
        "value": MODEL_ALIAS,
        "description": f"Chat model; default {DEFAULT_MODEL_ALIAS}",
    })
    fields.append({
        "name": "DOCS_DIR",
        "value": str(DOCS_DIR),
        "description": "Directory containing documents to index",
    })
    fields.append({
        "name": "DB_PATH",
        "value": str(DB_PATH),
        "description": "Generated SQLite index path",
    })
    fields.append({
        "name": "PROJECT_ROOT",
        "value": str(project_paths.root),
        "description": "Active Local RAG project directory",
    })

    return {"fields": fields}
