"""FastAPI service for OpsPilot queries, documents, streaming and feedback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.errors import APIError, ErrorCode
from app.model_service import ModelQueryService
from app.schemas import (
    DocumentCreateRequest,
    DocumentResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
)
from app.services import DemoQueryService, DocumentRegistry, new_request_id
from core.model_settings import ModelSettings
from core.observability import JsonEventLogger, question_fingerprint


def error_payload(
    request: Request, code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "error_code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", "unknown"),
        "details": details or {},
    }


def create_app(
    *,
    ready: bool = True,
    query_timeout_seconds: float | None = None,
    settings: ModelSettings | None = None,
    query_service: Any = None,
    log_path: Path | None = None,
) -> FastAPI:
    """Application factory keeps tests isolated and model loading explicit."""
    settings = settings or ModelSettings()
    owned = query_service is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> Any:
        try:
            yield
        finally:
            if owned and hasattr(application.state.query_service, "aclose"):
                await application.state.query_service.aclose()

    app = FastAPI(
        lifespan=lifespan,
        title="OpsPilot API",
        version="0.10.0",
        description="Evidence-grounded deployment troubleshooting API",
    )
    app.state.ready = ready
    app.state.registry = DocumentRegistry()
    app.state.query_service = query_service if query_service is not None else (
        ModelQueryService(settings) if settings.opspilot_mode == "model"
        else DemoQueryService(app.state.registry)
    )
    app.state.feedback = []
    app.state.query_timeout_seconds = (
        query_timeout_seconds if query_timeout_seconds is not None
        else settings.request_timeout_seconds
    )
    app.state.events = JsonEventLogger(
        log_path if log_path is not None else settings.event_log_path,
        stdout=settings.log_json_stdout,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request.state.request_id = request.headers.get("X-Request-ID") or new_request_id()
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            app.state.events.emit(
                "request_failed",
                request_id=request.state.request_id,
                method=request.method,
                path=request.url.path,
                error_type="unhandled_exception",
            )
            raise
        response.headers["X-Request-ID"] = request.state.request_id
        app.state.events.emit(
            "http_request",
            request_id=request.state.request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(request, exc.code.value, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_payload(
                request,
                ErrorCode.INVALID_REQUEST.value,
                "Request validation failed",
                {"errors": exc.errors()},
            ),
        )

    @app.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        """Liveness only proves that the Python process can answer."""
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        """Readiness prevents traffic before models and indexes are usable."""
        if not app.state.ready or (
            hasattr(app.state.query_service, "health")
            and not await app.state.query_service.health()
        ):
            raise APIError(
                ErrorCode.NOT_READY,
                "Models or indexes are still loading",
                503,
            )
        return {"status": "ready"}

    @app.post("/v1/query", response_model=QueryResponse, tags=["query"])
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        if not app.state.ready:
            raise APIError(ErrorCode.NOT_READY, "Service is not ready", 503)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(app.state.query_timeout_seconds):
                result = await app.state.query_service.answer(
                    body.question, body.top_k,
                        **({} if body.use_cache else {"use_cache": False})
                )
        except TimeoutError as exc:
            raise APIError(
                ErrorCode.QUERY_TIMEOUT,
                "Query exceeded the configured timeout",
                504,
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        app.state.events.emit(
            "query_completed",
            request_id=request.state.request_id,
            **question_fingerprint(body.question),
            latency_ms=latency_ms,
            stage_ms=result["timings_ms"],
            token_total=result["token_total"],
            cache_hit=result["cache_hit"],
            error_type=result.get("error_code", ""),
            status=result["status"],
        )
        sources = result["sources"] if body.include_sources else []
        return QueryResponse(
            request_id=request.state.request_id,
            status=result["status"],
            answer=result["answer"],
            clarification=result.get("clarification", ""),
            citations=result.get("citations", []),
            sources=sources,
            timings_ms=result["timings_ms"],
            cache_hit=result["cache_hit"],
            degraded=result.get("degraded", False),
            error_code=result.get("error_code", ""),
            citation_valid=result.get("citation_valid"),
            usage=result.get("usage", {}),
        )

    @app.post("/v1/query/stream", tags=["query"])
    async def stream_query(body: QueryRequest, request: Request) -> StreamingResponse:
        """Send Server-Sent Events and stop when the client disconnects."""
        if not app.state.ready:
            raise APIError(ErrorCode.NOT_READY, "Service is not ready", 503)

        async def generate_model() -> Any:
            started = time.perf_counter()
            try:
                async with asyncio.timeout(app.state.query_timeout_seconds):
                    async with aclosing(
                        app.state.query_service.stream(
                            body.question, body.top_k,
                            **({} if body.use_cache else {"use_cache": False}),
                        )
                    ) as events:
                        async for event, data in events:
                            if event == "meta":
                                data = {**data, "request_id": request.state.request_id}
                            elif event == "sources" and not body.include_sources:
                                data = []
                            elif event == "result":
                                app.state.events.emit(
                                    "query_completed",
                                    request_id=request.state.request_id,
                                    **question_fingerprint(body.question),
                                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                                    stage_ms=data["timings_ms"],
                                    token_total=data["token_total"],
                                    cache_hit=data["cache_hit"],
                                    error_type=data.get("error_code", ""),
                                    status=data["status"], transport="sse",
                                )
                                data = QueryResponse(
                                    **{**data, "request_id": request.state.request_id,
                                       "sources": data["sources"] if body.include_sources else []}
                                ).model_dump()
                            yield _sse(event, data)
            except TimeoutError:
                yield _sse("error", {
                    "error_code": ErrorCode.QUERY_TIMEOUT.value,
                    "request_id": request.state.request_id,
                    "discard_partial": True,
                })
            except asyncio.CancelledError:
                app.state.events.emit("stream_cancelled", request_id=request.state.request_id)
                raise

        async def generate() -> Any:
            started = time.perf_counter()
            try:
                async with asyncio.timeout(app.state.query_timeout_seconds):
                    result = await app.state.query_service.answer(
                        body.question, body.top_k,
                        **({} if body.use_cache else {"use_cache": False})
                    )
            except TimeoutError:
                yield _sse(
                    "error",
                    {
                        "error_code": ErrorCode.QUERY_TIMEOUT.value,
                        "request_id": request.state.request_id,
                    },
                )
                return
            meta = {"request_id": request.state.request_id, "status": result["status"]}
            yield _sse("meta", meta)
            answer = result["answer"]
            for start in range(0, len(answer), 12):
                if await request.is_disconnected():
                    app.state.events.emit(
                        "stream_cancelled", request_id=request.state.request_id
                    )
                    return
                yield _sse("token", {"delta": answer[start : start + 12]})
                await asyncio.sleep(0)
            yield _sse("sources", result["sources"] if body.include_sources else [])
            yield _sse("done", {"timings_ms": result["timings_ms"]})
            app.state.events.emit(
                "query_completed",
                request_id=request.state.request_id,
                **question_fingerprint(body.question),
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                stage_ms=result["timings_ms"],
                token_total=result["token_total"],
                cache_hit=result["cache_hit"],
                error_type="",
                status=result["status"],
                transport="sse",
            )

        return StreamingResponse(
            generate_model() if hasattr(app.state.query_service, "stream") else generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/documents", response_model=DocumentResponse, tags=["documents"])
    async def create_document(body: DocumentCreateRequest) -> DocumentResponse:
        row = app.state.registry.create(
            body.title, str(body.source_url), body.product, body.content
        )
        return _document_view(row)

    @app.get(
        "/v1/documents",
        response_model=list[DocumentResponse],
        tags=["documents"],
    )
    async def list_documents() -> list[DocumentResponse]:
        return [_document_view(row) for row in app.state.registry.list()]

    @app.delete("/v1/documents/{document_id}", tags=["documents"])
    async def delete_document(document_id: str) -> dict[str, bool]:
        if not app.state.registry.delete(document_id):
            raise APIError(
                ErrorCode.DOCUMENT_NOT_FOUND,
                f"Document {document_id} was not found",
                404,
            )
        return {"deleted": True}

    @app.post("/v1/feedback", response_model=FeedbackResponse, tags=["feedback"])
    async def feedback(body: FeedbackRequest) -> FeedbackResponse:
        feedback_id = f"fb_{uuid.uuid4().hex}"
        # Raw question text is never part of feedback. Comments are kept only
        # in this in-memory demo; production should apply retention controls.
        app.state.feedback.append(
            {
                "feedback_id": feedback_id,
                "request_id": body.request_id,
                "rating": body.rating,
                "category": body.category,
                "comment": body.comment,
            }
        )
        app.state.events.emit(
            "feedback_received",
            feedback_id=feedback_id,
            request_id=body.request_id,
            rating=body.rating,
            category=body.category,
        )
        return FeedbackResponse(accepted=True, feedback_id=feedback_id)

    return app


def _document_view(row: dict[str, Any]) -> DocumentResponse:
    content = str(row["content"])
    digest = row.get("content_sha256") or hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
    return DocumentResponse(
        document_id=row["document_id"],
        title=row["title"],
        source_url=row["source_url"],
        product=row["product"],
        content_sha256=digest,
    )


def _sse(event: str, data: Any) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


app = create_app()
