from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from serving.dependencies import RetinaService, get_service, load_service
from serving.errors import RetinaInvalidRequestError, RetinaServiceError
from serving.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ImageRecommendationResponse,
    MetadataResponse,
    MetricsSummaryResponse,
    ProfileRecommendationRequest,
    ProfileRecommendationResponse,
    ServiceMetadata,
    TextRecommendationRequest,
    TextRecommendationResponse,
)

logger = logging.getLogger("retina.serving")


def _service_metadata(service: RetinaService) -> ServiceMetadata:
    return ServiceMetadata(**service.service_metadata())


def _render_error(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def create_app(
    service: RetinaService | None = None,
    config_path: str | Path = "configs/retina.yaml",
) -> FastAPI:
    if service is None:
        service = load_service(config_path)

    app = FastAPI(title="Retina API", version="0.2.0")
    app.state.retina_service = service
    app.state.retina_settings = service.settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=service.settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        started = perf_counter()
        response = await call_next(request)
        latency_ms = (perf_counter() - started) * 1000.0
        response.headers["X-Request-Latency-Ms"] = f"{latency_ms:.2f}"
        logger.info(
            "request_summary method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return _render_error(422, "validation_error", "Request validation failed", {"errors": exc.errors()})

    @app.exception_handler(RetinaServiceError)
    async def retina_error_handler(_: Request, exc: RetinaServiceError):
        return _render_error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(HTTPException)
    async def http_error_handler(_: Request, exc: HTTPException):
        details = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        code = str(details.get("code", "http_error"))
        message = str(details.get("message", exc.detail))
        return _render_error(exc.status_code, code, message, details)

    @app.on_event("startup")
    async def startup_checks() -> None:
        logger.info(
            "retina_startup model=%s dataset=%s device=%s config=%s",
            service.settings.model_name,
            service.settings.dataset_name,
            service.settings.device,
            service.settings.config_path,
        )

    @app.get("/health", response_model=HealthResponse)
    def health(service: RetinaService = Depends(get_service)) -> HealthResponse:
        return HealthResponse(service=_service_metadata(service), **service.health())

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(service: RetinaService = Depends(get_service)) -> MetadataResponse:
        return MetadataResponse(service=_service_metadata(service))

    @app.get("/metrics/summary", response_model=MetricsSummaryResponse)
    def metrics_summary(service: RetinaService = Depends(get_service)) -> MetricsSummaryResponse:
        payload = service.metrics_summary()
        return MetricsSummaryResponse(
            service=_service_metadata(service),
            metrics=payload["metrics"],
            summary_path=payload["summary_path"],
        )

    @app.get("/recommend/text", response_model=TextRecommendationResponse)
    def recommend_text(
        query: str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=100),
        service: RetinaService = Depends(get_service),
    ) -> TextRecommendationResponse:
        if not query.strip():
            raise RetinaInvalidRequestError("query is required", details={"field": "query"})
        results, latency_ms = service.recommend_text(query=query, top_k=top_k)
        return TextRecommendationResponse(
            service=_service_metadata(service),
            query=query,
            top_k=top_k,
            latency_ms=latency_ms,
            results=results,
        )

    @app.post("/search/text", response_model=TextRecommendationResponse)
    def search_text(
        payload: TextRecommendationRequest,
        service: RetinaService = Depends(get_service),
    ) -> TextRecommendationResponse:
        results, latency_ms = service.recommend_text(query=payload.query, top_k=payload.top_k)
        return TextRecommendationResponse(
            service=_service_metadata(service),
            query=payload.query,
            top_k=payload.top_k,
            latency_ms=latency_ms,
            results=results,
        )

    @app.get("/recommend/image", response_model=ImageRecommendationResponse)
    def recommend_image(
        image_id: str = Query(..., min_length=1),
        top_k: int = Query(default=10, ge=1, le=100),
        service: RetinaService = Depends(get_service),
    ) -> ImageRecommendationResponse:
        if not image_id.strip():
            raise RetinaInvalidRequestError("image_id is required", details={"field": "image_id"})
        results, latency_ms = service.recommend_image(image_id=image_id, top_k=top_k)
        return ImageRecommendationResponse(
            service=_service_metadata(service),
            image_id=image_id,
            top_k=top_k,
            latency_ms=latency_ms,
            results=results,
        )

    @app.post("/recommend/profile", response_model=ProfileRecommendationResponse)
    def recommend_profile(
        payload: ProfileRecommendationRequest,
        service: RetinaService = Depends(get_service),
    ) -> ProfileRecommendationResponse:
        results, latency_ms = service.recommend_profile(
            text_queries=payload.text_queries,
            liked_image_ids=payload.liked_image_ids,
            top_k=payload.top_k,
        )
        return ProfileRecommendationResponse(
            service=_service_metadata(service),
            text_queries=payload.text_queries,
            liked_image_ids=payload.liked_image_ids,
            top_k=payload.top_k,
            latency_ms=latency_ms,
            results=results,
        )

    @app.get("/artifacts/images/{relative_path:path}")
    def serve_artifact_image(relative_path: str, service: RetinaService = Depends(get_service)) -> FileResponse:
        root = service.settings.image_artifact_dir.resolve()
        candidate = (root / relative_path).resolve()
        if root not in candidate.parents and candidate != root:
            raise RetinaInvalidRequestError(
                "Requested image is outside the configured artifact directory",
                details={"relative_path": relative_path},
            )
        if not candidate.exists():
            raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Image artifact not found"})
        return FileResponse(str(candidate))

    return app
