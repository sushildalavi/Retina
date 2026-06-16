from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationResult(BaseModel):
    image_id: str
    image_path: str
    image_url: Optional[str] = None
    captions: List[str] = Field(default_factory=list)
    caption: str = ""
    score: float
    rank: int
    recommendation_reason: str
    index_position: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextRecommendationRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class ImageRecommendationRequest(BaseModel):
    image_id: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class ProfileRecommendationRequest(BaseModel):
    text_queries: List[str] = Field(default_factory=list)
    liked_image_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)


class ServiceMetadata(BaseModel):
    config_path: str
    model_name: str
    dataset_name: str
    indexed_image_count: int
    indexed_caption_count: int
    device: str
    report_paths: Dict[str, str]
    setup_command: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    checks: Dict[str, bool]
    service: ServiceMetadata


class MetadataResponse(BaseModel):
    service: ServiceMetadata


class MetricsSummaryResponse(BaseModel):
    service: ServiceMetadata
    metrics: Dict[str, Any]
    summary_path: str


class RecommendationBaseResponse(BaseModel):
    service: ServiceMetadata
    top_k: int
    latency_ms: float
    results: List[RecommendationResult]


class TextRecommendationResponse(RecommendationBaseResponse):
    query: str


class ImageRecommendationResponse(RecommendationBaseResponse):
    image_id: str


class ProfileRecommendationResponse(RecommendationBaseResponse):
    text_queries: List[str] = Field(default_factory=list)
    liked_image_ids: List[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
