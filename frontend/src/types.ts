export interface RecommendationResult {
  image_id: string;
  image_path: string;
  image_url?: string | null;
  captions: string[];
  caption: string;
  score: number;
  rank: number;
  recommendation_reason: string;
  index_position?: number | null;
  metadata: Record<string, unknown>;
}

export interface ServiceMetadata {
  config_path: string;
  model_name: string;
  dataset_name: string;
  indexed_image_count: number;
  indexed_caption_count: number;
  device: string;
  report_paths: Record<string, string>;
  setup_command: string;
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  checks: Record<string, boolean>;
  service: ServiceMetadata;
}

export interface MetricsSummaryResponse {
  service: ServiceMetadata;
  metrics: FinalMetricsSummary;
  summary_path: string;
}

export interface BaseRecommendationResponse {
  service: ServiceMetadata;
  top_k: number;
  latency_ms: number;
  results: RecommendationResult[];
}

export interface TextRecommendationResponse extends BaseRecommendationResponse {
  query: string;
}

export interface ImageRecommendationResponse extends BaseRecommendationResponse {
  image_id: string;
}

export interface ProfileRecommendationResponse extends BaseRecommendationResponse {
  text_queries: string[];
  liked_image_ids: string[];
}

export interface ErrorPayload {
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
  };
}

export interface FinalMetricsSummary {
  dataset: {
    name: string;
    images: number;
    captions: number;
    splits: Record<string, number>;
    sample_size: string;
    requested_sample_size: string;
    metadata_path: string;
    image_artifact_path: string;
  };
  model: {
    name: string;
    device: string;
  };
  retrieval: {
    recall_at_1: number;
    recall_at_5: number;
    recall_at_10: number;
    mrr: number;
    median_rank: number;
    map_at_10: number;
    ndcg_at_10: number;
    faiss_search_latency_p95_ms: number;
    end_to_end_text_search_latency_p95_ms: number;
  };
  recommendation: {
    text_to_image: Record<string, number>;
    profile: Record<string, number>;
    image_to_image: Record<string, number>;
  };
  random_baseline: {
    queries: number;
    candidate_images: number;
    method: string;
    recall_at_10: number;
    mrr: number;
    ndcg_at_10: number;
    median_rank: number;
  };
  runtime: {
    image_embeddings_per_sec: number;
    text_embeddings_per_sec: number;
    search_queries_per_sec: number;
    image_embedding_p95_ms: number;
    text_embedding_p95_ms: number;
    end_to_end_search_p95_ms: number;
  };
  failure_analysis: {
    failure_count: number;
    query_count: number;
    dominant_categories: string[];
    summary_path: string;
    raw_path: string;
  };
  api_command: string;
  demo_command: string;
  resume_claim: string;
  must_not_claim: string[];
  second_model_baseline: {
    status: string;
    model_name: string;
    sample_size: number;
    device: string;
    recall_at_1: number;
    recall_at_5: number;
    recall_at_10: number;
    mrr: number;
    ndcg_at_10: number;
    image_embeddings_per_sec: number;
    end_to_end_search_p95_ms: number;
  };
  scaling_experiment: Array<{
    sample_size: number | string;
    image_count: number;
    caption_count: number;
    recall_at_10: number;
    mrr: number;
    search_latency_p95_ms: number;
    image_embeddings_per_sec: number;
    total_runtime_seconds: number;
    index_size_bytes: number;
  }>;
}
