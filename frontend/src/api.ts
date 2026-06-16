import type {
  ErrorPayload,
  ImageRecommendationResponse,
  MetricsSummaryResponse,
  ProfileRecommendationResponse,
  ServiceMetadata,
  TextRecommendationResponse,
  HealthResponse,
} from './types';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

export class ApiError extends Error {
  status: number;
  code: string;
  details: Record<string, unknown>;

  constructor(status: number, code: string, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(new URL(path, `${API_BASE_URL}/`).toString(), {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
    ...init,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const payload = data as Partial<ErrorPayload>;
    throw new ApiError(
      response.status,
      payload.error?.code || 'http_error',
      payload.error?.message || response.statusText,
      (payload.error?.details as Record<string, unknown>) || {},
    );
  }
  return data as T;
}

export function resolveImageUrl(imageUrl?: string | null): string | null {
  if (!imageUrl) {
    return null;
  }
  try {
    return new URL(imageUrl, `${API_BASE_URL}/`).toString();
  } catch {
    return imageUrl;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health');
}

export async function getMetadata(): Promise<{ service: ServiceMetadata }> {
  return requestJson<{ service: ServiceMetadata }>('/metadata');
}

export async function getMetricsSummary(): Promise<MetricsSummaryResponse> {
  return requestJson<MetricsSummaryResponse>('/metrics/summary');
}

export async function recommendText(query: string, topK: number): Promise<TextRecommendationResponse> {
  const url = new URL('/recommend/text', `${API_BASE_URL}/`);
  url.searchParams.set('query', query);
  url.searchParams.set('top_k', String(topK));
  return requestJson<TextRecommendationResponse>(url.pathname + url.search + url.hash);
}

export async function recommendImage(imageId: string, topK: number): Promise<ImageRecommendationResponse> {
  const url = new URL('/recommend/image', `${API_BASE_URL}/`);
  url.searchParams.set('image_id', imageId);
  url.searchParams.set('top_k', String(topK));
  return requestJson<ImageRecommendationResponse>(url.pathname + url.search + url.hash);
}

export async function recommendProfile(
  textQueries: string[],
  likedImageIds: string[],
  topK: number,
): Promise<ProfileRecommendationResponse> {
  return requestJson<ProfileRecommendationResponse>('/recommend/profile', {
    method: 'POST',
    body: JSON.stringify({
      text_queries: textQueries,
      liked_image_ids: likedImageIds,
      top_k: topK,
    }),
  });
}
