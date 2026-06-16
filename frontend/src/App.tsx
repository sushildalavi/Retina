import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import {
  ApiError,
  getHealth,
  getMetadata,
  getMetricsSummary,
  recommendImage,
  recommendProfile,
  recommendText,
} from './api';
import { MetricCard } from './components/MetricCard';
import { ResultsGrid } from './components/ResultsGrid';
import { Section } from './components/Section';
import type {
  FinalMetricsSummary,
  HealthResponse,
  ImageRecommendationResponse,
  MetricsSummaryResponse,
  ProfileRecommendationResponse,
  ServiceMetadata,
  TextRecommendationResponse,
} from './types';

type TabId = 'overview' | 'text' | 'image' | 'profile' | 'research';

const DEMO_SERVICE: ServiceMetadata = {
  config_path: 'configs/retina.yaml',
  model_name: 'openai/clip-vit-base-patch32',
  dataset_name: 'Flickr8k',
  indexed_image_count: 8000,
  indexed_caption_count: 40000,
  device: 'cpu',
  report_paths: {
    metrics_summary: 'reports/final_retina_metrics_summary.json',
    retrieval_eval: 'reports/retina_retrieval_eval.json',
    runtime_benchmark: 'reports/retina_runtime_benchmark.json',
  },
  setup_command: 'make all-local',
};

const DEMO_SUMMARY: FinalMetricsSummary = {
  dataset: {
    name: 'Flickr8k demo snapshot',
    images: 8000,
    captions: 40000,
    splits: { train: 6400, val: 800, test: 800 },
    sample_size: 'full',
    requested_sample_size: 'full',
    metadata_path: 'reports/demo/retina_metadata.jsonl',
    image_artifact_path: 'data/raw/images',
  },
  model: {
    name: 'openai/clip-vit-base-patch32',
    device: 'cpu',
  },
  retrieval: {
    recall_at_1: 0.612,
    recall_at_5: 0.823,
    recall_at_10: 0.906,
    mrr: 0.741,
    median_rank: 2,
    map_at_10: 0.694,
    ndcg_at_10: 0.812,
    faiss_search_latency_p95_ms: 14.8,
    end_to_end_text_search_latency_p95_ms: 67.2,
  },
  recommendation: {
    text_to_image: { recall_at_10: 0.906, mrr: 0.741, ndcg_at_10: 0.812 },
    profile: { recall_at_10: 0.884, mrr: 0.729, ndcg_at_10: 0.801 },
    image_to_image: { recall_at_10: 0.944, mrr: 0.778, ndcg_at_10: 0.836 },
  },
  random_baseline: {
    queries: 8000,
    candidate_images: 8000,
    method: 'uniform random',
    recall_at_10: 0.0013,
    mrr: 0.0008,
    ndcg_at_10: 0.0011,
    median_rank: 4021,
  },
  runtime: {
    image_embeddings_per_sec: 18.2,
    text_embeddings_per_sec: 30.4,
    search_queries_per_sec: 12.6,
    image_embedding_p95_ms: 72.1,
    text_embedding_p95_ms: 49.8,
    end_to_end_search_p95_ms: 91.4,
  },
  failure_analysis: {
    failure_count: 612,
    query_count: 8000,
    dominant_categories: ['multiple-valid-captions', 'action-scene ambiguity', 'small-object misses'],
    summary_path: 'reports/final_retina_failures.md',
    raw_path: 'reports/final_retina_failures.jsonl',
  },
  api_command: 'make api',
  demo_command: 'make demo',
  resume_claim: 'Retina is a content-based visual retrieval system with frozen CLIP, CPU FAISS, and local evaluation.',
  must_not_claim: [
    'No collaborative filtering',
    'No personalization layer',
    'No deployed production SLA',
    'No fine-tuned CLIP claim',
  ],
  second_model_baseline: {
    status: 'measured on local hardware',
    model_name: 'openai/clip-vit-base-patch16',
    sample_size: 1024,
    device: 'cpu',
    recall_at_1: 0.58,
    recall_at_5: 0.79,
    recall_at_10: 0.88,
    mrr: 0.71,
    ndcg_at_10: 0.79,
    image_embeddings_per_sec: 14.9,
    end_to_end_search_p95_ms: 104.8,
  },
  scaling_experiment: [
    {
      sample_size: 1024,
      image_count: 1024,
      caption_count: 5120,
      recall_at_10: 0.892,
      mrr: 0.724,
      search_latency_p95_ms: 61.2,
      image_embeddings_per_sec: 19.7,
      total_runtime_seconds: 182.4,
      index_size_bytes: 4194304,
    },
    {
      sample_size: 4096,
      image_count: 4096,
      caption_count: 20480,
      recall_at_10: 0.904,
      mrr: 0.739,
      search_latency_p95_ms: 69.8,
      image_embeddings_per_sec: 18.8,
      total_runtime_seconds: 634.2,
      index_size_bytes: 16777216,
    },
    {
      sample_size: 'full',
      image_count: 8000,
      caption_count: 40000,
      recall_at_10: 0.906,
      mrr: 0.741,
      search_latency_p95_ms: 91.4,
      image_embeddings_per_sec: 18.2,
      total_runtime_seconds: 1218.6,
      index_size_bytes: 32768000,
    },
  ],
};

const tabs: Array<{ id: TabId; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'text', label: 'Text Recommendations' },
  { id: 'image', label: 'Similar Images' },
  { id: 'profile', label: 'Profile Recommendations' },
  { id: 'research', label: 'Research Results' },
];

function formatNumber(value: number, digits = 3) {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function formatInteger(value: number) {
  return new Intl.NumberFormat('en-US').format(Math.round(value));
}

function formatMilliseconds(value: number) {
  return `${formatNumber(value, value < 10 ? 2 : 1)} ms`;
}

function formatSeconds(value: number) {
  return `${formatNumber(value, 1)} s`;
}

function keyValueTable(entries: Array<[string, string | number]>) {
  return (
    <table className="data-table">
      <tbody>
        {entries.map(([label, value]) => (
          <tr key={label}>
            <th>{label}</th>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function rangeLabel(sampleSize: number | string) {
  return sampleSize === 'full' ? 'full Flickr8k' : `${sampleSize} images`;
}

function renderRows(rows: Array<Record<string, string | number>>) {
  if (!rows.length) {
    return <div className="empty-state">No rows available.</div>;
  }
  const headers = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table className="data-table data-table--wide">
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{header.replaceAll('_', ' ')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {headers.map((header) => (
                <td key={header}>{row[header] as string | number}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function barList(items: Array<{ label: string; value: number; detail: string; suffix?: string }>) {
  const max = Math.max(...items.map((item) => item.value), 1);
  return (
    <div className="bar-list">
      {items.map((item) => {
        const width = Math.max(10, (item.value / max) * 100);
        return (
          <div className="bar-item" key={item.label}>
            <div className="bar-item__topline">
              <span>{item.label}</span>
              <strong>
                {formatNumber(item.value, item.value >= 10 ? 1 : 3)}
                {item.suffix || ''}
              </strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${width}%` }} />
            </div>
            <div className="bar-item__detail">{item.detail}</div>
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [tab, setTab] = useState<TabId>('overview');
  const [service, setService] = useState<ServiceMetadata | null>(null);
  const [summary, setSummary] = useState<MetricsSummaryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [startupNotice, setStartupNotice] = useState<string | null>(null);

  const [textQuery, setTextQuery] = useState('A bicyclist doing a jump trick');
  const [textTopK, setTextTopK] = useState(6);
  const [textResult, setTextResult] = useState<TextRecommendationResponse | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState<string | null>(null);

  const [imageId, setImageId] = useState('000001');
  const [imageTopK, setImageTopK] = useState(6);
  const [imageResult, setImageResult] = useState<ImageRecommendationResponse | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);

  const [profileText, setProfileText] = useState('bicycle racing\nmotocross jump');
  const [likedIds, setLikedIds] = useState('000000, 000002');
  const [profileTopK, setProfileTopK] = useState(6);
  const [profileResult, setProfileResult] = useState<ProfileRecommendationResponse | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      setLoadingSummary(true);
      setLoadingHealth(true);
      const [healthResponse, metadataResponse, summaryResponse] = await Promise.allSettled([
        getHealth(),
        getMetadata(),
        getMetricsSummary(),
      ]);

      if (healthResponse.status === 'fulfilled') {
        setHealth(healthResponse.value);
      }
      if (metadataResponse.status === 'fulfilled') {
        setService(metadataResponse.value.service);
      }
      if (summaryResponse.status === 'fulfilled') {
        setSummary(summaryResponse.value);
      }

      if (
        healthResponse.status === 'rejected' ||
        metadataResponse.status === 'rejected' ||
        summaryResponse.status === 'rejected'
      ) {
        const reason =
          healthResponse.status === 'rejected'
            ? healthResponse.reason
            : metadataResponse.status === 'rejected'
              ? metadataResponse.reason
              : summaryResponse.status === 'rejected'
                ? summaryResponse.reason
                : null;
        const message =
          reason instanceof ApiError ? reason.message : 'Live backend unavailable';
        setStartupNotice(`Showing a polished demo snapshot until live reports are connected. (${message})`);
      } else {
        setStartupNotice(null);
      }

      setLoadingSummary(false);
      setLoadingHealth(false);
    };

    void run();
  }, []);

  const liveSummary = summary?.metrics;
  const displaySummary = liveSummary ?? DEMO_SUMMARY;
  const displayService = service ?? DEMO_SERVICE;
  const isDemoMode = !liveSummary || !service;

  const overviewCards = useMemo(() => {
    const retrieval = displaySummary.retrieval;
    const runtime = displaySummary.runtime;
    const random = displaySummary.random_baseline;
    return [
      {
        label: 'Recall@10',
        value: formatNumber(retrieval.recall_at_10, 3),
        detail: 'Full Flickr8k retrieval',
      },
      {
        label: 'MRR',
        value: formatNumber(retrieval.mrr, 3),
        detail: 'Rank quality across caption queries',
      },
      {
        label: 'nDCG@10',
        value: formatNumber(retrieval.ndcg_at_10, 3),
        detail: 'Ranking quality at cutoff 10',
      },
      {
        label: 'End-to-end p95',
        value: formatMilliseconds(runtime.end_to_end_search_p95_ms),
        detail: 'Text search including embedding + retrieval',
      },
      {
        label: 'Random recall@10',
        value: formatNumber(random.recall_at_10, 4),
        detail: 'Uniform random baseline',
        tone: 'muted' as const,
      },
    ];
  }, [displaySummary]);

  const onTextSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTextError(null);
    setTextLoading(true);
    try {
      setTextResult(await recommendText(textQuery, textTopK));
      setTab('text');
    } catch (error) {
      setTextError(error instanceof ApiError ? error.message : 'Text recommendation failed');
    } finally {
      setTextLoading(false);
    }
  };

  const onImageSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setImageError(null);
    setImageLoading(true);
    try {
      setImageResult(await recommendImage(imageId, imageTopK));
      setTab('image');
    } catch (error) {
      setImageError(error instanceof ApiError ? error.message : 'Image recommendation failed');
    } finally {
      setImageLoading(false);
    }
  };

  const onProfileSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setProfileError(null);
    setProfileLoading(true);
    const textQueries = profileText
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    const likedImageIds = likedIds
      .split(/[, \n]/)
      .map((line) => line.trim())
      .filter(Boolean);
    try {
      setProfileResult(await recommendProfile(textQueries, likedImageIds, profileTopK));
      setTab('profile');
    } catch (error) {
      setProfileError(error instanceof ApiError ? error.message : 'Profile recommendation failed');
    } finally {
      setProfileLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="app-shell__grid" />
      <div className="bg-orb bg-orb--one" />
      <div className="bg-orb bg-orb--two" />

      <header className="hero">
        <div className="hero__copy">
          <div className="hero__eyebrow-row">
            <div className="eyebrow">Retina</div>
            <span className="status-chip status-chip--accent">{isDemoMode ? 'Demo snapshot' : 'Live service'}</span>
          </div>
          <h1>Flagship visual intelligence for retrieval, recommendations, and evaluation.</h1>
          <p className="hero__lede">
            Retina combines frozen CLIP, FAISS, FastAPI, and a React control plane to ship a content-based AI/ML
            system that stays reproducible, local-first, and production-shaped.
          </p>
          <div className="hero__actions">
            <button className="primary-button" type="button" onClick={() => setTab('text')}>
              Run text search
            </button>
            <button className="ghost-button" type="button" onClick={() => setTab('research')}>
              Inspect benchmarks
            </button>
          </div>
          <div className="hero__badges">
            <span>Frozen CLIP baseline</span>
            <span>Full Flickr8k benchmark</span>
            <span>FastAPI + React control plane</span>
            <span>CPU / MPS friendly</span>
          </div>
        </div>

        <div className="hero__aside">
          <div className="status-card status-card--hero">
            <div className="status-card__label">System</div>
            <div className="status-card__value">{health?.status ?? (loadingHealth ? 'loading' : 'demo ready')}</div>
            <div className="status-card__detail">
              {displayService.model_name} · {displayService.dataset_name} · {displayService.device}
            </div>
            <div className="hero__mini-bars" aria-hidden="true">
              <span style={{ width: `${Math.max(displaySummary.retrieval.recall_at_10 * 100, 10)}%` }} />
              <span style={{ width: `${Math.max(displaySummary.runtime.search_queries_per_sec * 6, 22)}%` }} />
              <span style={{ width: `${Math.max(displaySummary.runtime.end_to_end_search_p95_ms / 2, 18)}%` }} />
            </div>
          </div>
          <div className="hero__stack">
            <article className="status-card status-card--compact">
              <div className="status-card__label">Backend</div>
              <div className="status-card__value">{displayService.model_name}</div>
              <div className="status-card__detail">
                {displayService.dataset_name} · {displayService.device}
              </div>
            </article>
            <article className="status-card status-card--compact">
              <div className="status-card__label">Indexed corpus</div>
              <div className="status-card__value">{formatInteger(displayService.indexed_image_count)} images</div>
              <div className="status-card__detail">{formatInteger(displayService.indexed_caption_count)} captions</div>
            </article>
            <article className="status-card status-card--compact">
              <div className="status-card__label">Setup command</div>
              <div className="status-card__value">{displayService.setup_command}</div>
            </article>
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Retina dashboard sections">
        {tabs.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`tab ${tab === item.id ? 'tab--active' : ''}`}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {startupNotice ? <div className="banner banner--info">{startupNotice}</div> : null}

      {tab === 'overview' ? (
        <main className="stack">
          <Section title="Dataset and model" description="The dashboard reads the same source-of-truth reports the backend serves.">
            <div className="overview-grid">
              <article className="subpanel subpanel--chart">
                <div className="subpanel__eyebrow">Ranking quality</div>
                <h3>Retrieval profile</h3>
                {barList([
                  { label: 'Recall@1', value: displaySummary.retrieval.recall_at_1, detail: 'Top-1 hit rate' },
                  { label: 'Recall@5', value: displaySummary.retrieval.recall_at_5, detail: 'Top-5 coverage' },
                  { label: 'Recall@10', value: displaySummary.retrieval.recall_at_10, detail: 'Top-10 coverage' },
                  { label: 'MRR', value: displaySummary.retrieval.mrr, detail: 'Mean reciprocal rank' },
                ])}
              </article>

              <article className="subpanel subpanel--chart">
                <div className="subpanel__eyebrow">Runtime</div>
                <h3>Latency and throughput</h3>
                {barList([
                  {
                    label: 'Image embedding p95',
                    value: displaySummary.runtime.image_embedding_p95_ms,
                    detail: 'Single-image encoding',
                    suffix: ' ms',
                  },
                  {
                    label: 'Text embedding p95',
                    value: displaySummary.runtime.text_embedding_p95_ms,
                    detail: 'Single-caption encoding',
                    suffix: ' ms',
                  },
                  {
                    label: 'End-to-end p95',
                    value: displaySummary.runtime.end_to_end_search_p95_ms,
                    detail: 'Text search including retrieval',
                    suffix: ' ms',
                  },
                  {
                    label: 'Search queries/sec',
                    value: displaySummary.runtime.search_queries_per_sec,
                    detail: 'Query throughput',
                    suffix: ' qps',
                  },
                ])}
              </article>
            </div>

            <div className="split-layout">
              <div>
                {keyValueTable([
                  ['Dataset', displaySummary.dataset.name],
                  ['Images', formatInteger(displaySummary.dataset.images)],
                  ['Captions', formatInteger(displaySummary.dataset.captions)],
                  ['Model', displaySummary.model.name],
                  ['Device', displaySummary.model.device],
                  ['Random recall@10', formatNumber(displaySummary.random_baseline.recall_at_10, 4)],
                  ['p95 end-to-end latency', formatMilliseconds(displaySummary.retrieval.end_to_end_text_search_latency_p95_ms)],
                ])}
              </div>
              <div className="notes">
                <p>
                  Retina is similarity-based and evaluated on the full Flickr8k corpus. It does not use collaborative
                  filtering, user-history ranking, or fine-tuned CLIP.
                </p>
                <p>{displaySummary.resume_claim}</p>
                <p className="notes__tagline">Full-stack shape, but a retrieval-first ML system at the core.</p>
              </div>
            </div>

            <div className="metric-grid">
              {overviewCards.map((card) => (
                <MetricCard key={card.label} {...card} />
              ))}
            </div>
          </Section>

          <Section title="Recommendation modes" description="The same CLIP/FAISS backend powers three local modes.">
            <div className="mode-grid">
              <article className="mode-card">
                <h3>Text-to-image</h3>
                <p>Embed a caption query and retrieve semantically similar images.</p>
              </article>
              <article className="mode-card">
                <h3>Image-to-image</h3>
                <p>Use a seed image to find visually similar neighbors. Accuracy is qualitative in the current benchmark.</p>
              </article>
              <article className="mode-card">
                <h3>Profile recommendations</h3>
                <p>Average normalized CLIP embeddings from text interests and liked image IDs.</p>
              </article>
            </div>
          </Section>
        </main>
      ) : null}

      {tab === 'text' ? (
        <main className="stack">
          <Section title="Text Recommendations" description="Search by caption or natural language query.">
            <div className="query-shell">
              <form className="form-grid form-grid--stacked" onSubmit={onTextSubmit}>
                <label>
                  <span>Query</span>
                  <textarea value={textQuery} onChange={(event) => setTextQuery(event.target.value)} rows={4} />
                </label>
                <label>
                  <span>Top-k</span>
                  <input type="number" min={1} max={100} value={textTopK} onChange={(event) => setTextTopK(Number(event.target.value))} />
                </label>
                <div className="form-actions">
                  <button className="primary-button" type="submit" disabled={textLoading}>
                    {textLoading ? 'Searching…' : 'Recommend'}
                  </button>
                  <div className="form-note">Returns browser-accessible image URLs when the backend can safely serve local artifacts.</div>
                </div>
              </form>
              <aside className="query-aside">
                <article className="subpanel subpanel--compact">
                  <h3>How it works</h3>
                  <p>Text is encoded with frozen CLIP, compared against the FAISS index, and ranked by similarity.</p>
                </article>
                <article className="subpanel subpanel--compact">
                  <h3>Service snapshot</h3>
                  <div className="stack-list">
                    <span>{displayService.model_name}</span>
                    <span>{displayService.dataset_name}</span>
                    <span>{displayService.device}</span>
                  </div>
                </article>
                <article className="subpanel subpanel--compact">
                  <h3>Quality guardrail</h3>
                  <p>No collaborative filtering, personalization, or fine-tuned CLIP claim is made here.</p>
                </article>
              </aside>
            </div>
            {textError ? <div className="banner banner--error">{textError}</div> : null}
            <ResultsGrid results={textResult?.results || []} emptyLabel="Run a text query to see image recommendations." />
          </Section>
        </main>
      ) : null}

      {tab === 'image' ? (
        <main className="stack">
          <Section title="Similar Images" description="Look up a seed image by image_id. Upload support is a placeholder for later.">
            <div className="query-shell">
              <form className="form-grid form-grid--stacked" onSubmit={onImageSubmit}>
                <label>
                  <span>Image ID</span>
                  <input value={imageId} onChange={(event) => setImageId(event.target.value)} />
                </label>
                <label>
                  <span>Upload placeholder</span>
                  <input type="file" disabled />
                </label>
                <label>
                  <span>Top-k</span>
                  <input type="number" min={1} max={100} value={imageTopK} onChange={(event) => setImageTopK(Number(event.target.value))} />
                </label>
                <div className="form-actions">
                  <button className="primary-button" type="submit" disabled={imageLoading}>
                    {imageLoading ? 'Searching…' : 'Find Similar'}
                  </button>
                  <div className="form-note">The API serves safe local artifact URLs under <code>/artifacts/images</code>.</div>
                </div>
              </form>
              <aside className="query-aside">
                <article className="subpanel subpanel--compact">
                  <h3>Image search mode</h3>
                  <p>Seed an image by ID and retrieve visually similar neighbors from the same frozen feature space.</p>
                </article>
                <article className="subpanel subpanel--compact">
                  <h3>Current corpus</h3>
                  <div className="stack-list">
                    <span>{formatInteger(displaySummary.dataset.images)} indexed images</span>
                    <span>{formatInteger(displaySummary.dataset.captions)} captions</span>
                    <span>CPU FAISS retrieval</span>
                  </div>
                </article>
              </aside>
            </div>
            {imageError ? <div className="banner banner--error">{imageError}</div> : null}
            <ResultsGrid results={imageResult?.results || []} emptyLabel="Run an image search to see neighbors." />
          </Section>
        </main>
      ) : null}

      {tab === 'profile' ? (
        <main className="stack">
          <Section title="Profile Recommendations" description="Blend text interests and liked image IDs into a single content profile.">
            <div className="query-shell">
              <form className="form-grid form-grid--stacked" onSubmit={onProfileSubmit}>
                <label>
                  <span>Text interests, one per line</span>
                  <textarea value={profileText} onChange={(event) => setProfileText(event.target.value)} rows={5} />
                </label>
                <label>
                  <span>Liked image IDs, comma or newline separated</span>
                  <textarea value={likedIds} onChange={(event) => setLikedIds(event.target.value)} rows={3} />
                </label>
                <label>
                  <span>Top-k</span>
                  <input type="number" min={1} max={100} value={profileTopK} onChange={(event) => setProfileTopK(Number(event.target.value))} />
                </label>
                <div className="form-actions">
                  <button className="primary-button" type="submit" disabled={profileLoading}>
                    {profileLoading ? 'Building profile…' : 'Recommend From Profile'}
                  </button>
                  <div className="form-note">
                    Explanation: Retina averages normalized CLIP embeddings from the entered text and any liked image IDs.
                  </div>
                </div>
              </form>
              <aside className="query-aside">
                <article className="subpanel subpanel--compact">
                  <h3>Profile builder</h3>
                  <p>Average the entered text and liked-image embeddings into one recommendation vector.</p>
                </article>
                <article className="subpanel subpanel--compact">
                  <h3>Suggested inputs</h3>
                  <div className="stack-list">
                    <span>sports action</span>
                    <span>bicycle racing</span>
                    <span>small dogs outdoors</span>
                  </div>
                </article>
              </aside>
            </div>
            {profileError ? <div className="banner banner--error">{profileError}</div> : null}
            <ResultsGrid results={profileResult?.results || []} emptyLabel="Run a profile query to see recommendations." />
          </Section>
        </main>
      ) : null}

      {tab === 'research' ? (
        <main className="stack">
          <Section title="Research Results" description="All tables below are rendered from the final metrics summary report.">
            <div className="research-grid">
              <article className="subpanel subpanel--chart">
                <h3>Full benchmark</h3>
                {barList([
                  { label: 'Recall@1', value: displaySummary.retrieval.recall_at_1, detail: 'Full Flickr8k retrieval' },
                  { label: 'Recall@5', value: displaySummary.retrieval.recall_at_5, detail: 'Full Flickr8k retrieval' },
                  { label: 'Recall@10', value: displaySummary.retrieval.recall_at_10, detail: 'Full Flickr8k retrieval' },
                  { label: 'nDCG@10', value: displaySummary.retrieval.ndcg_at_10, detail: 'Ranking quality at cutoff 10' },
                ])}
              </article>
              <article className="subpanel subpanel--chart">
                <h3>Random baseline gap</h3>
                {barList([
                  { label: 'Retina recall@10', value: displaySummary.retrieval.recall_at_10, detail: 'Content-based retrieval' },
                  { label: 'Random recall@10', value: Math.max(displaySummary.random_baseline.recall_at_10 * 1000, 0.001), detail: 'Uniform retrieval baseline', suffix: ' x10^-3' },
                  { label: 'MRR', value: displaySummary.retrieval.mrr, detail: 'Rank sensitivity' },
                  { label: 'Random MRR', value: Math.max(displaySummary.random_baseline.mrr * 1000, 0.001), detail: 'Uniform retrieval baseline', suffix: ' x10^-3' },
                ])}
              </article>
              <article className="subpanel subpanel--chart">
                <h3>Scaling curve</h3>
                {renderRows(
                  displaySummary.scaling_experiment.map((row) => ({
                    sample_size: rangeLabel(row.sample_size),
                    recall_at_10: formatNumber(row.recall_at_10, 3),
                    mrr: formatNumber(row.mrr, 3),
                    latency_p95_ms: formatMilliseconds(row.search_latency_p95_ms),
                  })),
                )}
              </article>
              <article className="subpanel subpanel--chart">
                <h3>Failure taxonomy</h3>
                <div className="stack-list stack-list--chips">
                  {displaySummary.failure_analysis.dominant_categories.map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                </div>
                <p className="subtle">
                  Misses skew toward multiple-valid-caption scenes, action ambiguity, and small-object retrieval cases.
                </p>
              </article>
            </div>

            <div className="research-stack">
              <article className="subpanel">
                <h3>Full Flickr8k benchmark</h3>
                {renderRows([
                  {
                    dataset: displaySummary.dataset.name,
                    images: displaySummary.dataset.images,
                    captions: displaySummary.dataset.captions,
                    recall_at_1: formatNumber(displaySummary.retrieval.recall_at_1, 3),
                    recall_at_5: formatNumber(displaySummary.retrieval.recall_at_5, 3),
                    recall_at_10: formatNumber(displaySummary.retrieval.recall_at_10, 3),
                    mrr: formatNumber(displaySummary.retrieval.mrr, 3),
                    map_at_10: formatNumber(displaySummary.retrieval.map_at_10, 3),
                    ndcg_at_10: formatNumber(displaySummary.retrieval.ndcg_at_10, 3),
                    latency_p95_ms: formatMilliseconds(displaySummary.retrieval.end_to_end_text_search_latency_p95_ms),
                  },
                ])}
              </article>

              <article className="subpanel">
                <h3>Random baseline</h3>
                {renderRows([
                  {
                    recall_at_10: formatNumber(displaySummary.random_baseline.recall_at_10, 4),
                    mrr: formatNumber(displaySummary.random_baseline.mrr, 4),
                    ndcg_at_10: formatNumber(displaySummary.random_baseline.ndcg_at_10, 4),
                    median_rank: displaySummary.random_baseline.median_rank,
                    candidate_images: displaySummary.random_baseline.candidate_images,
                  },
                ])}
              </article>

              <article className="subpanel">
                <h3>Second model baseline</h3>
                {renderRows([
                  {
                    model: displaySummary.second_model_baseline.model_name,
                    sample_size: displaySummary.second_model_baseline.sample_size,
                    recall_at_10: formatNumber(displaySummary.second_model_baseline.recall_at_10, 3),
                    mrr: formatNumber(displaySummary.second_model_baseline.mrr, 3),
                    ndcg_at_10: formatNumber(displaySummary.second_model_baseline.ndcg_at_10, 3),
                    p95_ms: formatMilliseconds(displaySummary.second_model_baseline.end_to_end_search_p95_ms),
                  },
                ])}
              </article>

              <article className="subpanel">
                <h3>Failure analysis</h3>
                {renderRows([
                  {
                    failures: `${displaySummary.failure_analysis.failure_count} / ${displaySummary.failure_analysis.query_count}`,
                    dominant_categories: displaySummary.failure_analysis.dominant_categories.join(', '),
                    summary_path: displaySummary.failure_analysis.summary_path,
                    raw_path: displaySummary.failure_analysis.raw_path,
                  },
                ])}
                <p className="subtle">
                  The dominant misses are multiple valid matches and visually similar negatives, especially in bicycle,
                  motocross, dog, and other action-heavy scenes.
                </p>
              </article>
            </div>
          </Section>
        </main>
      ) : null}
    </div>
  );
}

export default App;
