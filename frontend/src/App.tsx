import { useEffect, useMemo, useRef, useState } from 'react';
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
import type {
  FinalMetricsSummary,
  HealthResponse,
  ImageRecommendationResponse,
  MetricsSummaryResponse,
  ProfileRecommendationResponse,
  ServiceMetadata,
  TextRecommendationResponse,
} from './types';

type NavId = 'overview' | 'text' | 'image' | 'profile' | 'research' | 'settings';

const NAV_ITEMS: Array<{ id: NavId; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'text', label: 'Text Recommendations' },
  { id: 'image', label: 'Similar Images' },
  { id: 'profile', label: 'Profile Recommendations' },
  { id: 'research', label: 'Research Results' },
  { id: 'settings', label: 'Settings' },
];

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
    name: 'Flickr8k',
    images: 8000,
    captions: 40000,
    splits: { train: 6400, val: 800, test: 800 },
    sample_size: 'full',
    requested_sample_size: 'full',
    metadata_path: 'reports/demo/retina_metadata.jsonl',
    image_artifact_path: 'data/raw/images',
  },
  model: { name: 'openai/clip-vit-base-patch32', device: 'cpu' },
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
  must_not_claim: ['No collaborative filtering', 'No personalization layer', 'No deployed production SLA', 'No fine-tuned CLIP claim'],
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

function rangeLabel(sampleSize: number | string) {
  return sampleSize === 'full' ? 'full Flickr8k' : `${sampleSize} images`;
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

function renderRows(rows: Array<Record<string, string | number>>) {
  if (!rows.length) {
    return <div className="empty-state">No benchmark rows available.</div>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="table-wrap">
      <table className="data-table data-table--wide">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column.replace(/_/g, ' ')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column) => (
                <td key={column}>{String(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [nav, setNav] = useState<NavId>('overview');
  const [service, setService] = useState<ServiceMetadata | null>(null);
  const [summary, setSummary] = useState<MetricsSummaryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [startupNotice, setStartupNotice] = useState<string | null>(null);
  const [motionEnabled, setMotionEnabled] = useState(true);

  const [command, setCommand] = useState('a dog jumping through water');
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

  const searchRef = useRef<HTMLElement | null>(null);
  const benchmarksRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const run = async () => {
      const [healthResponse, metadataResponse, summaryResponse] = await Promise.allSettled([
        getHealth(),
        getMetadata(),
        getMetricsSummary(),
      ]);

      if (healthResponse.status === 'fulfilled') setHealth(healthResponse.value);
      if (metadataResponse.status === 'fulfilled') setService(metadataResponse.value.service);
      if (summaryResponse.status === 'fulfilled') setSummary(summaryResponse.value);

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
        const message = reason instanceof ApiError ? reason.message : 'Live backend unavailable';
        setStartupNotice(`Showing the demo shell until live backend data is available. (${message})`);
      } else {
        setStartupNotice(null);
      }
    };

    void run();
  }, []);

  const displaySummary = summary?.metrics ?? DEMO_SUMMARY;
  const displayService = service ?? DEMO_SERVICE;
  const isLive = Boolean(summary?.metrics && service && health?.ready);

  const featureCards = useMemo(() => {
    const retrieval = displaySummary.retrieval;
    return [
      { label: 'Recall@10', value: formatNumber(retrieval.recall_at_10, 3), detail: 'Full Flickr8k retrieval' },
      { label: 'MRR', value: formatNumber(retrieval.mrr, 3), detail: 'Rank quality across caption queries' },
      { label: 'nDCG@10', value: formatNumber(retrieval.ndcg_at_10, 3), detail: 'Ranking quality at cutoff 10' },
      { label: 'End-to-end p95', value: formatMilliseconds(retrieval.end_to_end_text_search_latency_p95_ms), detail: 'Text search latency' },
      { label: 'Random recall@10', value: formatNumber(displaySummary.random_baseline.recall_at_10, 4), detail: 'Uniform random baseline', tone: 'muted' as const },
      { label: 'Profile recall@10', value: formatNumber(displaySummary.recommendation.profile.recall_at_10, 3), detail: 'Blended content profiles' },
    ];
  }, [displaySummary]);

  const jumpTo = (section: NavId) => {
    setNav(section);
    if (section === 'search') {
      searchRef.current?.scrollIntoView({ behavior: motionEnabled ? 'smooth' : 'auto', block: 'start' });
    } else if (section === 'research') {
      benchmarksRef.current?.scrollIntoView({ behavior: motionEnabled ? 'smooth' : 'auto', block: 'start' });
    }
  };

  const onTextSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTextError(null);
    setTextLoading(true);
    try {
      setTextResult(await recommendText(textQuery, textTopK));
      setNav('text');
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
      setNav('image');
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
      setNav('profile');
    } catch (error) {
      setProfileError(error instanceof ApiError ? error.message : 'Profile recommendation failed');
    } finally {
      setProfileLoading(false);
    }
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand__mark">Retina</div>
        </div>
        <nav className="nav" aria-label="Retina sections">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav__item ${nav === item.id ? 'nav__item--active' : ''}`}
              onClick={() => jumpTo(item.id)}
            >
              <span className="nav__dot" />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="status-panel">
            <div className="status-panel__label">System status</div>
            <div className="status-panel__value">{isLive ? 'Connected' : 'Demo ready'}</div>
            <div className="status-panel__detail">
              {displayService.model_name}
              <br />
              {displayService.device.toUpperCase()}
            </div>
          </div>
          <button className="secondary-button" type="button" onClick={() => jumpTo('overview')}>
            Collapse
          </button>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <form
            className="omnibox"
            onSubmit={(event) => {
              event.preventDefault();
              setTextQuery(command);
              jumpTo('text');
            }}
          >
            <span className="omnibox__icon">⌕</span>
            <input
              value={command}
              onChange={(event) => setCommand(event.target.value)}
              placeholder="Search queries, images, or commands..."
            />
            <kbd>⌘ K</kbd>
          </form>

          <div className="topbar__actions">
            <button
              className={`motion-toggle ${motionEnabled ? 'motion-toggle--on' : ''}`}
              type="button"
              onClick={() => setMotionEnabled((value) => !value)}
            >
              <span>Motion</span>
              <span className="motion-toggle__switch" />
            </button>
            <button className="icon-button" type="button" aria-label="Theme toggle">
              ☼
            </button>
            <button className="user-chip" type="button" aria-label="User menu">
              <span className="user-chip__title">Researcher</span>
              <span className="user-chip__avatar">R</span>
              <span className="user-chip__caret">⌄</span>
            </button>
          </div>
        </header>

        <main className={`page ${motionEnabled ? 'page--motion' : 'page--static'}`}>
          <section className="hero" id="overview">
            <div className="hero__copy">
              <div className="eyebrow">Overview</div>
              <h1>Visual intelligence, powered by recommendations</h1>
              <p>
                Retina is a visual intelligence engine that understands text, images, and profiles to surface the most
                relevant results fast.
              </p>
            </div>

            <aside className="hero__sidecard">
              <div className="card-head">
                <div>
                  <div className="card-head__eyebrow">Corpus</div>
                  <h2>Flickr8k full</h2>
                </div>
                <span className="badge badge--good">Indexed</span>
              </div>
              <dl className="stats-list">
                <div>
                  <dt>Images</dt>
                  <dd>{formatInteger(displaySummary.dataset.images)}</dd>
                </div>
                <div>
                  <dt>Captions</dt>
                  <dd>{formatInteger(displaySummary.dataset.captions)}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{displayService.model_name}</dd>
                </div>
                <div>
                  <dt>Device</dt>
                  <dd>{displayService.device.toUpperCase()}</dd>
                </div>
              </dl>
            </aside>
          </section>

          {startupNotice ? <div className="banner banner--info">{startupNotice}</div> : null}

          <section className="metrics">
            {featureCards.map((card) => (
              <MetricCard key={card.label} {...card} />
            ))}
          </section>

          <section className="quick-actions">
            <div className="quick-actions__label">Quick actions</div>
            <button className="quick-action" type="button" onClick={() => jumpTo('text')}>
              <span className="quick-action__icon">⌕</span>
              <span>
                <strong>Try a query</strong>
                <small>Search text or image</small>
              </span>
              <span className="quick-action__arrow">→</span>
            </button>
            <button className="quick-action" type="button" onClick={() => jumpTo('research')}>
              <span className="quick-action__icon">▣</span>
              <span>
                <strong>Open benchmarks</strong>
                <small>View latest results</small>
              </span>
              <span className="quick-action__arrow">→</span>
            </button>
            <button className="quick-action" type="button" onClick={() => jumpTo('settings')}>
              <span className="quick-action__icon">☰</span>
              <span>
                <strong>View model tradeoffs</strong>
                <small>Compare settings</small>
              </span>
              <span className="quick-action__arrow">→</span>
            </button>
          </section>

          <section className="mode-grid" id="search" ref={searchRef}>
            <article className="mode-card mode-card--text">
              <div className="mode-card__header">
                <div className="mode-card__icon">T</div>
                <div>
                  <h3>Text-to-image</h3>
                  <p>Describe what you are looking for and find the best matching images.</p>
                </div>
              </div>
              <button className="mode-card__cta" type="button" onClick={() => setNav('text')}>
                Try text search <span>→</span>
              </button>
            </article>

            <article className="mode-card mode-card--image">
              <div className="mode-card__header">
                <div className="mode-card__icon">◫</div>
                <div>
                  <h3>Similar Images</h3>
                  <p>Find visually similar images using an example image.</p>
                </div>
              </div>
              <button className="mode-card__cta" type="button" onClick={() => setNav('image')}>
                Find similar <span>→</span>
              </button>
            </article>

            <article className="mode-card mode-card--profile">
              <div className="mode-card__header">
                <div className="mode-card__icon">◔</div>
                <div>
                  <h3>Profile Recommendations</h3>
                  <p>Get recommendations based on interests and history.</p>
                </div>
              </div>
              <button className="mode-card__cta" type="button" onClick={() => setNav('profile')}>
                Open recommendations <span>→</span>
              </button>
            </article>
          </section>

          <section className="benchmarks" id="research" ref={benchmarksRef}>
            <div className="section-head">
              <div>
                <div className="eyebrow">Research Results</div>
                <h2>Measured retrieval performance, tradeoffs, and failure modes</h2>
                <p>Empirical results on Flickr8k with ablation experiments.</p>
              </div>
            </div>

            <div className="benchmarks__grid">
              <article className="bench-card bench-card--table">
                <div className="card-head">
                  <div>
                    <div className="card-head__eyebrow">Benchmark results</div>
                    <h3>Flickr8k full</h3>
                  </div>
                  <span className="badge badge--quiet">Indexed</span>
                </div>
                {renderRows([
                  {
                    images: displaySummary.dataset.images,
                    captions: displaySummary.dataset.captions,
                    recall_at_10: formatNumber(displaySummary.retrieval.recall_at_10, 3),
                    mrr: formatNumber(displaySummary.retrieval.mrr, 3),
                    ndcg_at_10: formatNumber(displaySummary.retrieval.ndcg_at_10, 3),
                    p95_latency: formatMilliseconds(displaySummary.retrieval.end_to_end_text_search_latency_p95_ms),
                  },
                ])}
              </article>

              <article className="bench-card bench-card--tradeoff">
                <div className="card-head">
                  <div>
                    <div className="card-head__eyebrow">Model tradeoff</div>
                    <h3>Quality vs speed</h3>
                  </div>
                  <button className="text-button" type="button" onClick={() => setNav('settings')}>
                    View details →
                  </button>
                </div>
                {barList([
                  {
                    label: 'Recall@10',
                    value: displaySummary.retrieval.recall_at_10,
                    detail: 'Full Flickr8k retrieval',
                  },
                  {
                    label: 'MRR',
                    value: displaySummary.retrieval.mrr,
                    detail: 'Caption ranking quality',
                  },
                  {
                    label: 'nDCG@10',
                    value: displaySummary.retrieval.ndcg_at_10,
                    detail: 'Top-k ordering gain',
                  },
                  {
                    label: 'p95 latency',
                    value: displaySummary.retrieval.end_to_end_text_search_latency_p95_ms,
                    detail: 'End-to-end search latency',
                    suffix: ' ms',
                  },
                ])}
              </article>

              <article className="bench-card bench-card--scale">
                <div className="card-head">
                  <div>
                    <div className="card-head__eyebrow">Scaling experiment</div>
                    <h3>ViT-B/32</h3>
                  </div>
                </div>
                {renderRows(
                  displaySummary.scaling_experiment.map((row) => ({
                    sample_size: rangeLabel(row.sample_size),
                    recall_at_10: formatNumber(row.recall_at_10, 3),
                    mrr: formatNumber(row.mrr, 3),
                    p95_latency: formatMilliseconds(row.search_latency_p95_ms),
                  })),
                )}
              </article>

              <article className="bench-card bench-card--failure">
                <div className="card-head">
                  <div>
                    <div className="card-head__eyebrow">Failure analysis</div>
                    <h3>{formatInteger(displaySummary.failure_analysis.failure_count)} / {formatInteger(displaySummary.failure_analysis.query_count)} misses</h3>
                  </div>
                  <button className="text-button" type="button">
                    View all cases →
                  </button>
                </div>
                <div className="failure-bars">
                  {displaySummary.failure_analysis.dominant_categories.map((item, index) => (
                    <div className="failure-bar" key={item}>
                      <div className="failure-bar__label">{item}</div>
                      <div className="failure-bar__track">
                        <div className="failure-bar__fill" style={{ width: `${72 - index * 18}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="chips">
                  {['people', 'animals', 'vehicles', 'indoor', 'outdoor'].map((item) => (
                    <span key={item} className="chip">
                      {item}
                    </span>
                  ))}
                </div>
              </article>

              <article className="bench-card bench-card--recommendations">
                <div className="card-head">
                  <div>
                    <div className="card-head__eyebrow">Recommendations</div>
                    <h3>Core modes</h3>
                  </div>
                </div>
                <div className="recommendation-links">
                  <button className="recommendation-link" type="button" onClick={() => setNav('text')}>
                    <span>
                      <strong>Text-to-image</strong>
                      <small>Describe what you are looking for</small>
                    </span>
                    <span>→</span>
                  </button>
                  <button className="recommendation-link" type="button" onClick={() => setNav('image')}>
                    <span>
                      <strong>Similar images</strong>
                      <small>Find visually similar images</small>
                    </span>
                    <span>→</span>
                  </button>
                  <button className="recommendation-link" type="button" onClick={() => setNav('profile')}>
                    <span>
                      <strong>Profile recommendations</strong>
                      <small>Personalized content blends</small>
                    </span>
                    <span>→</span>
                  </button>
                </div>
              </article>

              <article className="bench-card bench-card--claim">
                <div className="card-head">
                  <div>
                    <div className="card-head__eyebrow">What not to claim</div>
                    <h3>Keep the framing honest</h3>
                  </div>
                </div>
                <div className="chips">
                  {displaySummary.must_not_claim.map((item) => (
                    <span key={item} className="chip chip--muted">
                      {item}
                    </span>
                  ))}
                </div>
              </article>
            </div>
          </section>

          <section className="workflow" id="text">
            <div className="workflow__header">
              <div>
                <div className="eyebrow">Text Recommendations</div>
                <h2>Search by intent, get visually relevant results</h2>
              </div>
              <aside className="insight-card">
                <div className="insight-card__title">Query insights</div>
                <dl>
                  <div>
                    <dt>Mode</dt>
                    <dd>text-to-image</dd>
                  </div>
                  <div>
                    <dt>Latency (p95)</dt>
                    <dd>{formatMilliseconds(displaySummary.retrieval.end_to_end_text_search_latency_p95_ms)}</dd>
                  </div>
                  <div>
                    <dt>Retrieved from</dt>
                    <dd>{formatInteger(displaySummary.dataset.images)} images / {formatInteger(displaySummary.dataset.captions)} captions</dd>
                  </div>
                  <div>
                    <dt>Alignment</dt>
                    <dd>Strong semantic alignment</dd>
                  </div>
                </dl>
              </aside>
            </div>

            <div className="workflow__body">
              <div className="query-shell">
                <form className="query-card" onSubmit={onTextSubmit}>
                  <div className="query-field">
                    <span className="query-field__icon">⌕</span>
                    <input value={textQuery} onChange={(event) => setTextQuery(event.target.value)} />
                    <button className="icon-action" type="submit" disabled={textLoading}>
                      Search
                      <span>→</span>
                    </button>
                  </div>
                  <div className="filter-row">
                    <button type="button" className="filter-pill">
                      Top-k {textTopK}
                    </button>
                    <button type="button" className="filter-pill">
                      Similarity
                    </button>
                    <button type="button" className="filter-pill">
                      Flickr8k
                    </button>
                    <button type="button" className="filter-pill">
                      CLIP ViT-B/32
                    </button>
                    <button type="button" className="filter-link">
                      Clear all
                    </button>
                  </div>
                  <div className="query-grid">
                    <label>
                      <span>Top-k</span>
                      <input type="number" min={1} max={100} value={textTopK} onChange={(event) => setTextTopK(Number(event.target.value))} />
                    </label>
                    <label>
                      <span>Query notes</span>
                      <input value="Semantic match" readOnly />
                    </label>
                  </div>
                  <div className="query-note">Showing {textResult?.results.length || 6} results ranked by similarity.</div>
                </form>

                <div className="query-side">
                  <article className="why-card">
                    <div className="why-card__title">Why these results</div>
                    <div className="why-card__items">
                      {['Semantic match', 'Visual alignment', 'Context similarity', 'Ranking signals'].map((item) => (
                        <div className="why-item" key={item}>
                          <div className="why-item__dot" />
                          <div>
                            <strong>{item}</strong>
                            <p>
                              {item === 'Semantic match'
                                ? 'Captions explicitly mention the query concept.'
                                : item === 'Visual alignment'
                                  ? 'Motion, composition, and scene structure line up.'
                                  : item === 'Context similarity'
                                    ? 'Scene, setting, and objects align with intent.'
                                    : 'CLIP similarity and caption relevance balance ranking.'}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>

                  <article className="why-card why-card--warning">
                    <div className="why-card__title">Failure watchlist</div>
                    <div className="why-card__items">
                      {['Visually similar negatives', 'Multiple valid captions'].map((item) => (
                        <div className="why-item" key={item}>
                          <div className="why-item__dot" />
                          <div>
                            <strong>{item}</strong>
                            <p>
                              {item === 'Visually similar negatives'
                                ? 'Background or outdoor scenes may look similar.'
                                : 'One image can match many captions; ranking balances relevance and diversity.'}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>
                </div>
              </div>

              {textError ? <div className="banner banner--error">{textError}</div> : null}
              <ResultsGrid results={textResult?.results || []} emptyLabel="Run a query to see image recommendations." />
            </div>
          </section>

          <section className="workflow workflow--compact" id="image">
            <div className="workflow__header">
              <div>
                <div className="eyebrow">Similar Images</div>
                <h2>Find visually similar images from an example</h2>
              </div>
            </div>
            <div className="workflow__body workflow__body--stacked">
              <form className="compact-form" onSubmit={onImageSubmit}>
                <label>
                  <span>Image ID</span>
                  <input value={imageId} onChange={(event) => setImageId(event.target.value)} />
                </label>
                <label>
                  <span>Top-k</span>
                  <input type="number" min={1} max={100} value={imageTopK} onChange={(event) => setImageTopK(Number(event.target.value))} />
                </label>
                <button className="primary-button" type="submit" disabled={imageLoading}>
                  {imageLoading ? 'Searching…' : 'Find similar'}
                </button>
              </form>
              {imageError ? <div className="banner banner--error">{imageError}</div> : null}
              <ResultsGrid results={imageResult?.results || []} emptyLabel="Run an image lookup to see neighbors." />
            </div>
          </section>

          <section className="workflow" id="profile">
            <div className="workflow__header">
              <div>
                <div className="eyebrow">Profile Recommendations</div>
                <h2>Blend interests and history into one content profile</h2>
              </div>
            </div>
            <div className="workflow__body">
              <div className="query-shell query-shell--profile">
                <form className="query-card" onSubmit={onProfileSubmit}>
                  <label>
                    <span>Text interests, one per line</span>
                    <textarea value={profileText} onChange={(event) => setProfileText(event.target.value)} rows={5} />
                  </label>
                  <label>
                    <span>Liked image IDs</span>
                    <textarea value={likedIds} onChange={(event) => setLikedIds(event.target.value)} rows={3} />
                  </label>
                  <label>
                    <span>Top-k</span>
                    <input type="number" min={1} max={100} value={profileTopK} onChange={(event) => setProfileTopK(Number(event.target.value))} />
                  </label>
                  <button className="primary-button" type="submit" disabled={profileLoading}>
                    {profileLoading ? 'Building profile…' : 'Recommend'}
                  </button>
                </form>
                <aside className="query-side">
                  <article className="why-card">
                    <div className="why-card__title">Profile builder</div>
                    <div className="why-item">
                      <div className="why-item__dot" />
                      <div>
                        <strong>Vector blend</strong>
                        <p>Average the entered text and liked-image embeddings into one recommendation vector.</p>
                      </div>
                    </div>
                  </article>
                </aside>
              </div>
              {profileError ? <div className="banner banner--error">{profileError}</div> : null}
              <ResultsGrid results={profileResult?.results || []} emptyLabel="Run a profile query to see recommendations." />
            </div>
          </section>

          <section className="settings" id="settings">
            <div className="settings__header">
              <div>
                <div className="eyebrow">Settings</div>
                <h2>Model and claim guardrails</h2>
              </div>
            </div>
            <div className="settings__grid">
              <article className="setting-card">
                <div className="setting-card__label">Model</div>
                <div className="setting-card__value">{displayService.model_name}</div>
                <div className="setting-card__meta">{displayService.device.toUpperCase()}</div>
              </article>
              <article className="setting-card">
                <div className="setting-card__label">Corpus</div>
                <div className="setting-card__value">{displaySummary.dataset.name}</div>
                <div className="setting-card__meta">{formatInteger(displaySummary.dataset.images)} images</div>
              </article>
              <article className="setting-card">
                <div className="setting-card__label">Do not claim</div>
                <div className="chips">
                  {displaySummary.must_not_claim.map((item) => (
                    <span key={item} className="chip chip--muted">
                      {item}
                    </span>
                  ))}
                </div>
              </article>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
