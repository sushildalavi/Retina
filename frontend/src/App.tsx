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

function App() {
  const [tab, setTab] = useState<TabId>('overview');
  const [service, setService] = useState<ServiceMetadata | null>(null);
  const [summary, setSummary] = useState<MetricsSummaryResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [startupError, setStartupError] = useState<string | null>(null);

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
      try {
        const [healthResponse, metadataResponse, summaryResponse] = await Promise.all([
          getHealth(),
          getMetadata(),
          getMetricsSummary(),
        ]);
        setHealth(healthResponse);
        setService(metadataResponse.service);
        setSummary(summaryResponse);
      } catch (error) {
        const message = error instanceof ApiError ? `${error.code}: ${error.message}` : 'Failed to load backend summary';
        setStartupError(message);
      } finally {
        setLoadingSummary(false);
        setLoadingHealth(false);
      }
    };

    void run();
  }, []);

  const overviewCards = useMemo(() => {
    const retrieval = summary?.metrics.retrieval;
    const runtime = summary?.metrics.runtime;
    const random = summary?.metrics.random_baseline;
    if (!retrieval || !runtime || !random) {
      return [];
    }
    const randomBoost = random.recall_at_10 > 0 ? retrieval.recall_at_10 / random.recall_at_10 : 0;
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
        value: `${formatNumber(random.recall_at_10, 4)} (${formatNumber(randomBoost, 0)}x gap)`,
        detail: 'Baseline comparator',
        tone: 'muted' as const,
      },
    ];
  }, [summary]);

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
      .split(/[,\n]/)
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

  const summaryMetrics = summary?.metrics;

  return (
    <div className="app-shell">
      <div className="bg-orb bg-orb--one" />
      <div className="bg-orb bg-orb--two" />

      <header className="hero">
        <div className="hero__copy">
          <div className="eyebrow">Retina</div>
          <h1>Visual Intelligence Recommendation Engine</h1>
          <p className="hero__lede">
            A Mac-local content-based visual recommendation system using CLIP embeddings, FAISS indexing, FastAPI, and
            React/Gradio interfaces.
          </p>
          <div className="hero__badges">
            <span>Content-based, not collaborative filtering</span>
            <span>Full Flickr8k benchmark</span>
            <span>Local backend, local artifacts</span>
          </div>
        </div>

        <div className="hero__aside">
          <div className="status-card">
            <div className="status-card__label">Backend</div>
            <div className="status-card__value">{health?.status ?? (loadingHealth ? 'loading' : 'unknown')}</div>
            <div className="status-card__detail">
              {service ? `${service.model_name} · ${service.dataset_name} · ${service.device}` : 'Waiting for metadata'}
            </div>
          </div>
          <div className="status-card status-card--compact">
            <div className="status-card__label">Setup command</div>
            <div className="status-card__value">{service?.setup_command ?? 'make all-local'}</div>
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

      {startupError ? <div className="banner banner--error">{startupError}</div> : null}

      {tab === 'overview' ? (
        <main className="stack">
          <Section title="Dataset and model" description="The dashboard reads the same source-of-truth reports the backend serves.">
            {loadingSummary || !summaryMetrics ? (
              <div className="empty-state">Loading summary reports...</div>
            ) : (
              <div className="split-layout">
                <div>
                  {keyValueTable([
                    ['Dataset', summaryMetrics.dataset.name],
                    ['Images', formatInteger(summaryMetrics.dataset.images)],
                    ['Captions', formatInteger(summaryMetrics.dataset.captions)],
                    ['Model', summaryMetrics.model.name],
                    ['Device', summaryMetrics.model.device],
                    ['Random recall@10', formatNumber(summaryMetrics.random_baseline.recall_at_10, 4)],
                    ['p95 end-to-end latency', formatMilliseconds(summaryMetrics.runtime.end_to_end_search_p95_ms)],
                  ])}
                </div>
                <div className="notes">
                  <p>
                    Retina is similarity-based and evaluated on the full Flickr8k corpus. It does not use collaborative
                    filtering, user-history ranking, or fine-tuned CLIP.
                  </p>
                  <p>{summaryMetrics.resume_claim}</p>
                </div>
              </div>
            )}
          </Section>

          <div className="metric-grid">
            {overviewCards.map((card) => (
              <MetricCard key={card.label} {...card} />
            ))}
          </div>

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
            <form className="form-grid" onSubmit={onTextSubmit}>
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
            {textError ? <div className="banner banner--error">{textError}</div> : null}
            <ResultsGrid results={textResult?.results || []} emptyLabel="Run a text query to see image recommendations." />
          </Section>
        </main>
      ) : null}

      {tab === 'image' ? (
        <main className="stack">
          <Section title="Similar Images" description="Look up a seed image by image_id. Upload support is a placeholder for later.">
            <form className="form-grid" onSubmit={onImageSubmit}>
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
            {imageError ? <div className="banner banner--error">{imageError}</div> : null}
            <ResultsGrid results={imageResult?.results || []} emptyLabel="Run an image search to see neighbors." />
          </Section>
        </main>
      ) : null}

      {tab === 'profile' ? (
        <main className="stack">
          <Section title="Profile Recommendations" description="Blend text interests and liked image IDs into a single content profile.">
            <form className="form-grid" onSubmit={onProfileSubmit}>
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
            {profileError ? <div className="banner banner--error">{profileError}</div> : null}
            <ResultsGrid results={profileResult?.results || []} emptyLabel="Run a profile query to see recommendations." />
          </Section>
        </main>
      ) : null}

      {tab === 'research' ? (
        <main className="stack">
          <Section title="Research Results" description="All tables below are rendered from the final metrics summary report.">
            {!summaryMetrics ? (
              <div className="empty-state">Loading summary reports...</div>
            ) : (
              <div className="research-stack">
                <article className="subpanel">
                  <h3>Full Flickr8k benchmark</h3>
                  {renderRows([
                    {
                      dataset: summaryMetrics.dataset.name,
                      images: summaryMetrics.dataset.images,
                      captions: summaryMetrics.dataset.captions,
                      recall_at_1: formatNumber(summaryMetrics.retrieval.recall_at_1, 3),
                      recall_at_5: formatNumber(summaryMetrics.retrieval.recall_at_5, 3),
                      recall_at_10: formatNumber(summaryMetrics.retrieval.recall_at_10, 3),
                      mrr: formatNumber(summaryMetrics.retrieval.mrr, 3),
                      map_at_10: formatNumber(summaryMetrics.retrieval.map_at_10, 3),
                      ndcg_at_10: formatNumber(summaryMetrics.retrieval.ndcg_at_10, 3),
                      latency_p95_ms: formatMilliseconds(summaryMetrics.retrieval.end_to_end_text_search_latency_p95_ms),
                    },
                  ])}
                </article>

                <article className="subpanel">
                  <h3>Random baseline</h3>
                  {renderRows([
                    {
                      recall_at_10: formatNumber(summaryMetrics.random_baseline.recall_at_10, 4),
                      mrr: formatNumber(summaryMetrics.random_baseline.mrr, 4),
                      ndcg_at_10: formatNumber(summaryMetrics.random_baseline.ndcg_at_10, 4),
                      median_rank: summaryMetrics.random_baseline.median_rank,
                      candidate_images: summaryMetrics.random_baseline.candidate_images,
                    },
                  ])}
                </article>

                <article className="subpanel">
                  <h3>Second model baseline</h3>
                  {renderRows([
                    {
                      model: summaryMetrics.second_model_baseline.model_name,
                      sample_size: summaryMetrics.second_model_baseline.sample_size,
                      recall_at_10: formatNumber(summaryMetrics.second_model_baseline.recall_at_10, 3),
                      mrr: formatNumber(summaryMetrics.second_model_baseline.mrr, 3),
                      ndcg_at_10: formatNumber(summaryMetrics.second_model_baseline.ndcg_at_10, 3),
                      p95_ms: formatMilliseconds(summaryMetrics.second_model_baseline.end_to_end_search_p95_ms),
                    },
                  ])}
                </article>

                <article className="subpanel">
                  <h3>Scaling experiment</h3>
                  {renderRows(
                    summaryMetrics.scaling_experiment.map((row) => ({
                      sample_size: rangeLabel(row.sample_size),
                      images: row.image_count,
                      captions: row.caption_count,
                      recall_at_10: formatNumber(row.recall_at_10, 3),
                      mrr: formatNumber(row.mrr, 3),
                      latency_p95_ms: formatMilliseconds(row.search_latency_p95_ms),
                      embeddings_per_sec: formatNumber(row.image_embeddings_per_sec, 2),
                      runtime_s: formatSeconds(row.total_runtime_seconds),
                      index_bytes: formatInteger(row.index_size_bytes),
                    })),
                  )}
                </article>

                <article className="subpanel">
                  <h3>Failure analysis</h3>
                  {renderRows([
                    {
                      failures: `${summaryMetrics.failure_analysis.failure_count} / ${summaryMetrics.failure_analysis.query_count}`,
                      dominant_categories: summaryMetrics.failure_analysis.dominant_categories.join(', '),
                      summary_path: summaryMetrics.failure_analysis.summary_path,
                      raw_path: summaryMetrics.failure_analysis.raw_path,
                    },
                  ])}
                  <p className="subtle">
                    The dominant misses are multiple valid matches and visually similar negatives, especially in bicycle,
                    motocross, dog, and other action-heavy scenes.
                  </p>
                </article>
              </div>
            )}
          </Section>
        </main>
      ) : null}
    </div>
  );
}

export default App;
