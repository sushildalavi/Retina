import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiError, getHealth, getMetadata, recommendImage, recommendProfile, recommendText } from './api';
import { ResultsGrid } from './components/ResultsGrid';
import type {
  HealthResponse,
  ImageRecommendationResponse,
  ProfileRecommendationResponse,
  ServiceMetadata,
  TextRecommendationResponse,
} from './types';

type NavId = 'overview' | 'text' | 'image' | 'profile';

const NAV_ITEMS: Array<{ id: NavId; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'text', label: 'Text Recommendations' },
  { id: 'image', label: 'Similar Images' },
  { id: 'profile', label: 'Profile Recommendations' },
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

function formatInteger(value: number) {
  return new Intl.NumberFormat('en-US').format(Math.round(value));
}

function App() {
  const [nav, setNav] = useState<NavId>('overview');
  const [service, setService] = useState<ServiceMetadata | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [startupNotice, setStartupNotice] = useState<string | null>(null);
  const [motionEnabled, setMotionEnabled] = useState(true);
  const [command, setCommand] = useState('a dog jumping through water');

  const [textQuery, setTextQuery] = useState('a dog jumping through water');
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

  const textRef = useRef<HTMLElement | null>(null);
  const imageRef = useRef<HTMLElement | null>(null);
  const profileRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const run = async () => {
      const [healthResponse, metadataResponse] = await Promise.allSettled([getHealth(), getMetadata()]);

      if (healthResponse.status === 'fulfilled') setHealth(healthResponse.value);
      if (metadataResponse.status === 'fulfilled') setService(metadataResponse.value.service);

      if (healthResponse.status === 'rejected' || metadataResponse.status === 'rejected') {
        const reason =
          healthResponse.status === 'rejected'
            ? healthResponse.reason
            : metadataResponse.status === 'rejected'
              ? metadataResponse.reason
              : null;
        const message = reason instanceof ApiError ? reason.message : 'Live backend unavailable';
        setStartupNotice(`Showing the demo shell until live backend data is available. (${message})`);
      } else {
        setStartupNotice(null);
      }
    };

    void run();
  }, []);

  const displayService = service ?? DEMO_SERVICE;
  const isLive = Boolean(service && health?.ready);

  const jumpTo = (section: NavId) => {
    setNav(section);
    const behavior = motionEnabled ? 'smooth' : 'auto';
    if (section === 'text') {
      textRef.current?.scrollIntoView({ behavior, block: 'start' });
    } else if (section === 'image') {
      imageRef.current?.scrollIntoView({ behavior, block: 'start' });
    } else if (section === 'profile') {
      profileRef.current?.scrollIntoView({ behavior, block: 'start' });
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
              <h1>Search by intent, get visually relevant results</h1>
              <p>
                Retina is a content-based visual retrieval app. It is designed for users who want fast, ranked
                results from text, image, or profile-style queries.
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
                  <dd>{formatInteger(displayService.indexed_image_count)}</dd>
                </div>
                <div>
                  <dt>Captions</dt>
                  <dd>{formatInteger(displayService.indexed_caption_count)}</dd>
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

          <section className="mode-grid">
            <article className="mode-card">
              <div className="mode-card__header">
                <div className="mode-card__icon">T</div>
                <div>
                  <h3>Text-to-image</h3>
                  <p>Type a query and retrieve the closest visual matches.</p>
                </div>
              </div>
              <button className="mode-card__cta" type="button" onClick={() => jumpTo('text')}>
                Try text search <span>→</span>
              </button>
            </article>

            <article className="mode-card">
              <div className="mode-card__header">
                <div className="mode-card__icon">◫</div>
                <div>
                  <h3>Similar images</h3>
                  <p>Use a source image to find visually similar neighbors.</p>
                </div>
              </div>
              <button className="mode-card__cta" type="button" onClick={() => jumpTo('image')}>
                Find similar <span>→</span>
              </button>
            </article>

            <article className="mode-card">
              <div className="mode-card__header">
                <div className="mode-card__icon">◔</div>
                <div>
                  <h3>Profile recommendations</h3>
                  <p>Blend interests and liked IDs into one content profile.</p>
                </div>
              </div>
              <button className="mode-card__cta" type="button" onClick={() => jumpTo('profile')}>
                Open recommendations <span>→</span>
              </button>
            </article>
          </section>

          <section className="workflow" id="text" ref={textRef}>
            <div className="workflow__header">
              <div>
                <div className="eyebrow">Text Recommendations</div>
                <h2>Search the corpus with natural language</h2>
              </div>
            </div>

            <form className="query-card" onSubmit={onTextSubmit}>
              <div className="query-field">
                <span className="query-field__icon">⌕</span>
                <input value={textQuery} onChange={(event) => setTextQuery(event.target.value)} />
                <button className="icon-action" type="submit" disabled={textLoading}>
                  {textLoading ? 'Searching…' : 'Search'}
                  <span>→</span>
                </button>
              </div>
              <div className="query-grid">
                <label>
                  <span>Top-k</span>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={textTopK}
                    onChange={(event) => setTextTopK(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Search mode</span>
                  <input value="text-to-image" readOnly />
                </label>
              </div>
            </form>

            {textError ? <div className="banner banner--error">{textError}</div> : null}
            <ResultsGrid results={textResult?.results || []} emptyLabel="Run a query to see image recommendations." />
          </section>

          <section className="workflow workflow--compact" id="image" ref={imageRef}>
            <div className="workflow__header">
              <div>
                <div className="eyebrow">Similar Images</div>
                <h2>Search from an example image</h2>
              </div>
            </div>

            <form className="compact-form" onSubmit={onImageSubmit}>
              <label>
                <span>Image ID</span>
                <input value={imageId} onChange={(event) => setImageId(event.target.value)} />
              </label>
              <label>
                <span>Top-k</span>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={imageTopK}
                  onChange={(event) => setImageTopK(Number(event.target.value))}
                />
              </label>
              <button className="primary-button" type="submit" disabled={imageLoading}>
                {imageLoading ? 'Searching…' : 'Find similar'}
              </button>
            </form>

            {imageError ? <div className="banner banner--error">{imageError}</div> : null}
            <ResultsGrid results={imageResult?.results || []} emptyLabel="Run an image lookup to see neighbors." />
          </section>

          <section className="workflow" id="profile" ref={profileRef}>
            <div className="workflow__header">
              <div>
                <div className="eyebrow">Profile Recommendations</div>
                <h2>Combine interests and liked images into one search</h2>
              </div>
            </div>

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
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={profileTopK}
                    onChange={(event) => setProfileTopK(Number(event.target.value))}
                  />
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
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
