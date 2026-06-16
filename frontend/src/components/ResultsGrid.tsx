import type { RecommendationResult } from '../types';
import { resolveImageUrl } from '../api';

interface ResultsGridProps {
  results: RecommendationResult[];
  emptyLabel: string;
}

export function ResultsGrid({ results, emptyLabel }: ResultsGridProps) {
  if (!results.length) {
    return <div className="empty-state">{emptyLabel}</div>;
  }

  return (
    <div className="results-grid">
      {results.map((result) => {
        const imageUrl = resolveImageUrl(result.image_url);
        return (
          <article className="result-card" key={`${result.image_id}-${result.rank}`}>
            <div className="result-card__image-wrap">
              {imageUrl ? (
                <img className="result-card__image" src={imageUrl} alt={result.caption || result.image_id} loading="lazy" />
              ) : (
                <div className="result-card__image-fallback">No browser-accessible image</div>
              )}
            </div>
            <div className="result-card__body">
              <div className="result-card__topline">
                <span>#{result.rank}</span>
                <span className="result-card__score">{result.score.toFixed(3)}</span>
              </div>
              <h3>{result.caption || result.image_id}</h3>
              <p className="result-card__reason">{result.recommendation_reason}</p>
              <dl className="result-card__meta">
                <div>
                  <dt>Image</dt>
                  <dd>{result.image_id}</dd>
                </div>
                <div>
                  <dt>Path</dt>
                  <dd>{result.image_path}</dd>
                </div>
                {result.image_url ? (
                  <div>
                    <dt>URL</dt>
                    <dd>{result.image_url}</dd>
                  </div>
                ) : null}
              </dl>
            </div>
          </article>
        );
      })}
    </div>
  );
}
