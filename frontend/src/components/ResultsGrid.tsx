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
                <div className="result-card__image-fallback">
                  <span>Preview unavailable</span>
                  <strong>{result.caption || result.image_id}</strong>
                </div>
              )}
            </div>
            <div className="result-card__body">
              <div className="result-card__topline">
                <span>#{result.rank}</span>
                <span className="result-card__score">{result.score.toFixed(3)}</span>
              </div>
              <h3>{result.caption || result.image_id}</h3>
              <p className="result-card__reason">{result.recommendation_reason}</p>
              <div className="result-card__meta">
                <span className="chip">Image {result.image_id}</span>
                {result.captions?.[0] ? <span className="chip chip--muted">{result.captions[0]}</span> : null}
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}
