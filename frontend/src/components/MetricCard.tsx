import type { ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string;
  detail?: string;
  tone?: 'accent' | 'muted';
  icon?: ReactNode;
}

export function MetricCard({ label, value, detail, tone = 'accent', icon }: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__label">
        {icon ? <span className="metric-card__icon">{icon}</span> : null}
        <span>{label}</span>
      </div>
      <div className="metric-card__value">{value}</div>
      {detail ? <div className="metric-card__detail">{detail}</div> : null}
    </article>
  );
}
