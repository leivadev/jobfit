import type { Recommendation } from '../api/client';
import { normalizeScores } from '../lib/normalizeScores';
import { RecommendationCard } from './RecommendationCard';

interface RecommendationListProps {
  recommendations: Recommendation[];
}

export function RecommendationList({ recommendations }: RecommendationListProps) {
  const normalizedScores = normalizeScores(recommendations.map((recommendation) => recommendation.rerank_score));

  return (
    <ul className="flex flex-col gap-3">
      {recommendations.map((recommendation, index) => (
        <RecommendationCard
          key={`${recommendation.company}-${recommendation.position}-${index}`}
          recommendation={recommendation}
          normalizedScore={normalizedScores[index]}
        />
      ))}
    </ul>
  );
}
