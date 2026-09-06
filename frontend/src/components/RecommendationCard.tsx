import type { Recommendation } from '../api/client';

interface RecommendationCardProps {
  recommendation: Recommendation;
  normalizedScore: number;
}

export function RecommendationCard({ recommendation, normalizedScore }: RecommendationCardProps) {
  const percent = Math.round(normalizedScore * 100);

  return (
    <li className="rounded-md border border-gray-200 p-4">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="font-medium text-gray-900">{recommendation.position}</h3>
        <span className="text-sm text-gray-500">{recommendation.company}</span>
      </div>
      <p className="mt-1 text-sm text-gray-600">{recommendation.snippet}</p>
      <div className="mt-3 flex items-center gap-2">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100">
          <div className="h-full rounded-full bg-purple-600" style={{ width: `${percent}%` }} />
        </div>
        <span className="text-xs font-medium text-gray-600">{percent}%</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <span className="rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-800">
          {recommendation.keyword}
        </span>
        <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
          {recommendation.exp_years}
        </span>
      </div>
    </li>
  );
}
