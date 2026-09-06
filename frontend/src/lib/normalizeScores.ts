/**
 * Min-max normalizes rerank_score (an unbounded Cross-encoder logit) to 0..1
 * within a single response only. When every score is equal there's no spread
 * to normalize against, so every entry gets a full bar rather than dividing
 * by zero.
 */
export function normalizeScores(scores: number[]): number[] {
  if (scores.length === 0) {
    return [];
  }

  const min = Math.min(...scores);
  const max = Math.max(...scores);

  if (min === max) {
    return scores.map(() => 1);
  }

  return scores.map((score) => (score - min) / (max - min));
}
