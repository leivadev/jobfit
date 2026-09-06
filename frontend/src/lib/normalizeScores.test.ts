import { describe, expect, it } from 'vitest';
import { normalizeScores } from './normalizeScores';

describe('normalizeScores', () => {
  it('normalizes a single result to a full bar', () => {
    expect(normalizeScores([0.42])).toEqual([1]);
  });

  it('min-max normalizes multiple distinct scores to 0..1', () => {
    expect(normalizeScores([0.2, 0.5, 0.9])).toEqual([0, (0.5 - 0.2) / (0.9 - 0.2), 1]);
  });

  it('gives every score a full bar when all scores are equal, without dividing by zero', () => {
    expect(normalizeScores([0.5, 0.5, 0.5])).toEqual([1, 1, 1]);
  });

  it('returns an empty array for no scores', () => {
    expect(normalizeScores([])).toEqual([]);
  });
});
