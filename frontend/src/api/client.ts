export interface Recommendation {
  position: string;
  company: string;
  rerank_score: number;
  exp_years: string;
  keyword: string;
  snippet: string;
  keyword_match: boolean;
  exp_distance: number | null;
}

export interface RecommendResponse {
  results: Recommendation[];
}

/** HTTP 429: the visitor should wait and retry, not assume the app is broken. */
export class RateLimitError extends Error {
  constructor() {
    super('Rate limited');
    this.name = 'RateLimitError';
  }
}

/** Everything else past the client-side pre-check: timeout, other 4xx/5xx, network failure. */
export class RecommendError extends Error {
  constructor(message = 'Recommend request failed') {
    super(message);
    this.name = 'RecommendError';
  }
}

const TIMEOUT_MS = 30_000;

export interface RecommendOptions {
  signal?: AbortSignal;
}

export async function recommend(file: File, { signal }: RecommendOptions = {}): Promise<RecommendResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const onExternalAbort = () => controller.abort();
  signal?.addEventListener('abort', onExternalAbort);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/recommend`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (response.status === 429) {
      throw new RateLimitError();
    }
    if (!response.ok) {
      throw new RecommendError();
    }

    return (await response.json()) as RecommendResponse;
  } catch (error) {
    if (error instanceof RateLimitError || error instanceof RecommendError) {
      throw error;
    }
    throw new RecommendError();
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener('abort', onExternalAbort);
  }
}
