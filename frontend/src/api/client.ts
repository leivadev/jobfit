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

/** The backend never came up within the poll budget: distinct from a normal request failure. */
export class ServerStartupTimeoutError extends Error {
  constructor() {
    super('Server did not become healthy in time');
    this.name = 'ServerStartupTimeoutError';
  }
}

/** Everything else past the client-side pre-check: timeout, other 4xx/5xx, network failure. */
export class RecommendError extends Error {
  constructor(message = 'Recommend request failed') {
    super(message);
    this.name = 'RecommendError';
  }
}

export type RecommendStage = 'checking' | 'waking' | 'processing' | 'slow';

export interface RecommendOptions {
  signal?: AbortSignal;
  onStage?: (stage: RecommendStage) => void;
}

const HEALTH_POLL_INTERVAL_MS = 2_000;
const HEALTH_ATTEMPT_TIMEOUT_MS = 5_000;
const HEALTH_MAX_WAIT_MS = 90_000;
const RECOMMEND_TIMEOUT_MS = 30_000;
const RECOMMEND_SLOW_THRESHOLD_MS = 4_000;

function withAbortTimeout(
  timeoutMs: number,
  externalSignal: AbortSignal | undefined,
): { signal: AbortSignal; cleanup: () => void } {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const onExternalAbort = () => controller.abort();
  externalSignal?.addEventListener('abort', onExternalAbort);
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timeoutId);
      externalSignal?.removeEventListener('abort', onExternalAbort);
    },
  };
}

async function pingHealth(signal: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/health`, { signal });
    return response.ok;
  } catch {
    return false;
  }
}

/** Short-interval polling, matching the backend's own Azure startup probe (infra/containerapp-probes.yaml). */
async function waitUntilHealthy(externalSignal: AbortSignal | undefined, onFirstMiss: () => void): Promise<void> {
  const deadline = Date.now() + HEALTH_MAX_WAIT_MS;
  let firstAttempt = true;

  for (;;) {
    const { signal, cleanup } = withAbortTimeout(HEALTH_ATTEMPT_TIMEOUT_MS, externalSignal);
    const healthy = await pingHealth(signal);
    cleanup();
    if (healthy) {
      return;
    }

    if (externalSignal?.aborted) {
      throw new RecommendError();
    }
    if (firstAttempt) {
      onFirstMiss();
      firstAttempt = false;
    }
    if (Date.now() >= deadline) {
      throw new ServerStartupTimeoutError();
    }
    await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_INTERVAL_MS));
  }
}

async function postRecommend(file: File, signal: AbortSignal): Promise<RecommendResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/recommend`, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (response.status === 429) {
    throw new RateLimitError();
  }
  if (!response.ok) {
    throw new RecommendError();
  }

  return (await response.json()) as RecommendResponse;
}

export async function recommend(file: File, { signal, onStage }: RecommendOptions = {}): Promise<RecommendResponse> {
  try {
    onStage?.('checking');
    await waitUntilHealthy(signal, () => onStage?.('waking'));

    onStage?.('processing');
    const { signal: recommendSignal, cleanup } = withAbortTimeout(RECOMMEND_TIMEOUT_MS, signal);
    const slowId = setTimeout(() => onStage?.('slow'), RECOMMEND_SLOW_THRESHOLD_MS);
    try {
      return await postRecommend(file, recommendSignal);
    } finally {
      clearTimeout(slowId);
      cleanup();
    }
  } catch (error) {
    if (error instanceof RateLimitError || error instanceof RecommendError || error instanceof ServerStartupTimeoutError) {
      throw error;
    }
    throw new RecommendError();
  }
}
