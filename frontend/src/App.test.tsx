import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import {
  RateLimitError,
  RecommendError,
  ServerStartupTimeoutError,
  type Recommendation,
  type RecommendResponse,
} from './api/client';

function makeRecommendation(overrides: Partial<Recommendation> = {}): Recommendation {
  return {
    position: 'Backend Developer (Python)',
    company: 'Acme Corp',
    rerank_score: 0.83,
    exp_years: '2y',
    keyword: 'Python',
    snippet: 'Build and maintain backend services.',
    keyword_match: false,
    exp_distance: null,
    ...overrides,
  };
}

function makeCvFile(name = 'cv.txt', type = 'text/plain', size = 1024) {
  const file = new File([new Uint8Array(size)], name, { type });
  return file;
}

async function selectFile(file: File) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  // applyAccept: false — a real file picker's `accept` filter can be bypassed
  // (e.g. "All Files"), so the client-side pre-check must be the real gate.
  const user = userEvent.setup({ applyAccept: false });
  await user.upload(input, file);
  return user;
}

describe('App happy path', () => {
  it('selects a file, submits, shows loading, then renders ranked results', async () => {
    let resolveRecommend: (response: RecommendResponse) => void = () => {};
    const fakeRecommend = vi.fn(
      () =>
        new Promise<RecommendResponse>((resolve) => {
          resolveRecommend = resolve;
        }),
    );

    render(<App recommend={fakeRecommend} />);

    const user = await selectFile(makeCvFile());
    expect(screen.getByText('cv.txt')).toBeInTheDocument();

    const submitButton = screen.getByRole('button', { name: /recommend jobs/i });
    expect(submitButton).toBeEnabled();
    await user.click(submitButton);

    expect(submitButton).toBeDisabled();
    expect(screen.getByRole('status', { hidden: true })).toBeInTheDocument();

    resolveRecommend({
      results: [
        makeRecommendation({ position: 'Backend Developer (Python)', rerank_score: 0.9 }),
        makeRecommendation({ position: 'QA Engineer', company: 'Globex', rerank_score: 0.2, keyword: 'QA' }),
      ],
    });

    await waitFor(() => expect(screen.getByText('Backend Developer (Python)')).toBeInTheDocument());
    expect(screen.getByText('QA Engineer')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByText('0%')).toBeInTheDocument();
    expect(screen.getAllByText('Python').length).toBeGreaterThan(0);
    expect(screen.getAllByText('2y').length).toBeGreaterThan(0);

    // Form stays mounted and usable after results render.
    expect(screen.getByText(/processed in memory/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /recommend jobs/i })).toBeEnabled();
  });

  it('shows the empty state for a valid response with zero results', async () => {
    const fakeRecommend = vi.fn(async (): Promise<RecommendResponse> => ({ results: [] }));
    render(<App recommend={fakeRecommend} />);

    const user = await selectFile(makeCvFile());
    await user.click(screen.getByRole('button', { name: /recommend jobs/i }));

    await waitFor(() => expect(screen.getByText(/no matching jobs found/i)).toBeInTheDocument());
  });

  it('shows a distinct message on rate limit (429) and allows retry', async () => {
    const fakeRecommend = vi
      .fn()
      .mockRejectedValueOnce(new RateLimitError())
      .mockResolvedValueOnce({ results: [makeRecommendation()] });

    render(<App recommend={fakeRecommend} />);

    const user = await selectFile(makeCvFile());
    await user.click(screen.getByRole('button', { name: /recommend jobs/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/too many requests/i));

    await user.click(screen.getByRole('button', { name: /recommend jobs/i }));

    await waitFor(() => expect(screen.getByText('Backend Developer (Python)')).toBeInTheDocument());
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows a generic message for a timeout/5xx/network failure', async () => {
    const fakeRecommend = vi.fn().mockRejectedValueOnce(new RecommendError());
    render(<App recommend={fakeRecommend} />);

    const user = await selectFile(makeCvFile());
    await user.click(screen.getByRole('button', { name: /recommend jobs/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/something went wrong/i));
  });

  it('shows a distinct message when the server never becomes healthy', async () => {
    const fakeRecommend = vi.fn().mockRejectedValueOnce(new ServerStartupTimeoutError());
    render(<App recommend={fakeRecommend} />);

    const user = await selectFile(makeCvFile());
    await user.click(screen.getByRole('button', { name: /recommend jobs/i }));

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(/taking longer than usual/i));
  });

  it('rejects an oversized file before ever calling recommend', async () => {
    const fakeRecommend = vi.fn();
    render(<App recommend={fakeRecommend} />);

    await selectFile(makeCvFile('huge.pdf', 'application/pdf', 6 * 1024 * 1024));

    expect(screen.getByText(/over 5 mb/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /recommend jobs/i })).toBeDisabled();
    expect(fakeRecommend).not.toHaveBeenCalled();
  });

  it('rejects a wrong-MIME-type file before ever calling recommend', async () => {
    const fakeRecommend = vi.fn();
    render(<App recommend={fakeRecommend} />);

    await selectFile(makeCvFile('cv.exe', 'application/x-msdownload'));

    expect(screen.getByText(/unsupported file type/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /recommend jobs/i })).toBeDisabled();
    expect(fakeRecommend).not.toHaveBeenCalled();
  });
});
