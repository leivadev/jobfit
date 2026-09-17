import { useState, type FormEvent } from 'react';
import {
  recommend as defaultRecommend,
  RateLimitError,
  ServerStartupTimeoutError,
  type Recommendation,
  type RecommendOptions,
  type RecommendResponse,
  type RecommendStage,
} from './api/client';
import { Dropzone } from './components/Dropzone';
import { EmptyState } from './components/EmptyState';
import { ErrorBanner, type ErrorVariant } from './components/ErrorBanner';
import { PrivacyNotice } from './components/PrivacyNotice';
import { RecommendButton } from './components/RecommendButton';
import { RecommendationList } from './components/RecommendationList';

const LOADING_TEXT: Record<RecommendStage, string> = {
  checking: 'Checking server status…',
  waking: 'Waking up the server, this can take up to a minute…',
  processing: 'Finding recommendations…',
  slow: 'Still finding recommendations, reranking can take a few extra seconds…',
};

type ViewState =
  | { kind: 'idle' }
  | { kind: 'loading'; stage: RecommendStage }
  | { kind: 'recommendations'; recommendations: Recommendation[] }
  | { kind: 'empty' }
  | { kind: 'error'; variant: ErrorVariant };

interface AppProps {
  recommend?: (file: File, options?: RecommendOptions) => Promise<RecommendResponse>;
}

function App({ recommend = defaultRecommend }: AppProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [view, setView] = useState<ViewState>({ kind: 'idle' });

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!selectedFile || view.kind === 'loading') {
      return;
    }

    setView({ kind: 'loading', stage: 'checking' });
    try {
      const response = await recommend(selectedFile, {
        onStage: (stage) => setView({ kind: 'loading', stage }),
      });
      setView(
        response.results.length === 0
          ? { kind: 'empty' }
          : { kind: 'recommendations', recommendations: response.results },
      );
    } catch (error) {
      let variant: ErrorVariant = 'generic';
      if (error instanceof RateLimitError) {
        variant = 'rate-limit';
      } else if (error instanceof ServerStartupTimeoutError) {
        variant = 'timeout';
      }
      setView({ kind: 'error', variant });
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 px-4 py-8">
      <h1 className="text-2xl font-semibold text-gray-900">JobFit</h1>
      <PrivacyNotice />
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <Dropzone selectedFile={selectedFile} onSelectFile={setSelectedFile} onClear={() => setSelectedFile(null)} />
        <RecommendButton
          disabled={!selectedFile}
          loadingText={view.kind === 'loading' ? LOADING_TEXT[view.stage] : null}
        />
      </form>
      <div>
        {view.kind === 'recommendations' && <RecommendationList recommendations={view.recommendations} />}
        {view.kind === 'empty' && <EmptyState />}
        {view.kind === 'error' && <ErrorBanner variant={view.variant} />}
      </div>
    </div>
  );
}

export default App;
