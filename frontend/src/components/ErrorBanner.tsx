export type ErrorVariant = 'rate-limit' | 'timeout' | 'generic';

const MESSAGES: Record<ErrorVariant, string> = {
  'rate-limit': 'Too many requests, try again in a minute.',
  timeout: 'The server is taking longer than usual to start up. Please try again in a moment.',
  generic: 'Something went wrong, try again.',
};

interface ErrorBannerProps {
  variant: ErrorVariant;
}

export function ErrorBanner({ variant }: ErrorBannerProps) {
  return (
    <p role="alert" className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      {MESSAGES[variant]}
    </p>
  );
}
