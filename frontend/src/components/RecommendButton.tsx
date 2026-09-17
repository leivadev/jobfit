interface RecommendButtonProps {
  disabled: boolean;
  loadingText: string | null;
}

export function RecommendButton({ disabled, loadingText }: RecommendButtonProps) {
  const loading = loadingText !== null;
  return (
    <button
      type="submit"
      disabled={disabled || loading}
      className="inline-flex items-center justify-center gap-2 rounded-md bg-purple-700 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
    >
      {loading && (
        <span
          role="status"
          aria-hidden="true"
          className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
        />
      )}
      {loading ? loadingText : 'Recommend jobs'}
    </button>
  );
}
