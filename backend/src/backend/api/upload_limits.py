# Matches the canonical 5 MB limit already enforced client-side in
# frontend/src/lib/validateFile.ts; server-side enforcement here is
# defense-in-depth, since the client check is trivially bypassable.
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
