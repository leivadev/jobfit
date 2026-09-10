import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  fullyParallel: false,
  use: {
    baseURL: 'http://localhost:5173',
  },
  // Auto-starts both dev servers for the run; reuses ones already running
  // locally instead of double-spawning if a developer already has `pnpm dev`
  // / `uv run poe dev` open.
  webServer: [
    {
      command: 'pnpm dev',
      cwd: __dirname,
      url: 'http://localhost:5173',
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      // No --reload: its file-watcher thread can deadlock with the
      // embedding/rerank pipeline's own multiprocessing fork mid-request.
      // Cold start downloads Artifacts from R2 and loads the models —
      // budget well past a warm reload.
      command: 'uv run uvicorn backend.api.app:app --env-file .env',
      cwd: path.resolve(__dirname, '../backend'),
      url: 'http://localhost:8000/health',
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});
