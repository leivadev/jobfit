import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_CV_PATH = path.join(__dirname, 'fixtures', 'fake-cv.docx');

test('uploads a real CV through the UI and renders recommendations', async ({ page }) => {
  await page.goto('/');

  await page.setInputFiles('input[type="file"]', FIXTURE_CV_PATH);
  await expect(page.getByText('fake-cv.docx')).toBeVisible();

  await page.getByRole('button', { name: /recommend jobs/i }).click();

  // Recommendation positions are real, dynamic data — assert a rendered
  // recommendation list item exists, not specific job text.
  await expect(page.getByRole('listitem').first()).toBeVisible({ timeout: 30_000 });
});
