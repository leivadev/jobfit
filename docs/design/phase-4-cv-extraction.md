# Phase 4: Robust CV extraction

See `docs/design/api-contract.md` (upload handling, size/MIME limits already decided) and `CONTEXT.md` (CV vs. Candidate Profile — this phase produces the Candidate Profile from a CV).

## Already decided

- Formats: PDF, DOCX, plain text.
- Library: `pypdf` for PDF, `python-docx` for DOCX (README stack table). A `pdfplumber` fallback for layout-heavy CVs is deferred — not installed, not yet needed.
- File size limit (e.g. 5 MB) and MIME type validation before parsing (`docs/design/api-contract.md`).
- Never written to disk (`docs/adr/0001-no-database.md`, privacy requirement).
- Files with no extractable text (scanned/image-only PDFs, empty DOCX, blank/whitespace-only plain text) are rejected with a clear error (`CvExtractionError`) for the MVP; OCR is out of scope.
- Password-protected PDFs: an empty-password decrypt is attempted first, so owner-password-only PDFs (no real user password) still extract. A PDF with a genuine user password is rejected with a distinct password-protected `CvExtractionError` message.
- Corrupted or malformed PDF/DOCX files are rejected with a tailored `CvExtractionError` message — known library exceptions are caught specifically, with a broad fallback for anything else — rather than crashing or hanging.

## Definition of Done

- [x] Valid PDF and DOCX CVs extract to clean text
- [ ] Oversized files rejected before parsing
- [x] Wrong MIME type rejected before parsing
- [x] Unparseable/corrupted files fail with a clear error, not a crash
- [ ] Unit tests with fixture files covering each case above
