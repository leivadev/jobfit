SNIPPET_MAX_CHARS = 280


def build_snippet(job_text: str, position: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Build a display snippet from a Job Metadata row's `job_text`.

    `job_text` is `position + " " + description` with all whitespace collapsed
    (see `app/dataset.py::build_metadata`), so the position is stripped back off
    when it's a leading prefix to avoid repeating it verbatim next to the
    `position` field already in the response. Truncates on a word boundary
    rather than mid-word. Pure.
    """
    description = _strip_position_prefix(job_text, position)

    if len(description) <= max_chars:
        return description

    truncated = description[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}…"


def _strip_position_prefix(job_text: str, position: str) -> str:
    if job_text.startswith(position):
        return job_text[len(position):].lstrip()
    return job_text
