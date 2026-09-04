import re

BOLD_EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_EMPHASIS_RE = re.compile(r"\*(.+?)\*")
BULLET_MARKER_RE = re.compile(r"(?m)^[ \t]*[-*•]\s+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize line endings, strip asterisk emphasis/bullets, collapse whitespace. Pure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = BULLET_MARKER_RE.sub("", text)
    text = BOLD_EMPHASIS_RE.sub(r"\1", text)
    text = ITALIC_EMPHASIS_RE.sub(r"\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def build_job_text(position: str, description: str) -> str:
    """Join position and description into one embedding-ready string. Pure."""
    return f"{position}\n{description}"


def chunk_text(text: str, tokenizer, max_tokens: int = 256) -> list[str]:
    """Split text into non-overlapping windows of at most max_tokens tokens.

    Uses the real model tokenizer rather than a char/word-count heuristic
    (see ADR-0007). Pure given the tokenizer.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    chunks = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + max_tokens]
        chunk = tokenizer.decode(window)
        # decode() isn't a perfect inverse of encode() (e.g. WordPiece
        # normalization can change token boundaries), so a decoded window
        # can re-encode to more than max_tokens. Shrink until it doesn't,
        # so callers who re-tokenize this chunk never silently truncate it.
        while len(window) > 1 and len(tokenizer.encode(chunk, add_special_tokens=False)) > max_tokens:
            window = window[:-1]
            chunk = tokenizer.decode(window)
        chunks.append(chunk)
        start += len(window)
    return chunks
