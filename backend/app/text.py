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
