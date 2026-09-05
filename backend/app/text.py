import re

BOLD_EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_EMPHASIS_RE = re.compile(r"\*(.+?)\*")
BULLET_MARKER_RE = re.compile(r"(?m)^[ \t]*[-*•]\s+")
WHITESPACE_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
CYRILLIC_PARENTHETICAL_RE = re.compile(r"\([^()]*[Ѐ-ӿ][^()]*\)")
CYRILLIC_PRESENCE_THRESHOLD = 0.2


def clean_text(text: str) -> str:
    """Normalize line endings, drop Ukrainian/Russian content, strip
    asterisk emphasis/bullets, collapse whitespace. Pure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CYRILLIC_PARENTHETICAL_RE.sub("", text)
    text = _strip_cyrillic_lines(text)
    text = BULLET_MARKER_RE.sub("", text)
    text = BOLD_EMPHASIS_RE.sub(r"\1", text)
    text = ITALIC_EMPHASIS_RE.sub(r"\1", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _strip_cyrillic_lines(text: str) -> str:
    """Drop lines with meaningful Cyrillic content.

    Postings in this dataset are sometimes bilingual: an English section
    followed by a Ukrainian (or Russian) mirror translation of the same
    content — sometimes a full mirrored section, sometimes just a stray
    untranslated header line — often keeping English loanwords (job
    titles, tech stack) untranslated. Only the English content should be
    analyzed/stored/embedded, so lines are dropped once their
    Cyrillic-letter share passes a low threshold rather than requiring an
    outright majority, and line-level (not paragraph-level) granularity
    catches headers glued to an English block by a single line break.

    Titles are also sometimes "English / Ukrainian" pairs on one line
    (e.g. "Account manager / Менеджер по роботі з клієнтами"); each
    slash-separated segment is checked on its own first so the English
    half survives instead of the whole line being dropped.
    """
    lines = [_strip_cyrillic_segment(line) for line in text.split("\n")]
    kept = [line for line in lines if not _has_cyrillic_content(line)]
    return "\n".join(kept)


def _strip_cyrillic_segment(line: str) -> str:
    """Drop slash-separated segments of a line that are themselves Cyrillic.

    No-op unless the split leaves at least one Cyrillic and one non-Cyrillic
    segment — otherwise the line-level check in `_strip_cyrillic_lines`
    already does the right thing (keep an all-English line untouched, drop
    an all-Cyrillic line entirely).
    """
    if "/" not in line:
        return line
    segments = line.split("/")
    kept = [segment for segment in segments if not _has_cyrillic_content(segment)]
    if not kept or len(kept) == len(segments):
        return line
    return "/".join(kept)


def _has_cyrillic_content(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    cyrillic_count = sum(1 for c in letters if CYRILLIC_RE.match(c))
    return cyrillic_count / len(letters) > CYRILLIC_PRESENCE_THRESHOLD


def build_job_text(position: str, description: str) -> str:
    """Join position and description into one embedding-ready string. Pure."""
    return f"{position}\n{description}"
