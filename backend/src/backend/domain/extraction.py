import io
from enum import Enum, auto

import pypdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


class CvFormat(Enum):
    PDF = auto()
    DOCX = auto()
    TEXT = auto()


_MIME_TYPES: dict[str, CvFormat] = {
    "application/pdf": CvFormat.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        CvFormat.DOCX
    ),
    "text/plain": CvFormat.TEXT,
}
_EXTENSIONS: dict[str, CvFormat] = {
    ".pdf": CvFormat.PDF,
    ".docx": CvFormat.DOCX,
    ".txt": CvFormat.TEXT,
}


class UnsupportedCvFormatError(ValueError):
    """Raised when a CV's format can't be determined from MIME type or filename."""


def extract_text(
    content: bytes, filename: str | None = None, mime_type: str | None = None
) -> str:
    """Turn an uploaded CV (PDF, DOCX, or plain text) into raw text.

    `content` is read entirely in memory (io.BytesIO) and never written to
    disk, per ADR-0001 / the privacy requirement. Format is resolved by
    `_detect_format`; MIME type takes precedence over filename extension.
    """
    format_ = _detect_format(filename, mime_type)

    if format_ is CvFormat.PDF:
        return _extract_pdf_text(content)
    if format_ is CvFormat.DOCX:
        return _extract_docx_text(content)
    return _extract_plain_text(content)


def _detect_format(filename: str | None, mime_type: str | None) -> CvFormat:
    if mime_type in _MIME_TYPES:
        return _MIME_TYPES[mime_type]

    if filename:
        _, _, extension = filename.rpartition(".")
        if f".{extension.lower()}" in _EXTENSIONS:
            return _EXTENSIONS[f".{extension.lower()}"]

    raise UnsupportedCvFormatError(
        f"Can't determine CV format from mime_type={mime_type!r}, "
        f"filename={filename!r}"
    )


def _extract_pdf_text(content: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_docx_text(content: bytes) -> str:
    """Extract paragraph and table text, in document order.

    CVs commonly lay out skills/experience in tables, so `document.paragraphs`
    alone would silently drop that content.
    """
    document = Document(io.BytesIO(content))
    lines: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            lines.append(item.text)
        elif isinstance(item, Table):
            for row in item.rows:
                lines.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(lines).strip()


def _extract_plain_text(content: bytes) -> str:
    try:
        return content.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise UnsupportedCvFormatError(
            "Plain-text CV is not valid UTF-8"
        ) from error
