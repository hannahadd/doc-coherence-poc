from dataclasses import dataclass
import io
import re

from pypdf import PdfReader


@dataclass
class ExtractedPDF:
    text: str
    num_pages: int


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_NOISE_LINES_RE = re.compile(
    r"(all rights reserved|privacy|legal|cookie|accessibility|credits|contact us|sustainability|"
    r"our company|careers|media gallery|cert|nachhaltigkeit|integrity line|terms of use)",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)

    lines = []
    for ln in text.splitlines():
        ln2 = ln.strip()
        if not ln2:
            continue
        if _NOISE_LINES_RE.search(ln2):
            continue
        if len(ln2) <= 2:
            continue
        lines.append(ln2)

    text = "\n".join(lines)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> ExtractedPDF:
    """
    pypdf.PdfReader attend un chemin ou un file-like (avec seek()).
    On wrap donc les bytes dans io.BytesIO.
    """
    stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(stream)

    pages_text = []
    for p in reader.pages:
        try:
            pages_text.append(p.extract_text() or "")
        except Exception:
            pages_text.append("")

    joined = "\n".join(pages_text)
    joined = clean_text(joined)
    return ExtractedPDF(text=joined, num_pages=len(reader.pages))
