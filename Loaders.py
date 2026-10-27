"""
loaders.py - File extension dispatcher.

`load_file(path)` returns a list of LangChain `Document` objects whatever the
file type. Every Document carries normalized metadata:

    source    stable identifier of the file (posix path, relative to data/raw
              when possible) - used for hash synchronization
    title     human-readable title (frontmatter, PDF metadata, or file name)
    doc_type  pdf | markdown | notebook | text | csv
    page      1-based page number (PDFs only)

Supported: .pdf  .md  .mdx  .txt  .ipynb  .csv
Qiskit documentation ships as .mdx and .ipynb, so both get light cleaning
(frontmatter, JSX components, imports, link URLs) before chunking.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable, Optional

import nbformat
import yaml
from langchain_core.documents import Document

from config import SUPPORTED_EXTENSIONS

__all__ = ["LoaderError", "load_file", "make_source_id", "clean_markdown", "notebook_to_markdown"]


class LoaderError(Exception):
    """Raised when a file cannot be read or contains no usable text."""


# --------------------------------------------------------------------------
# Markdown / MDX cleaning
# --------------------------------------------------------------------------

_FENCE_LINE_RE = re.compile(r"^\s*(```+|~~~+)")
_FRONTMATTER_RE = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_IMPORT_RE = re.compile(r"^[ \t]*import\s+.+?\s+from\s+['\"][^'\"]+['\"];?[ \t]*$", re.MULTILINE)
_JSX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
# Only capitalised tags (<Admonition>, <Tabs>, <IBMVideo .../>): these are MDX
# components. Lower-case HTML is left alone.
_JSX_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9.]*(?:\s[^<>]*)?/?>")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((?:[^)\n]*)\)")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _split_fences(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_code, chunk) segments so cleaning never touches code."""
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_fence = False
    marker = ""

    def flush(is_code: bool) -> None:
        if buffer:
            segments.append((is_code, "\n".join(buffer)))
            buffer.clear()

    for line in text.split("\n"):
        match = _FENCE_LINE_RE.match(line)
        if match:
            token = match.group(1)
            if not in_fence:
                flush(False)
                in_fence, marker = True, token[0]
                buffer.append(line)
                continue
            if token[0] == marker:
                buffer.append(line)
                flush(True)
                in_fence = False
                continue
        buffer.append(line)
    flush(in_fence)
    return segments


def _clean_prose(text: str) -> str:
    text = _JSX_COMMENT_RE.sub("", text)
    text = _IMPORT_RE.sub("", text)
    text = _JSX_TAG_RE.sub("", text)
    text = _IMAGE_RE.sub(lambda m: f"[Figure: {m.group(1).strip()}]" if len(m.group(1).strip()) > 30 else "", text)
    text = _LINK_RE.sub(r"\1", text)
    return html.unescape(text)


def clean_markdown(text: str) -> tuple[str, dict]:
    """Strip frontmatter and MDX noise. Returns (clean_text, frontmatter_dict)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    meta: dict = {}
    match = _FRONTMATTER_RE.match(text)
    if match:
        try:
            parsed = yaml.safe_load(match.group(1))
            if isinstance(parsed, dict):
                meta = parsed
        except yaml.YAMLError:
            pass
        text = text[match.end():]

    parts = [chunk if is_code else _clean_prose(chunk) for is_code, chunk in _split_fences(text)]
    cleaned = "\n".join(parts)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    return _BLANK_RUN_RE.sub("\n\n", cleaned).strip(), meta


# --------------------------------------------------------------------------
# Notebooks
# --------------------------------------------------------------------------

_NOISE_CODE_RE = re.compile(r"^\s*(#.*|[!%]pip\s.*|%matplotlib.*|)$")


def notebook_to_markdown(path: Path) -> str:
    """Convert a .ipynb to one markdown string: markdown cells stay text, code cells
    become fenced blocks, outputs are dropped, install/comment-only cells are skipped."""
    with open(path, encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    language = (
        notebook.metadata.get("language_info", {}).get("name")
        or notebook.metadata.get("kernelspec", {}).get("language")
        or "python"
    )
    blocks: list[str] = []
    for cell in notebook.cells:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = source.strip("\n")
        if not source.strip():
            continue
        if cell.cell_type == "markdown":
            blocks.append(source)
        elif cell.cell_type == "code":
            if all(_NOISE_CODE_RE.match(line) for line in source.split("\n")):
                continue
            blocks.append(f"```{language}\n{source}\n```")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Per-type loaders (each returns Documents; metadata is normalized afterwards)
# --------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError(f"Cannot decode {path.name} as text.")


def _load_markdown(path: Path) -> list[Document]:
    text, frontmatter = clean_markdown(_read_text(path))
    return [Document(page_content=text, metadata={"title": str(frontmatter.get("title", "")), "doc_type": "markdown"})]


def _load_notebook(path: Path) -> list[Document]:
    try:
        raw = notebook_to_markdown(path)
    except Exception as error:  # nbformat raises several different exception types
        raise LoaderError(f"Cannot parse notebook {path.name}: {error}") from error
    text, frontmatter = clean_markdown(raw)
    return [Document(page_content=text, metadata={"title": str(frontmatter.get("title", "")), "doc_type": "notebook"})]


def _load_text(path: Path) -> list[Document]:
    text = _read_text(path).replace("\r\n", "\n").strip()
    return [Document(page_content=text, metadata={"doc_type": "text"})]


_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_INLINE_SPACES_RE = re.compile(r"[ \t]{2,}")


def _load_pdf(path: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as error:
        raise LoaderError(f"Cannot read PDF {path.name}: {error}") from error

    title = ""
    documents: list[Document] = []
    for page in pages:
        title = title or str(page.metadata.get("title", "") or "")
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", page.page_content)
        text = _INLINE_SPACES_RE.sub(" ", text).strip()
        if not text:
            continue
        number = int(page.metadata.get("page", 0)) + 1
        documents.append(Document(page_content=text, metadata={"title": title, "doc_type": "pdf", "page": number}))
    return documents


def _load_csv(path: Path) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            rows = CSVLoader(file_path=str(path), encoding=encoding).load()
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise LoaderError(f"Cannot decode CSV {path.name}: {last_error}")
    return [
        Document(page_content=row.page_content, metadata={"doc_type": "csv", "row": int(row.metadata.get("row", 0))})
        for row in rows
        if row.page_content.strip()
    ]


_LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": _load_pdf,
    ".md": _load_markdown,
    ".mdx": _load_markdown,
    ".txt": _load_text,
    ".ipynb": _load_notebook,
    ".csv": _load_csv,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def make_source_id(path: Path, base_dir: Optional[Path] = None) -> str:
    """Stable identifier for a file: path relative to base_dir if inside it, else absolute."""
    path = Path(path).resolve()
    if base_dir is not None:
        try:
            return path.relative_to(Path(base_dir).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _primitive(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def load_file(
    path: str | Path,
    *,
    title: Optional[str] = None,
    base_dir: Optional[Path] = None,
    extra_metadata: Optional[dict] = None,
) -> list[Document]:
    """Load any supported file into normalized Documents.

    Raises LoaderError for unsupported types, unreadable files or files without text.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise LoaderError(f"File not found: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise LoaderError(f"Unsupported file type '{extension}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    documents = [doc for doc in _LOADERS[extension](path) if doc.page_content.strip()]
    if not documents:
        raise LoaderError(f"No extractable text in {path.name} (scanned PDF or empty file?).")

    source = make_source_id(path, base_dir)
    fallback_title = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
    for doc in documents:
        meta = {k: v for k, v in doc.metadata.items() if _primitive(v)}
        meta["title"] = title or meta.get("title") or fallback_title
        meta["source"] = source
        meta["file_name"] = path.name
        meta.update({k: v for k, v in (extra_metadata or {}).items() if _primitive(v)})
        doc.metadata = meta
    return documents