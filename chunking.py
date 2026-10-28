"""
chunking.py - Two-stage, header-aware splitting.

Stage 1 (markdown, notebooks):  MarkdownHeaderTextSplitter cuts the document at
        # / ## / ### / #### headings, so every piece knows its section path.
Stage 2 (everything):           RecursiveCharacterTextSplitter caps each piece at
        ~CHUNK_SIZE characters with ~CHUNK_OVERLAP overlap, preferring paragraph
        and code-fence boundaries.
PDFs, text files and CSV rows skip stage 1 and go straight to stage 2.

Before embedding, the document title and section path are prepended to every
chunk ("Context: Title > Section > Subsection"), so a chunk about "the oracle"
carries the knowledge that it belongs to Grover's algorithm.

Chunk metadata: source, title, doc_type, section, chunk_index, (page).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import settings

__all__ = ["chunk_documents", "strip_context_header", "CONTEXT_PREFIX"]

CONTEXT_PREFIX = "Context: "
_HEADER_TYPES = {"markdown", "notebook"}
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def _section_path(metadata: dict) -> str:
    return " > ".join(str(metadata[key]).strip() for _, key in _HEADERS if metadata.get(key))


def _build_header(title: str, section: str) -> str:
    title = (title or "").strip()
    if not section:
        return title
    if not title or section.lower().startswith(title.lower()):
        return section
    return f"{title} > {section}"


def strip_context_header(text: str) -> tuple[str, str]:
    """Split a stored chunk into (header, body). Used to show clean bodies in prompts."""
    if text.startswith(CONTEXT_PREFIX) and "\n\n" in text:
        header, body = text.split("\n\n", 1)
        return header[len(CONTEXT_PREFIX):], body
    return "", text


def _markdown_sections(doc: Document) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=True)
    sections: list[Document] = []
    for piece in splitter.split_text(doc.page_content):
        metadata = {k: v for k, v in doc.metadata.items()}
        metadata["section"] = _section_path(piece.metadata)
        sections.append(Document(page_content=piece.page_content, metadata=metadata))
    return sections


def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    min_chunk_chars: Optional[int] = None,
) -> list[Document]:
    """Split loaded Documents into final, embedding-ready chunks."""
    size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    minimum = settings.min_chunk_chars if min_chunk_chars is None else min_chunk_chars

    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=size, chunk_overlap=overlap
    )
    plain_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    counters: dict[str, int] = defaultdict(int)

    for doc in docs:
        if doc.metadata.get("doc_type") in _HEADER_TYPES:
            pieces = markdown_splitter.split_documents(_markdown_sections(doc))
        else:
            piece_metadata = dict(doc.metadata)
            piece_metadata.setdefault("section", "")
            pieces = plain_splitter.split_documents([Document(page_content=doc.page_content, metadata=piece_metadata)])

        for piece in pieces:
            body = piece.page_content.strip()
            if len(body) < minimum:
                continue
            metadata = dict(piece.metadata)
            source = str(metadata.get("source", ""))
            metadata["chunk_index"] = counters[source]
            counters[source] += 1
            header = _build_header(str(metadata.get("title", "")), str(metadata.get("section", "")))
            text = f"{CONTEXT_PREFIX}{header}\n\n{body}" if header else body
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks