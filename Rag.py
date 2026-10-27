"""
rag.py - Retrieval + augmented generation.

    ask(question)  ->  RagAnswer
      1. retrieve   store.search(): MMR (+ optional BM25 hybrid) + relevance gate
      2. augment    numbered context passages are injected into the prompt
      3. generate   Gemini streams a tutor-style answer with inline [n] citations
      4. extract    the [n] markers are mapped back to the retrieved sources

Two modes, always made explicit to the user:
    grounded   the answer is built from retrieved passages and cites them
    general    nothing relevant was found, so Gemini answers from its own knowledge
               and the answer starts with a fixed notice (added by this code,
               not left to the model)

`on_token` receives text as it streams, so any front-end (CLI now, web UI in
Stage 2) can render the answer live.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from google.genai import types

from chunking import strip_context_header
from config import settings
from embeddings import GeminiError, call_with_retry, get_client
from store import RetrievedChunk, SearchResult, VectorStore

__all__ = ["Source", "RagAnswer", "RagPipeline", "extract_citations"]

GROUNDED_SYSTEM_PROMPT = """You are Quanta, a patient and precise quantum computing tutor. You help students \
understand quantum theory and write correct Qiskit code.

You receive numbered context passages retrieved from a trusted knowledge base (Qiskit documentation, IBM \
Quantum Learning courses, research papers, Wikipedia, and the user's own files).

Rules:
1. Base every factual statement on the passages and cite them inline with their number, like [1] or [2][3], \
right after the sentence they support. Never cite a number that was not provided.
2. If the passages only partly answer the question, say exactly which part is not covered. If you add \
background that is not in the passages, introduce it with "Beyond the documents:" so the reader can tell.
3. Teach: start with the intuition in plain language, then give the formal details (state vectors, gates, \
complexity) and, when useful, a short Qiskit example.
4. Prefer the APIs shown in the passages for code. Qiskit changes between versions, so mention when \
something may be version-dependent. Put code in fenced blocks.
5. Write math in plain text or Unicode (|ψ⟩ = α|0⟩ + β|1⟩), not LaTeX.
6. Be concise: no filler, no repetition of the question."""

GENERAL_SYSTEM_PROMPT = """You are Quanta, a patient and precise quantum computing tutor. You help students \
understand quantum theory and write correct Qiskit code.

No passages from the knowledge base matched this question, so answer from your own general knowledge. Do \
not invent citations or claim the answer comes from the documents. Teach with intuition first, then formal \
details. Qiskit APIs change between versions: say when code may be version-dependent and keep it minimal. \
Write math in plain text or Unicode, not LaTeX. Be concise."""

NOTICE_BELOW_THRESHOLD = "Not found in your documents - answering from general knowledge (no citations).\n\n"
NOTICE_EMPTY_INDEX = ("The knowledge base is empty (run `python ingest.py`, or /upload a file) - "
                      "answering from general knowledge (no citations).\n\n")

class _MidStreamError(Exception):
    """The stream broke after text was already shown; never retried (would duplicate output)."""


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Source:
    index: int                    # the [n] used in the prompt and answer
    title: str
    section: str
    page: Optional[int]
    doc_type: str
    origin: str
    collection: str
    score: float                  # cosine similarity with the question
    text: str                     # chunk body (context header removed)
    url: str = ""
    cited: bool = False

    def label(self) -> str:
        parts = [self.title]
        if self.section and not self.section.lower().startswith(self.title.lower()):
            parts.append(self.section)
        elif self.section:
            parts = [self.section]
        label = " > ".join(parts)
        if self.page:
            label += f" (p.{self.page})"
        return label


@dataclass
class RagAnswer:
    question: str
    mode: str                     # "grounded" | "general"
    answer: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    best_score: float = 0.0
    reason: str = "ok"


def extract_citations(answer: str, source_count: int) -> list[int]:
    """Return the sorted, valid [n] indices cited in `answer` (handles [1], [1][2], [1, 2])."""
    found: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        for number in re.split(r"\s*,\s*", match.group(1)):
            if 1 <= int(number) <= source_count:
                found.add(int(number))
    return sorted(found)


def _to_source(index: int, chunk: RetrievedChunk) -> Source:
    header, body = strip_context_header(chunk.text)
    meta = chunk.metadata
    page = meta.get("page")
    return Source(
        index=index,
        title=str(meta.get("title") or meta.get("file_name") or "Untitled"),
        section=str(meta.get("section") or ""),
        page=int(page) if isinstance(page, (int, float)) else None,
        doc_type=str(meta.get("doc_type") or ""),
        origin=str(meta.get("origin") or meta.get("scope") or ""),
        collection=chunk.collection,
        score=chunk.score,
        text=body,
        url=str(meta.get("url") or ""),
    )


def _format_context(sources: list[Source]) -> str:
    blocks = []
    for source in sources:
        blocks.append(f"[{source.index}] {source.label()}\n{source.text}")
    return "\n\n".join(blocks)


class RagPipeline:
    def __init__(self, store: Optional[VectorStore] = None) -> None:
        self.store = store or VectorStore()

    # ------------------------------------------------------------------

    def _contents(self, question: str, history: Optional[list[dict]], context: Optional[str]) -> list[types.Content]:
        contents: list[types.Content] = []
        turns = (history or [])[-2 * settings.max_history_turns:] if settings.max_history_turns > 0 else []
        for turn in turns:
            role = "user" if turn.get("role") == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=str(turn.get("content", "")))]))
        if context:
            prompt = f"Context passages:\n\n{context}\n\n---\nQuestion: {question}"
        else:
            prompt = question
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return contents

    def _generate(self, system_prompt: str, contents: list[types.Content], on_token: Optional[Callable[[str], None]]) -> str:
        client = get_client()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
        emitted: list[str] = []

        def run() -> None:
            # Retrying is only safe while nothing has been shown to the user yet,
            # so a failure after the first token is turned into a non-retryable error.
            try:
                for piece in client.models.generate_content_stream(
                    model=settings.generation_model, contents=contents, config=config
                ):
                    text = piece.text or ""
                    if text:
                        emitted.append(text)
                        if on_token:
                            on_token(text)
            except Exception as error:  # noqa: BLE001
                if emitted:
                    raise _MidStreamError(str(error)) from error
                raise

        try:
            call_with_retry(run, label="answer generation")
        except GeminiError:
            if emitted:
                return "".join(emitted)
            raise
        return "".join(emitted)

    # ------------------------------------------------------------------

    def retrieve(self, question: str, *, hybrid: Optional[bool] = None) -> SearchResult:
        return self.store.search(question, hybrid=hybrid)

    def ask(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        hybrid: Optional[bool] = None,
    ) -> RagAnswer:
        """Answer `question`. `history` = [{"role": "user"|"assistant", "content": str}, ...]
        is used only to keep the generation conversational; retrieval looks at the question alone
        (query rewriting for follow-ups arrives with memory in Stage 2)."""
        question = question.strip()
        result = self.retrieve(question, hybrid=hybrid)

        if result.grounded:
            sources = [_to_source(i, chunk) for i, chunk in enumerate(result.chunks, start=1)]
            contents = self._contents(question, history, _format_context(sources))
            answer = self._generate(GROUNDED_SYSTEM_PROMPT, contents, on_token)
            cited = extract_citations(answer, len(sources))
            for source in sources:
                source.cited = source.index in cited
            return RagAnswer(question, "grounded", answer, sources, cited, result.best_score, result.reason)

        notice = NOTICE_EMPTY_INDEX if result.reason == "empty_index" else NOTICE_BELOW_THRESHOLD
        if on_token:
            on_token(notice)
        contents = self._contents(question, history, None)
        answer = self._generate(GENERAL_SYSTEM_PROMPT, contents, on_token)
        return RagAnswer(question, "general", notice + answer, [], [], result.best_score, result.reason)