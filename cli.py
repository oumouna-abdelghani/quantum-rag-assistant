"""
cli.py - Interactive terminal chat for the Quantum RAG Assistant.

    python cli.py [--hybrid] [--k 6] [--threshold 0.55]

Commands
    /upload <path>       index a PDF / md / mdx / txt / ipynb / csv into your private collection
    /sources             show every passage retrieved for the last answer (with scores)
    /docs                list indexed documents (knowledge base and uploads)
    /hybrid on|off       toggle dense + BM25 hybrid retrieval
    /threshold <0-1>     change the relevance threshold (try values from /sources scores)
    /clear_uploads       delete everything you uploaded
    /help, /quit
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from config import BASE_COLLECTION, UPLOAD_COLLECTION, configure_console, settings
from embeddings import GeminiError
from ingest import ingest_uploaded_file
from loaders import LoaderError
from rag import RagAnswer, RagPipeline
from store import VectorStore

BANNER = """Quantum RAG Assistant - ask about quantum theory and Qiskit.
Type /help for commands, /quit to leave."""

HELP = __doc__.split("Commands", 1)[1].rstrip()


def print_cited_sources(answer: RagAnswer) -> None:
    if answer.mode != "grounded":
        return
    shown = [s for s in answer.sources if s.cited] or answer.sources
    heading = "Sources" if any(s.cited for s in answer.sources) else "Retrieved (not explicitly cited)"
    print(f"\n{heading}:")
    for source in shown:
        extra = f"  {source.url}" if source.url else ""
        print(f"  [{source.index}] {source.label()}  (score {source.score:.2f}){extra}")


def print_all_sources(answer: RagAnswer | None) -> None:
    if answer is None:
        print("No question asked yet.")
        return
    if answer.mode != "grounded":
        print(f"Last answer used general knowledge (best score {answer.best_score:.2f}). "
              f"If that seems wrong, lower the threshold: /threshold {max(answer.best_score - 0.02, 0):.2f}")
        return
    for source in answer.sources:
        flag = "cited" if source.cited else "retrieved"
        print(f"\n[{source.index}] {source.label()}  - {flag}, score {source.score:.2f}, {source.collection}")
        snippet = " ".join(source.text.split())[:300]
        print(textwrap.indent(textwrap.fill(snippet + ("..." if len(source.text) > 300 else ""), 88), "    "))


def print_docs(store: VectorStore) -> None:
    for name, label in ((BASE_COLLECTION, "Knowledge base"), (UPLOAD_COLLECTION, "Your uploads")):
        rows = store.list_sources(name)
        total = sum(row["chunks"] for row in rows)
        print(f"\n{label}: {len(rows)} documents, {total} chunks")
        for row in rows[:40]:
            print(f"  - {row['title']}  [{row['doc_type']}, {row['chunks']} chunks]")
        if len(rows) > 40:
            print(f"  ... and {len(rows) - 40} more")


def handle_command(line: str, store: VectorStore, last: RagAnswer | None) -> bool:
    """Run a slash command. Returns False when the user asked to quit."""
    command, _, argument = line.partition(" ")
    command, argument = command.lower(), argument.strip()

    if command in ("/quit", "/exit", "/q"):
        return False
    if command == "/help":
        print("Commands" + HELP)
    elif command == "/upload":
        if not argument:
            print("Usage: /upload <path to file>")
        else:
            try:
                print("Indexing...")
                report = ingest_uploaded_file(store, argument)
                print(f"Indexed: {report.total} chunks ({report.embedded} new, {report.skipped} unchanged). "
                      f"Ask a question about it now.")
            except (LoaderError, GeminiError, RuntimeError) as error:
                print(f"Upload failed: {error}")
    elif command == "/sources":
        print_all_sources(last)
    elif command == "/docs":
        print_docs(store)
    elif command == "/hybrid":
        if argument.lower() in ("on", "off"):
            settings.hybrid_enabled = argument.lower() == "on"
        print(f"Hybrid retrieval: {'ON' if settings.hybrid_enabled else 'OFF'}")
    elif command == "/threshold":
        try:
            value = float(argument)
            if not 0.0 <= value <= 1.0:
                raise ValueError
            settings.relevance_threshold = value
        except ValueError:
            print("Usage: /threshold <number between 0 and 1>")
        print(f"Relevance threshold: {settings.relevance_threshold:.2f}")
    elif command == "/clear_uploads":
        store.reset(UPLOAD_COLLECTION)
        print("Uploads deleted.")
    else:
        print(f"Unknown command {command}. Type /help.")
    return True


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Chat with the Quantum RAG Assistant.")
    parser.add_argument("--hybrid", action="store_true", help="start with dense + BM25 hybrid retrieval")
    parser.add_argument("--k", type=int, help="chunks passed to the LLM (default TOP_K)")
    parser.add_argument("--threshold", type=float, help="relevance threshold (default RELEVANCE_THRESHOLD)")
    args = parser.parse_args()

    if args.hybrid:
        settings.hybrid_enabled = True
    if args.k:
        settings.top_k = args.k
    if args.threshold is not None:
        settings.relevance_threshold = args.threshold

    try:
        settings.require_api_key()
    except RuntimeError as error:
        print(error)
        return 1

    store = VectorStore()
    pipeline = RagPipeline(store)
    print(BANNER)
    print(f"Indexed: {store.count(BASE_COLLECTION)} knowledge-base chunks, {store.count(UPLOAD_COLLECTION)} upload chunks | "
          f"hybrid {'ON' if settings.hybrid_enabled else 'OFF'} | threshold {settings.relevance_threshold:.2f}")
    if store.count(BASE_COLLECTION) == 0:
        print("The knowledge base is empty. Run `python ingest.py` first (or /upload a file).")

    history: list[dict] = []
    last: RagAnswer | None = None

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(line, store, last):
                break
            continue

        print("\nquanta> ", end="", flush=True)
        try:
            last = pipeline.ask(line, history=history, on_token=lambda token: print(token, end="", flush=True))
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        except (GeminiError, RuntimeError) as error:
            print(f"\n[error] {error}")
            continue
        print()
        print_cited_sources(last)
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": last.answer}]

    print("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())