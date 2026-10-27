"""
ingest.py - Build (or refresh) the quantum_kb knowledge base from sources.yaml.

    python ingest.py                      # everything enabled in sources.yaml
    python ingest.py --only wikipedia     # one or more groups: qiskit arxiv wikipedia local
    python ingest.py --limit 5            # first 5 files per group (cheap test run)
    python ingest.py --dry-run            # load + chunk + count, no embedding, no API key needed
    python ingest.py --prune              # also delete chunks of sources no longer listed
    python ingest.py --reset              # drop quantum_kb first (needed after changing the embedding model/dim)

Pipeline per file:  fetch -> load -> chunk -> hash -> embed (only new chunks) -> Chroma.
Remote sources (Qiskit docs repo, arXiv PDFs, Wikipedia pages) are first
materialized under data/raw/, so every run is reproducible and re-runs never
re-download or re-embed anything that did not change.

Also exposes ingest_uploaded_file(), used by the CLI's /upload command.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import yaml

from chunking import chunk_documents
from config import BASE_COLLECTION, PROJECT_ROOT, SUPPORTED_EXTENSIONS, UPLOAD_COLLECTION, configure_console, settings
from loaders import LoaderError, load_file
from store import SyncReport, VectorStore

GROUPS = ("qiskit", "arxiv", "wikipedia", "local")
_CONFIG_KEYS = {"qiskit": "qiskit_docs", "arxiv": "arxiv", "wikipedia": "wikipedia", "local": "local"}
_HEADERS = {"User-Agent": "quantum-rag-assistant/1.0 (educational project; https://github.com/)"}


@dataclass
class Entry:
    path: Path
    group: str
    title: Optional[str] = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Qiskit documentation (sparse git checkout)
# --------------------------------------------------------------------------

def _git(args: list[str], cwd: Optional[Path] = None) -> None:
    try:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("git is required to fetch the Qiskit docs. Install git, or set qiskit_docs.enabled: false.")
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"git {' '.join(args)} failed:\n{error.stderr}")


def ensure_qiskit_checkout(config: dict, update: bool) -> Path:
    target = settings.raw_dir / "qiskit-documentation"
    if not (target / ".git").exists():
        print("Cloning Qiskit/documentation (sparse, first run only)...")
        target.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--depth", "1", "--filter=blob:none", "--sparse", config["repo"], str(target)])
    elif update:
        print("Updating Qiskit/documentation...")
        _git(["pull", "--ff-only"], cwd=target)
    _git(["sparse-checkout", "set", *config["paths"]], cwd=target)
    return target


def qiskit_entries(config: dict, update: bool) -> list[Entry]:
    root = ensure_qiskit_checkout(config, update)
    extensions = {e.lower() for e in config.get("extensions", [".mdx", ".md", ".ipynb"])}
    excludes = [x.lower() for x in config.get("exclude", [])]
    entries: list[Entry] = []
    for relative in config["paths"]:
        for path in sorted((root / relative).rglob("*")):
            posix = path.as_posix().lower()
            if path.is_file() and path.suffix.lower() in extensions and not any(x in posix for x in excludes):
                entries.append(Entry(path, "qiskit", extra={"origin": "qiskit-docs"}))
    return entries


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------

def download_arxiv(arxiv_id: str, refresh: bool) -> Path:
    target = settings.raw_dir / "arxiv" / f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
    if target.exists() and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading arXiv:{arxiv_id}")
    response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=_HEADERS, timeout=90, allow_redirects=True)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"arXiv:{arxiv_id} did not return a PDF (wrong id?)")
    target.write_bytes(response.content)
    time.sleep(3)  # be polite to arxiv.org
    return target


def arxiv_entries(config: dict, refresh: bool) -> list[Entry]:
    entries: list[Entry] = []
    for paper in config.get("papers", []):
        try:
            path = download_arxiv(str(paper["id"]), refresh)
        except Exception as error:  # noqa: BLE001 - one bad paper must not stop the run
            print(f"  [skip] arXiv:{paper['id']}: {error}")
            continue
        entries.append(Entry(path, "arxiv", paper.get("title"), {
            "origin": "arxiv",
            "url": f"https://arxiv.org/abs/{paper['id']}",
        }))
    return entries


# --------------------------------------------------------------------------
# Wikipedia (MediaWiki API -> markdown file)
# --------------------------------------------------------------------------

_WIKI_HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")
_WIKI_SKIP = {"see also", "references", "external links", "further reading", "notes", "bibliography",
              "sources", "footnotes", "citations"}


def _unwrap_displaystyle(text: str) -> str:
    """Wikipedia plain-text extracts wrap formulas as {\\displaystyle ...}; keep the LaTeX, drop the wrapper."""
    marker = "{\\displaystyle"
    out: list[str] = []
    index = 0
    while True:
        start = text.find(marker, index)
        if start < 0:
            out.append(text[index:])
            break
        out.append(text[index:start])
        depth, cursor = 0, start
        while cursor < len(text):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        inner = " ".join(text[start + len(marker):cursor].split())
        out.append(f"${inner}$" if inner else "")
        index = cursor + 1
    return "".join(out)


def wikipedia_to_markdown(title: str, extract: str) -> str:
    lines = [f"# {title}", ""]
    skip_level: Optional[int] = None
    for line in _unwrap_displaystyle(extract).splitlines():
        match = _WIKI_HEADING_RE.match(line)
        if match:
            level, name = len(match.group(1)), match.group(2)
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and name.lower() in _WIKI_SKIP:
                skip_level = level
            elif skip_level is None:
                lines.append(f"{'#' * level} {name}")
            continue
        if skip_level is None:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def download_wikipedia(page_title: str, refresh: bool) -> tuple[Path, str]:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", page_title).strip("_").lower()
    target = settings.raw_dir / "wikipedia" / f"{slug}.md"
    if target.exists() and not refresh:
        return target, page_title
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading Wikipedia: {page_title}")
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
                "exsectionformat": "wiki", "redirects": 1, "titles": page_title},
        headers=_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise RuntimeError("page not found or empty")
    title = page.get("title", page_title)
    target.write_text(wikipedia_to_markdown(title, page["extract"]), encoding="utf-8")
    time.sleep(0.5)
    return target, title


def wikipedia_entries(config: dict, refresh: bool) -> list[Entry]:
    entries: list[Entry] = []
    for page_title in config.get("pages", []):
        try:
            path, title = download_wikipedia(str(page_title), refresh)
        except Exception as error:  # noqa: BLE001
            print(f"  [skip] Wikipedia '{page_title}': {error}")
            continue
        entries.append(Entry(path, "wikipedia", title, {
            "origin": "wikipedia",
            "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
        }))
    return entries


# --------------------------------------------------------------------------
# Local folders
# --------------------------------------------------------------------------

def local_entries(config: dict) -> list[Entry]:
    entries: list[Entry] = []
    for relative in config.get("paths", []):
        folder = Path(relative)
        folder = folder if folder.is_absolute() else PROJECT_ROOT / folder
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                entries.append(Entry(path, "local", extra={"origin": "local"}))
    return entries


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_entries(config: dict, groups: list[str], *, refresh: bool, update: bool, limit: Optional[int]) -> list[Entry]:
    entries: list[Entry] = []
    for group in groups:
        section = config.get(_CONFIG_KEYS[group]) or {}
        if not section.get("enabled", False):
            print(f"[{group}] disabled in sources.yaml - skipped")
            continue
        print(f"[{group}] collecting files...")
        if group == "qiskit":
            found = qiskit_entries(section, update)
        elif group == "arxiv":
            found = arxiv_entries(section, refresh)
        elif group == "wikipedia":
            found = wikipedia_entries(section, refresh)
        else:
            found = local_entries(section)
        entries.extend(found[:limit] if limit else found)
    return entries


def chunk_entries(entries: list[Entry]) -> list:
    chunks: list = []
    per_group: dict[str, list[int]] = {}
    for entry in entries:
        try:
            docs = load_file(entry.path, title=entry.title, base_dir=settings.raw_dir, extra_metadata=entry.extra)
        except LoaderError as error:
            print(f"  [skip] {entry.path.name}: {error}")
            continue
        pieces = chunk_documents(docs)
        stats = per_group.setdefault(entry.group, [0, 0])
        stats[0] += 1
        stats[1] += len(pieces)
        chunks.extend(pieces)
    for group, (files, count) in per_group.items():
        print(f"  {group:<10} {files:>5} files -> {count:>6} chunks")
    return chunks


def ingest_uploaded_file(store: VectorStore, path: str | Path) -> SyncReport:
    """Index one user file into the `user_uploads` collection (same loader/chunker/hash path)."""
    cleaned = str(path).strip().strip('"').strip("'")
    file_path = Path(cleaned).expanduser()
    if not file_path.is_file():
        raise LoaderError(f"File not found: {file_path}")
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise LoaderError(f"File is {size_mb:.1f} MB; the limit is {settings.max_upload_mb} MB (MAX_UPLOAD_MB).")
    docs = load_file(file_path, extra_metadata={"origin": "upload"})
    chunks = chunk_documents(docs)
    if not chunks:
        raise LoaderError(f"{file_path.name} produced no usable chunks.")
    return store.sync(UPLOAD_COLLECTION, chunks)


def main(argv: Optional[list[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Build the quantum_kb knowledge base.")
    parser.add_argument("--only", nargs="+", choices=GROUPS, help="restrict to these source groups")
    parser.add_argument("--limit", type=int, help="process only the first N files of each group")
    parser.add_argument("--dry-run", action="store_true", help="load and chunk only; no embedding calls")
    parser.add_argument("--prune", action="store_true", help="delete chunks of sources no longer in sources.yaml")
    parser.add_argument("--reset", action="store_true", help="drop the knowledge base before ingesting")
    parser.add_argument("--refresh", action="store_true", help="re-download arXiv/Wikipedia files")
    parser.add_argument("--update", action="store_true", help="git pull the Qiskit docs")
    args = parser.parse_args(argv)

    if args.prune and (args.only or args.limit):
        raise SystemExit("--prune needs a full run: remove --only/--limit, or it would delete the other groups.")

    with open(settings.sources_file, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    groups = args.only or list(GROUPS)
    entries = build_entries(config, groups, refresh=args.refresh, update=args.update, limit=args.limit)
    if not entries:
        print("Nothing to ingest.")
        return 0

    print(f"\nLoading and chunking {len(entries)} files...")
    chunks = chunk_entries(entries)
    unique = len({(c.metadata.get("source"), c.page_content) for c in chunks})
    print(f"Total: {len(chunks)} chunks ({unique} unique)")
    if args.dry_run:
        print("Dry run - nothing embedded or stored.")
        return 0

    store = VectorStore()
    if args.reset:
        store.reset(BASE_COLLECTION)
        print("Knowledge base reset.")

    print("\nSynchronizing with ChromaDB (unchanged chunks are skipped)...")
    report = store.sync(BASE_COLLECTION, chunks, prune_missing_sources=args.prune)
    print(
        f"\nDone. wanted={report.total}  embedded={report.embedded}  "
        f"skipped(unchanged)={report.skipped}  deleted={report.deleted}  "
        f"-> {store.count(BASE_COLLECTION)} chunks in '{BASE_COLLECTION}'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())