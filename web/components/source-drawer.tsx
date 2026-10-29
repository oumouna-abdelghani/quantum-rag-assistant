"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, ChevronRight, ExternalLink, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SourceInfo } from "@/lib/types";

const ORIGINS: Record<string, string> = {
  "qiskit-docs": "Qiskit docs",
  arxiv: "arXiv",
  wikipedia: "Wikipedia",
  local: "Local file",
  upload: "Your upload",
  base: "Knowledge base",
};

interface SourceDrawerProps {
  open: boolean;
  sources: SourceInfo[];
  cited: number[];
  activeIndex: number | null;
  onClose: () => void;
}

function SourceCard({ source, active, cited }: { source: SourceInfo; active: boolean; cited: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const percent = Math.max(0, Math.min(1, source.score)) * 100;
  const path = source.section ? source.section.split(" > ").filter(Boolean) : [];

  return (
    <article
      id={`source-${source.index}`}
      className={cn(
        "rounded-2xl border bg-white/[0.03] p-4 transition-colors",
        active ? "border-indigo-400/60 bg-indigo-500/[0.07] shadow-[0_0_40px_-18px_rgba(129,140,248,0.9)]" : "border-white/10",
      )}
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-indigo-500/20 px-1.5 text-xs font-semibold text-indigo-200 ring-1 ring-indigo-400/30">
          {source.index}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium leading-snug text-zinc-100">{source.title}</h3>
          {path.length > 0 && (
            <p className="mt-1 flex flex-wrap items-center gap-x-1 text-xs text-zinc-500">
              {path.map((part, i) => (
                <span key={i} className="inline-flex items-center gap-1">
                  {i > 0 && <ChevronRight className="size-3 text-zinc-700" />}
                  {part}
                </span>
              ))}
            </p>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className="rounded-full bg-white/[0.06] px-2 py-0.5 text-zinc-400">{ORIGINS[source.origin] ?? source.origin ?? "Source"}</span>
        {source.doc_type && <span className="rounded-full bg-white/[0.06] px-2 py-0.5 uppercase tracking-wide text-zinc-500">{source.doc_type}</span>}
        {source.page ? <span className="rounded-full bg-white/[0.06] px-2 py-0.5 text-zinc-400">page {source.page}</span> : null}
        <span className={cn("rounded-full px-2 py-0.5", cited ? "bg-emerald-500/15 text-emerald-300" : "bg-white/[0.06] text-zinc-500")}>
          {cited ? "cited in answer" : "retrieved"}
        </span>
      </div>

      <div className="mt-3">
        <div className="flex items-center justify-between text-[11px] text-zinc-500">
          <span>Similarity to your question</span>
          <span className="font-mono text-zinc-300">{source.score.toFixed(2)}</span>
        </div>
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/[0.07]">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-cyan-300"
            initial={{ width: 0 }}
            animate={{ width: `${percent}%` }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          />
        </div>
      </div>

      <p className={cn("mt-3 whitespace-pre-wrap text-[13px] leading-6 text-zinc-400", !expanded && "line-clamp-6")}>{source.text}</p>
      <div className="mt-2 flex items-center justify-between">
        <button type="button" onClick={() => setExpanded((v) => !v)} className="text-xs text-indigo-300 hover:text-indigo-200">
          {expanded ? "Show less" : "Show full passage"}
        </button>
        {source.url && (
          <a href={source.url} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-200">
            Open source <ExternalLink className="size-3" />
          </a>
        )}
      </div>
    </article>
  );
}

export function SourceDrawer({ open, sources, cited, activeIndex, onClose }: SourceDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open || activeIndex == null) return;
    const timer = setTimeout(() => {
      document.getElementById(`source-${activeIndex}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 280);
    return () => clearTimeout(timer);
  }, [open, activeIndex]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="scrim"
            className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            key="panel"
            role="dialog"
            aria-modal="true"
            aria-label="Sources"
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-white/10 bg-zinc-950/95 shadow-2xl backdrop-blur-xl"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 36 }}
          >
            <header className="flex items-center justify-between border-b border-white/10 px-5 py-4">
              <div className="flex items-center gap-2.5">
                <BookOpen className="size-4 text-indigo-300" />
                <h2 className="text-sm font-semibold text-zinc-100">Sources</h2>
                <span className="rounded-full bg-white/[0.07] px-2 py-0.5 text-xs text-zinc-400">{sources.length}</span>
              </div>
              <button
                ref={closeRef}
                type="button"
                onClick={onClose}
                aria-label="Close sources"
                className="rounded-lg p-1.5 text-zinc-400 transition hover:bg-white/10 hover:text-white"
              >
                <X className="size-4" />
              </button>
            </header>

            <div className="flex-1 space-y-3 overflow-y-auto p-4">
              {sources.length === 0 ? (
                <p className="p-4 text-sm text-zinc-500">No passages were retrieved for this answer.</p>
              ) : (
                sources.map((source) => (
                  <SourceCard key={source.index} source={source} active={source.index === activeIndex} cited={cited.includes(source.index)} />
                ))
              )}
            </div>

            <footer className="border-t border-white/10 px-5 py-3 text-[11px] leading-5 text-zinc-600">
              Similarity is the cosine between your question and each passage (1.0 = identical meaning).
            </footer>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
