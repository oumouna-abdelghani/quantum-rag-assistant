"use client";

import { memo } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, BookOpen, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/lib/types";
import { CopyButton, Markdown } from "@/components/markdown";
import { QuantaLogo } from "@/components/quanta-logo";

interface ChatMessageProps {
  message: Message;
  onOpenSource: (messageId: string, index: number | null) => void;
  onRetry: (messageId: string) => void;
}

function ChatMessageImpl({ message, onOpenSource, onRetry }: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} className="flex justify-end">
        <div className="max-w-[88%] whitespace-pre-wrap break-words rounded-[24px] rounded-br-md border border-white/[0.07] bg-zinc-800/80 px-4 py-3 text-[15px] leading-7 text-zinc-100 shadow-sm shadow-black/10 sm:px-5">{message.content}</div>
      </motion.div>
    );
  }

  const streaming = message.status === "streaming";
  const waiting = streaming && !message.content;
  const hasSources = message.sources.length > 0;

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="flex gap-3 sm:gap-4">
      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.035] sm:size-9"><QuantaLogo size={27} working={streaming} paused={!streaming} interactive={false} /></div>
      <div className="min-w-0 flex-1 pt-1 text-[15px] leading-7 text-zinc-200">
        {message.mode === "grounded" && hasSources && <button type="button" onClick={() => onOpenSource(message.id, null)} className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-emerald-400/15 bg-emerald-400/[0.06] px-2.5 py-1 text-xs text-emerald-300 transition hover:bg-emerald-400/10"><BookOpen className="size-3" /> Grounded in {message.sources.length} {message.sources.length === 1 ? "passage" : "passages"}</button>}
        {message.mode === "general" && <div title={message.bestScore != null ? `Best similarity found: ${message.bestScore.toFixed(2)}` : undefined} className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-amber-400/15 bg-amber-400/[0.06] px-2.5 py-1 text-xs text-amber-300"><AlertTriangle className="size-3" /> General knowledge — not found in your documents</div>}
        {waiting && <p className="py-1 text-sm text-zinc-500">{hasSources ? `Reading ${message.sources.length} passages…` : "Searching the knowledge base…"}</p>}
        {message.content && <div className="break-words"><Markdown content={message.content} sources={message.sources} onCite={(index) => onOpenSource(message.id, index)} />{streaming && <span className="ml-0.5 inline-block h-4 w-[3px] translate-y-0.5 animate-caret rounded-sm bg-sky-300" />}</div>}
        {message.status === "error" && <div className="mt-3 flex flex-wrap items-center gap-3 rounded-2xl border border-red-500/20 bg-red-500/[0.06] px-3.5 py-3 text-sm text-red-200"><AlertTriangle className="size-4 shrink-0" /><span className="min-w-0 flex-1 break-words">{message.error ?? "Something went wrong."}</span><button type="button" onClick={() => onRetry(message.id)} className="inline-flex items-center gap-1.5 rounded-full bg-white/[0.07] px-3 py-1.5 text-xs font-medium text-white transition hover:bg-white/[0.12]"><RotateCcw className="size-3" /> Try again</button></div>}
        {hasSources && !streaming && <div className="mt-4 flex flex-wrap items-center gap-1.5">{message.sources.map((source) => { const cited = message.cited.includes(source.index); return <button key={source.index} type="button" onClick={() => onOpenSource(message.id, source.index)} title={source.section ? `${source.title} > ${source.section}` : source.title} className={cn("inline-flex max-w-[230px] items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition", cited ? "border-sky-300/20 bg-sky-300/[0.07] text-sky-200 hover:bg-sky-300/[0.12]" : "border-white/[0.08] bg-white/[0.02] text-zinc-500 hover:text-zinc-300")}><span className="font-semibold">{source.index}</span><span className="truncate">{source.title}</span></button>; })}</div>}
        {message.status === "done" && message.content && <div className="-ml-2 mt-2 flex items-center"><CopyButton text={message.content} label="Copy answer" /></div>}
      </div>
    </motion.div>
  );
}

export const ChatMessage = memo(ChatMessageImpl);
