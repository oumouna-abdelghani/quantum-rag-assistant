"use client";

import { useEffect, useRef, useState } from "react";
import type React from "react";
import { motion } from "framer-motion";
import { ArrowUp, Layers3, LoaderCircle, Paperclip, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { Spotlight } from "@/components/spotlight";

interface ChatInputProps {
  variant: "hero" | "dock";
  streaming: boolean;
  uploading: boolean;
  hybrid: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onAttach: () => void;
  onDropFile: (file: File) => void;
  onToggleHybrid: () => void;
}

/** Floating composer. Enter sends; Shift+Enter inserts a newline. */
export function ChatInput({ variant, streaming, uploading, hybrid, onSend, onStop, onAttach, onDropFile, onToggleHybrid }: ChatInputProps) {
  const [value, setValue] = useState("");
  const [dragging, setDragging] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = area.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);

  const canSend = value.trim().length > 0 && !streaming;
  const submit = () => {
    if (!canSend) return;
    onSend(value.trim());
    setValue("");
  };
  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <motion.div layoutId="composer" layout="position" transition={{ type: "spring", stiffness: 240, damping: 30 }} className="w-full">
      <div
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files?.[0]; if (file) onDropFile(file); }}
      >
        <Spotlight radius={28} className={cn("rounded-[26px]", dragging && "shadow-[0_0_65px_-20px_rgba(56,189,248,0.45)]", variant === "hero" && "shadow-[0_25px_90px_-45px_rgba(59,130,246,0.3)]")}>
          <div className={cn("rounded-[26px] border bg-zinc-900/75 transition-all duration-200", dragging ? "border-sky-300/50" : "border-white/[0.11]", "focus-within:border-sky-300/35 focus-within:ring-4 focus-within:ring-sky-400/[0.055]")}>
            <div className="px-4 pt-4 sm:px-5 sm:pt-5">
              <textarea
                ref={area}
                value={value}
                onChange={(event) => setValue(event.target.value)}
                onKeyDown={onKeyDown}
                rows={1}
                maxLength={4000}
                aria-label="Ask Quanta"
                placeholder={dragging ? "Drop a file to add it to Quanta's knowledge…" : "Ask anything about quantum computing…"}
                className="block max-h-[200px] min-h-[48px] w-full resize-none bg-transparent text-[15px] leading-7 text-zinc-100 outline-none placeholder:text-zinc-500/90"
              />
            </div>
            <div className="flex items-center gap-1.5 px-3 pb-3 pt-2 sm:px-4">
              <button type="button" onClick={onAttach} disabled={uploading} aria-label="Attach a document" title="Add a document to the knowledge base" className="inline-flex size-9 items-center justify-center rounded-full text-zinc-400 transition hover:bg-white/[0.07] hover:text-zinc-100 disabled:opacity-50">
                {uploading ? <LoaderCircle className="size-[17px] animate-spin" /> : <Paperclip className="size-[17px]" />}
              </button>
              <button type="button" onClick={onToggleHybrid} aria-pressed={hybrid} title="Combine semantic search with exact keyword matching (BM25)" className={cn("inline-flex h-9 items-center gap-2 rounded-full border px-3 text-xs font-medium transition", hybrid ? "border-sky-300/30 bg-sky-400/10 text-sky-200" : "border-white/[0.08] text-zinc-400 hover:border-white/15 hover:bg-white/[0.04] hover:text-zinc-200")}>
                <Layers3 className="size-3.5" /> Hybrid <span className={cn("size-1.5 rounded-full", hybrid ? "bg-sky-300" : "bg-zinc-600")} />
              </button>
              <div className="flex-1" />
              {streaming && <span className="mr-2 hidden items-center gap-2 text-xs text-zinc-500 sm:inline-flex"><span className="size-1.5 animate-pulse rounded-full bg-sky-400" />Thinking</span>}
              {streaming ? (
                <button type="button" onClick={onStop} aria-label="Stop generating" className="inline-flex size-9 items-center justify-center rounded-full bg-zinc-100 text-zinc-900 transition hover:bg-white"><Square className="size-3.5 fill-current" /></button>
              ) : (
                <button type="button" onClick={submit} disabled={!canSend} aria-label="Send message" className={cn("inline-flex size-9 items-center justify-center rounded-full transition-all duration-200", canSend ? "bg-sky-400 text-zinc-950 shadow-[0_0_24px_-8px_rgba(56,189,248,.75)] hover:bg-sky-300" : "bg-white/[0.055] text-zinc-600")}><ArrowUp className="size-[18px]" /></button>
              )}
            </div>
          </div>
        </Spotlight>
      </div>
    </motion.div>
  );
}
