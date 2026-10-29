"use client";

import { Activity, Check, Circle, FileUp, Trash2, WifiOff } from "lucide-react";
import { QuantaLogo } from "@/components/quanta-logo";
import type { BackendStatus } from "@/lib/types";

interface TopBarProps {
  status: BackendStatus | null;
  hasMessages: boolean;
  uploading: boolean;
  working: boolean;
  onClear: () => void;
  onAttach: () => void;
  onClearUploads: () => void;
}

export function TopBar({ status, hasMessages, uploading, working, onClear, onAttach, onClearUploads }: TopBarProps) {
  const online = Boolean(status?.online);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-white/[0.07] bg-zinc-950/65 px-4 py-3 backdrop-blur-2xl sm:px-6">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.045] shadow-inner shadow-white/[0.03]">
            <QuantaLogo size={25} working={working} interactive={false} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold tracking-tight text-zinc-100">Quanta</span>
              <span className="rounded-md border border-sky-300/15 bg-sky-400/[0.08] px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.16em] text-sky-200">RAG</span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-zinc-500">
              {online ? <Circle className={`size-1.5 fill-current ${working ? "text-sky-400" : "text-emerald-400"}`} /> : <WifiOff className="size-3 text-amber-400" />}
              <span>{!online ? "Backend offline" : working ? "Generating response" : "Quantum knowledge assistant"}</span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          <div className="hidden items-center gap-1.5 rounded-full border border-white/[0.07] bg-white/[0.025] px-3 py-1.5 text-[11px] text-zinc-400 md:flex">
            {online ? <Check className="size-3 text-emerald-400" /> : <Activity className="size-3 text-amber-400" />}
            {online ? "System ready" : "Connecting"}
          </div>
          <button type="button" onClick={onAttach} disabled={uploading} title="Upload a document" aria-label="Upload a document" className="inline-flex size-9 items-center justify-center rounded-full text-zinc-400 transition hover:bg-white/[0.07] hover:text-zinc-100 disabled:opacity-50">
            <FileUp className="size-4" />
          </button>
          <button type="button" onClick={onClearUploads} title="Clear uploaded documents" aria-label="Clear uploaded documents" className="inline-flex size-9 items-center justify-center rounded-full text-zinc-400 transition hover:bg-white/[0.07] hover:text-zinc-100">
            <Trash2 className="size-4" />
          </button>
          {hasMessages && <button type="button" onClick={onClear} className="rounded-full border border-white/10 px-3 py-2 text-xs text-zinc-300 transition hover:border-white/20 hover:bg-white/[0.06] hover:text-white">New chat</button>}
        </div>
      </div>
    </header>
  );
}

export default TopBar;