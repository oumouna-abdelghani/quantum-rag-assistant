"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Atom, CheckCircle2, CircuitBoard, Sigma, TriangleAlert, Waypoints } from "lucide-react";
import { ChatInput } from "@/components/chat-input";
import { ChatMessage } from "@/components/chat-message";
import { GreetingLoop } from "@/components/greeting-loop";
import { GridBackground } from "@/components/grid-background";
import { LensTitle } from "@/components/lens-title";
import { QuantaLogo } from "@/components/quanta-logo";
import { SourceDrawer } from "@/components/source-drawer";
import { TopBar } from "@/components/top-bar";
import { streamChat } from "@/lib/stream";
import type { AnswerMode, BackendStatus, HistoryTurn, Message } from "@/lib/types";
import { extractCitations, uid } from "@/lib/utils";

const SUGGESTIONS = [
  { icon: Atom, text: "Explain superposition and measurement" },
  { icon: CircuitBoard, text: "Show me a Bell state circuit in Qiskit" },
  { icon: Waypoints, text: "How does Grover's algorithm work?" },
  { icon: Sigma, text: "What is a variational quantum eigensolver?" },
];
const ACCEPT = ".pdf,.md,.mdx,.txt,.ipynb,.csv";
const NOTICE_PREFIX = /^(Not found in your documents|The knowledge base is empty)/;
interface Notice { kind: "ok" | "error"; text: string }

function buildHistory(messages: Message[]): HistoryTurn[] {
  const turns: HistoryTurn[] = [];
  for (let i = 0; i < messages.length - 1; i++) {
    const question = messages[i];
    const answer = messages[i + 1];
    if (question.role === "user" && answer.role === "assistant" && answer.status === "done" && answer.content) {
      turns.push({ role: "user", content: question.content }, { role: "assistant", content: answer.content });
    }
  }
  return turns.slice(-8);
}

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesRef = useRef<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const streamingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [hybrid, setHybrid] = useState(false);
  const hybridTouched = useRef(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [drawer, setDrawer] = useState<{ messageId: string; index: number | null; open: boolean }>({ messageId: "", index: null, open: false });
  const fileInput = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const commit = useCallback((updater: (previous: Message[]) => Message[]) => {
    const next = updater(messagesRef.current);
    messagesRef.current = next;
    setMessages(next);
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/status", { cache: "no-store" });
      if (!response.ok) throw new Error("Backend not ready");
      const data = (await response.json()) as BackendStatus;
      setStatus(data);
      if (data.online && !hybridTouched.current) {
        hybridTouched.current = true;
        setHybrid(Boolean(data.hybrid_default));
      }
    } catch {
      setStatus({ online: false } as BackendStatus);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, 15000);
    return () => clearInterval(timer);
  }, [refreshStatus]);

  const flash = useCallback((next: Notice) => {
    setNotice(next);
    setTimeout(() => setNotice((current) => (current === next ? null : current)), 5200);
  }, []);

  const onScroll = () => {
    const element = scroller.current;
    if (element) stick.current = element.scrollHeight - element.scrollTop - element.clientHeight < 140;
  };

  useEffect(() => {
    const element = scroller.current;
    if (element && stick.current) element.scrollTop = element.scrollHeight;
  }, [messages]);

  const send = useCallback(async (text: string) => {
    if (streamingRef.current) return;
    const history = buildHistory(messagesRef.current);
    const user: Message = { id: uid(), role: "user", content: text, sources: [], cited: [], status: "done" };
    const assistant: Message = { id: uid(), role: "assistant", content: "", sources: [], cited: [], status: "streaming" };
    commit((previous) => [...previous, user, assistant]);
    stick.current = true;
    streamingRef.current = true;
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const patch = (change: (message: Message) => Message) => commit((previous) => previous.map((message) => message.id === assistant.id ? change(message) : message));
    let buffer = "";
    let frame = 0;
    let mode: AnswerMode | undefined;
    let checkedNotice = false;
    const flush = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      if (!buffer) return;
      const chunk = buffer;
      buffer = "";
      patch((message) => ({ ...message, content: message.content + chunk }));
    };
    try {
      await streamChat({
        query: text, history, hybrid, signal: controller.signal,
        onEvent: (event) => {
          switch (event.type) {
            case "meta": flush(); mode = event.mode; patch((message) => ({ ...message, mode: event.mode, sources: event.sources, bestScore: event.best_score })); break;
            case "token": if (mode === "general" && !checkedNotice) { checkedNotice = true; if (NOTICE_PREFIX.test(event.text)) break; } buffer += event.text; if (!frame) frame = requestAnimationFrame(() => ((frame = 0), flush())); break;
            case "done": flush(); patch((message) => ({ ...message, status: "done", mode: event.mode, cited: event.cited })); break;
            case "error": flush(); patch((message) => ({ ...message, status: "error", error: event.message })); break;
          }
        },
      });
      flush();
      patch((message) => message.status !== "streaming" ? message : message.content ? { ...message, status: "done", cited: extractCitations(message.content, message.sources.length) } : { ...message, status: "error", error: "The connection closed before an answer arrived." });
    } catch (error) {
      flush();
      if (controller.signal.aborted) {
        commit((previous) => previous.filter((message) => !(message.id === assistant.id && !message.content)).map((message) => message.id === assistant.id ? { ...message, status: "done", cited: extractCitations(message.content, message.sources.length) } : message));
      } else { patch((message) => ({ ...message, status: "error", error: error instanceof Error ? error.message : "Request failed." })); }
    } finally { abortRef.current = null; streamingRef.current = false; setStreaming(false); }
  }, [commit, hybrid]);

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const retry = useCallback((messageId: string) => { const list = messagesRef.current; const index = list.findIndex((message) => message.id === messageId); const question = list[index - 1]; if (index < 1 || question?.role !== "user") return; commit((previous) => previous.filter((_, i) => i !== index && i !== index - 1)); void send(question.content); }, [commit, send]);
  const clearChat = useCallback(() => { abortRef.current?.abort(); commit(() => []); setDrawer((current) => ({ ...current, open: false })); }, [commit]);
  const openSource = useCallback((messageId: string, index: number | null) => setDrawer({ messageId, index, open: true }), []);
  const closeDrawer = useCallback(() => setDrawer((current) => ({ ...current, open: false })), []);
  const drawerMessage = messages.find((message) => message.id === drawer.messageId);

  const uploadFile = useCallback(async (file: File) => {
    if (uploading) return;
    setUploading(true);
    try {
      const form = new FormData(); form.append("file", file, file.name);
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data?.error ?? `Upload failed (${response.status}).`);
      flash({ kind: "ok", text: `Indexed "${data.name}". Ask about it now.` }); void refreshStatus();
    } catch (error) { flash({ kind: "error", text: error instanceof Error ? error.message : "Upload failed." }); } finally { setUploading(false); }
  }, [flash, refreshStatus, uploading]);

  const clearUploads = useCallback(async () => {
    if (!window.confirm("Delete uploads?")) return;
    try {
      const response = await fetch("/api/upload", { method: "DELETE" });
      if (!response.ok) throw new Error("Delete failed.");
      flash({ kind: "ok", text: "Your uploads were deleted." }); void refreshStatus();
    } catch (error) { flash({ kind: "error", text: "Could not delete uploads." }); }
  }, [flash, refreshStatus]);

  const attach = useCallback(() => fileInput.current?.click(), []);
  const toggleHybrid = useCallback(() => { hybridTouched.current = true; setHybrid((value) => !value); }, []);
  const empty = messages.length === 0;
  const composer = (variant: "hero" | "dock") => <ChatInput variant={variant} streaming={streaming} uploading={uploading} hybrid={hybrid} onSend={send} onStop={stop} onAttach={attach} onDropFile={uploadFile} onToggleHybrid={toggleHybrid} />;

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden bg-zinc-950 text-zinc-100">
      {/* TEST 1: UNCOMMENT THIS LINE */}
       <GridBackground /> 

      <div aria-hidden className="pointer-events-none absolute inset-0 -z-[1] overflow-hidden">
        <div className="absolute left-1/2 top-[18%] size-[30rem] -translate-x-1/2 rounded-full bg-sky-500/[0.055] blur-[130px]" />
        <div className="absolute right-[-10rem] top-[35%] size-[24rem] rounded-full bg-indigo-500/[0.045] blur-[120px]" />
      </div>

      <input ref={fileInput} type="file" accept={ACCEPT} hidden onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void uploadFile(file); }} />
      <TopBar status={status} hasMessages={!empty} uploading={uploading} working={streaming} onClear={clearChat} onAttach={attach} onClearUploads={clearUploads} />

      <AnimatePresence>
        {notice && <motion.div key={notice.text} initial={{ opacity: 0, y: -12, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -12 }} role="status" className={`fixed left-1/2 top-[4.75rem] z-50 flex max-w-[92vw] -translate-x-1/2 items-center gap-2 rounded-full border px-4 py-2.5 text-sm shadow-xl backdrop-blur-2xl ${notice.kind === "ok" ? "border-emerald-400/25 bg-emerald-950/85 text-emerald-100" : "border-red-400/25 bg-red-950/85 text-red-100"}`}>
          {notice.kind === "ok" ? <CheckCircle2 className="size-4 shrink-0" /> : <TriangleAlert className="size-4 shrink-0" />}<span className="truncate">{notice.text}</span>
        </motion.div>}
      </AnimatePresence>

      {empty ? (
        <main className="relative flex flex-1 flex-col items-center justify-center overflow-y-auto px-4 pb-12 pt-8 sm:pb-16">
          <div className="flex w-full max-w-[760px] flex-col items-center">
            <motion.div initial={{ opacity: 0, scale: 0.88, y: 8 }} animate={{ opacity: 1, scale: 1, y: 0 }} transition={{ type: "spring", stiffness: 150, damping: 18 }} className="relative mb-7">
              <div aria-hidden className="absolute inset-[-35%] -z-10 rounded-full bg-sky-400/[0.12] blur-3xl" />
              <div className="flex size-[68px] items-center justify-center rounded-[22px] border border-white/[0.1] bg-white/[0.045] shadow-[0_12px_50px_-25px_rgba(56,189,248,.35)] sm:size-[76px]">
                
                {/* TEST 2: UNCOMMENT THIS LINE */}
                 <QuantaLogo size={58} working={streaming} /> 
              
              </div>
            </motion.div>
            <div className="animate-rise text-center" style={{ animationDelay: "70ms" }}>
              <p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.32em] text-sky-300/80">Quantum RAG assistant</p>
              
              {/* TEST 3: UNCOMMENT THIS LINE */}
               <LensTitle text="Quanta" className="text-5xl font-semibold tracking-[-0.055em] sm:text-7xl" /> 
            
            </div>
            <p className="animate-rise mt-4 max-w-lg text-center text-sm leading-6 text-zinc-400 sm:text-[15px]" style={{ animationDelay: "140ms" }}>
              Explore quantum theory, algorithms, and Qiskit. Answers are grounded in your documents and include source citations.
            </p>
            <div className="animate-rise mt-9 w-full" style={{ animationDelay: "210ms" }}>
              <div className="mb-2.5 flex items-center justify-between px-1">
                
                {/* TEST 4: UNCOMMENT THIS LINE */}
                 <GreetingLoop /> 
                
                <span className="hidden text-[11px] text-zinc-600 sm:inline">Enter to send · Shift + Enter for a new line</span>
              </div>
              {composer("hero")}
              <p className="mt-3 text-center text-[11px] text-zinc-600">Ask a question or attach a document to expand Quanta’s knowledge.</p>
            </div>
            <div className="animate-rise mt-7 flex w-full flex-wrap justify-center gap-2" style={{ animationDelay: "280ms" }}>
              {SUGGESTIONS.map(({ icon: Icon, text }) => <button key={text} type="button" onClick={() => void send(text)} className="group inline-flex max-w-full items-center gap-2 rounded-full border border-white/[0.085] bg-white/[0.025] px-3.5 py-2.5 text-left text-[12px] text-zinc-400 transition duration-200 hover:border-sky-300/25 hover:bg-sky-300/[0.055] hover:text-zinc-100 sm:text-[13px]"><Icon className="size-3.5 shrink-0 text-sky-300/80 transition group-hover:text-sky-200" /><span>{text}</span></button>)}
            </div>
          </div>
        </main>
      ) : (
        <>
          <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto scroll-smooth">
            <main className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-10">
              {messages.map((message) => <ChatMessage key={message.id} message={message} onOpenSource={openSource} onRetry={retry} />)}
            </main>
          </div>
          <div className="relative z-20 px-4 pb-4 pt-3 sm:px-6">
            <div className="pointer-events-none absolute inset-x-0 -top-12 h-12 bg-gradient-to-t from-zinc-950 via-zinc-950/75 to-transparent" />
            <div className="mx-auto w-full max-w-3xl">{composer("dock")}<p className="mt-2.5 text-center text-[11px] text-zinc-600">Quanta can make mistakes. Check the cited sources.</p></div>
          </div>
        </>
      )}

      {/* TEST 5: UNCOMMENT THIS LINE */}
       <SourceDrawer open={drawer.open && !!drawerMessage} sources={drawerMessage?.sources ?? []} cited={drawerMessage?.cited ?? []} activeIndex={drawer.index} onClose={closeDrawer} /> 

    </div>
  );
}