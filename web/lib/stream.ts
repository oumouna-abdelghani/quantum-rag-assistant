import type { StreamEvent, HistoryTurn } from "./types";

const API_URL = "http://127.0.0.1:8000";

export interface StreamOptions {
  query: string;
  history?: HistoryTurn[];
  hybrid?: boolean;
  signal?: AbortSignal;
  onEvent: (event: StreamEvent) => void;
}

/** Parse one NDJSON line (also tolerates an SSE "data:" prefix). Returns null if it isn't valid JSON. */
function parseLine(line: string): StreamEvent | null {
  let text = line.trim();
  if (!text || text.startsWith(":")) return null;
  if (text.startsWith("data:")) text = text.slice(5).trim();
  if (!text || text === "[DONE]") return null;
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed.type === "string" ? (parsed as StreamEvent) : null;
  } catch {
    return null;
  }
}

export async function streamChat({ query, history = [], hybrid, signal, onEvent }: StreamOptions): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history, hybrid }),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error; // page.tsx handles the stop button
    onEvent({ type: "error", message: "Cannot reach the backend on port 8000. Is uvicorn running?" });
    return;
  }

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    onEvent({ type: "error", message: `Server error (${response.status})${detail ? `: ${detail}` : ""}` });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handle = (line: string) => {
    const event = parseLine(line);
    if (event) onEvent(event);
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) handle(line);
  }

  buffer += decoder.decode();
  if (buffer.trim()) handle(buffer);
}

export default streamChat;