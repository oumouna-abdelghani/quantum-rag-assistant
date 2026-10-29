import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND = (process.env.QUANTA_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

type Turn = { role: "user" | "assistant"; content: string };

function cleanHistory(value: unknown): Turn[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (turn): turn is Turn =>
        !!turn &&
        (turn.role === "user" || turn.role === "assistant") &&
        typeof turn.content === "string" &&
        turn.content.trim().length > 0,
    )
    .slice(-20)
    .map((turn) => ({ role: turn.role, content: turn.content.slice(0, 12000) }));
}

/**
 * Proxy: browser -> Next.js -> Python RAG server (POST /query).
 * The NDJSON stream is piped through untouched, so tokens reach the UI as they are generated.
 */
export async function POST(request: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const query = typeof body.query === "string" ? body.query.trim() : "";
  if (!query) return Response.json({ error: "Please type a question." }, { status: 400 });
  if (query.length > 4000) return Response.json({ error: "Question is too long (4000 characters max)." }, { status: 413 });

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        history: cleanHistory(body.history),
        hybrid: typeof body.hybrid === "boolean" ? body.hybrid : null,
      }),
      signal: request.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (request.signal.aborted) return new Response(null, { status: 499 });
    return Response.json(
      {
        error: `Can't reach the Quanta engine at ${BACKEND}. Start it with: uvicorn main:app --port 8000`,
        detail: error instanceof Error ? error.message : String(error),
      },
      { status: 502 },
    );
  }

  if (!upstream.ok || !upstream.body) {
    let message = `The engine answered with status ${upstream.status}.`;
    try {
      const data = await upstream.json();
      if (typeof data?.detail === "string") message = data.detail;
    } catch {
      /* not JSON */
    }
    return Response.json({ error: message }, { status: upstream.status === 422 ? 400 : upstream.status });
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}