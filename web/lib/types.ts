export type AnswerMode = "grounded" | "general";

export interface Source {
  index: number;
  title: string;
  section: string;
  page: number | null;
  doc_type: string;
  origin: string;
  collection: string;
  score: number;
  text: string;
  url: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  cited: number[];
  status: "streaming" | "done" | "error";
  mode?: AnswerMode;
  bestScore?: number;
  error?: string;
}

export interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

export type StreamEvent =
  | { type: "meta"; mode: AnswerMode; best_score: number; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; mode: AnswerMode; cited: number[]; best_score: number }
  | { type: "error"; message: string };

export interface BackendStatus {
  online: boolean;
  hybrid_default?: boolean;
}