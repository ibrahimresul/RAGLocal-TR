export type AnswerMode =
  | "extractive"
  | "generative"
  | "fallback_extractive"
  | "no_evidence"
  | "ungrounded";

export interface Source {
  id: number;
  source_name: string;
  chunk_text: string;
  score: number;
  source_type: string | null;
  page_number: number | null;
  chunk_index: number | null;
  context_role: "matched" | "neighbor";
}

export interface Timings {
  retrieval_seconds: number;
  generation_seconds: number;
  total_seconds: number;
}

export interface AskDone {
  question: string;
  answer: string;
  mode: AnswerMode;
  best_score: number;
  source_filter: string | null;
  warning: string | null;
  warning_solution: string | null;
  sources: Source[];
  timings: Timings;
}

export interface ChatMessage {
  id: string;
  question: string;
  askedAt: string;
  status: "streaming" | "done" | "error";
  liveText?: string;
  addedTerms?: string[];
  indexWarning?: { status: string; message: string };
  stage?: "retrieval" | "model" | "generation" | null;
  result?: AskDone;
  errorMessage?: string;
}

export interface SourceSummary {
  source_name: string;
  source_type: string | null;
  chunk_count: number;
  page_count: number;
}

export interface Stats {
  db_path: string;
  total_chunks: number;
  source_count: number;
}

export interface HealthCheck {
  name: string;
  status: "ok" | "warning" | "error";
  message: string;
  solution: string | null;
}

export interface HealthResponse {
  status: "ok" | "warning" | "error";
  checks: HealthCheck[];
}

export interface ModelInfo {
  alias: string;
  default_alias: string;
  source: string;
}

export interface ConfigField {
  name: string;
  value: string | number | boolean;
  description: string;
}

export interface HistoryEntry {
  id: number;
  created_at: string;
  question: string;
  answer: string;
  mode: AnswerMode;
  best_score: number;
  source_filter: string | null;
  sources: Omit<Source, "chunk_text">[];
  timings: Timings;
}

export interface BenchmarkCaseResult {
  name: string;
  question: string;
  retrieval_source: string;
  retrieval_score: number;
  expected_terms: string[];
  valid_answer: boolean;
  term_coverage: number;
  missing_terms: string[];
  answer: string;
}

export interface BenchmarkModelResult {
  model: string;
  status: "ok" | "partial" | "error";
  load_seconds: number;
  error: string | null;
  cases: BenchmarkCaseResult[];
  summary: {
    case_count: number;
    valid_case_count: number;
    average_term_coverage: number;
    cold_generation_seconds: number;
    warm_generation_seconds: number;
    average_selected_generation_seconds: number;
  } | null;
}
