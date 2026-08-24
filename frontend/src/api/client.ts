import type {
  ConfigField,
  HealthResponse,
  HistoryEntry,
  ModelInfo,
  SourceSummary,
  Stats,
} from "../types";

const BASE = "/api";

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<T>;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

export interface SSEEvent {
  event: string;
  data: unknown;
}

/**
 * sse-starlette çıktısı `event: X\ndata: {...}\n\n` biçimindedir. `EventSource`
 * POST body gönderemediği için burada `fetch` + `ReadableStream` ile elle
 * ayrıştırılır.
 */
async function* streamSSE(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // sse-starlette satırları CRLF ile bitirir; ayraç "\n\n" değil "\r\n\r\n"
    // olur. Baştan normalize etmek ayraç aramasını ve satır bölmesini basit
    // tutar.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

    let separatorIndex: number;
    while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);

      let eventName = "message";
      let data = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        yield { event: eventName, data: JSON.parse(data) };
      }
    }
  }
}

export function askStream(
  question: string,
  source: string | null,
  signal?: AbortSignal,
) {
  return streamSSE("/ask", { question, source }, signal);
}

export function benchmarkStream(models: string[]) {
  return streamSSE("/benchmark", { models });
}

export async function uploadDocument(file: File): Promise<{ source_name: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${BASE}/documents`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function deleteDocument(sourceName: string): Promise<void> {
  const response = await fetch(`${BASE}/documents/${encodeURIComponent(sourceName)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await readError(response));
}

export async function reindex(): Promise<{ chunk_count: number }> {
  const response = await fetch(`${BASE}/reindex`, { method: "POST" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export const getStats = () => getJSON<Stats>("/stats");
export const getSources = () => getJSON<{ sources: SourceSummary[] }>("/sources");
export const getHealth = () => getJSON<HealthResponse>("/health");
export const getModel = () => getJSON<ModelInfo>("/model");
export const getConfig = () => getJSON<{ fields: ConfigField[] }>("/config");
export const getHistory = () => getJSON<{ entries: HistoryEntry[] }>("/history");
export const getFilter = () => getJSON<{ source: string | null }>("/filter");

export async function setFilter(source: string | null): Promise<{ source: string | null }> {
  const response = await fetch(`${BASE}/filter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

export async function repeatEntry(
  entryId: number,
): Promise<{ question: string; source: string | null }> {
  return getJSON(`/history/${entryId}/repeat`);
}

export async function exportSession(format: "markdown" | "json"): Promise<void> {
  const response = await fetch(`${BASE}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format }),
  });
  if (!response.ok) throw new Error(await readError(response));
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  const filename = match ? match[1] : `session.${format === "markdown" ? "md" : "json"}`;

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
