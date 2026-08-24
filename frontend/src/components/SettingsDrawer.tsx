import { useEffect, useState } from "react";
import { benchmarkStream, getConfig, getHistory, repeatEntry } from "../api/client";
import type {
  BenchmarkModelResult,
  ConfigField,
  HealthResponse,
  HistoryEntry,
  ModelInfo,
} from "../types";

type Tab = "health" | "config" | "history" | "benchmark";

interface Props {
  health: HealthResponse | null;
  model: ModelInfo | null;
  onClose: () => void;
  onRepeat: (question: string, source: string | null) => void;
}

export function SettingsDrawer({ health, model, onClose, onRepeat }: Props) {
  const [tab, setTab] = useState<Tab>("health");
  const [config, setConfig] = useState<ConfigField[] | null>(null);
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkModelResult[]>([]);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);

  useEffect(() => {
    if (tab === "config" && !config) {
      getConfig().then((response) => setConfig(response.fields));
    }
    if (tab === "history" && !history) {
      getHistory().then((response) => setHistory(response.entries));
    }
  }, [tab, config, history]);

  const runBenchmark = async () => {
    setBenchmarkRunning(true);
    setBenchmarkResults([]);
    try {
      for await (const event of benchmarkStream(model ? [model.alias] : [])) {
        if (event.event === "model_done") {
          setBenchmarkResults((prev) => [...prev, event.data as BenchmarkModelResult]);
        } else if (event.event === "error") {
          window.alert((event.data as { message: string }).message);
        }
      }
    } finally {
      setBenchmarkRunning(false);
    }
  };

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="drawer">
        <div className="drawer__head">
          <span className="drawer__title">Ayarlar &amp; Durum</span>
          <button type="button" className="drawer__close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="drawer__tabs">
          {(["health", "config", "history", "benchmark"] as Tab[]).map((key) => (
            <button
              key={key}
              type="button"
              className={`drawer__tab ${tab === key ? "drawer__tab--active" : ""}`}
              onClick={() => setTab(key)}
            >
              {{ health: "Sağlık", config: "Yapılandırma", history: "Geçmiş", benchmark: "Benchmark" }[key]}
            </button>
          ))}
        </div>
        <div className="drawer__body">
          {tab === "health" && (
            <>
              {model && (
                <div className="health-row">
                  <div className="health-row__body">
                    <div className="health-row__name">Model</div>
                    <div className="health-row__message mono">
                      {model.alias} ({model.source})
                    </div>
                  </div>
                </div>
              )}
              {health?.checks.map((check) => (
                <div className="health-row" key={check.name}>
                  <span className={`status-dot status-dot--${check.status}`} style={{ marginTop: 5 }} />
                  <div className="health-row__body">
                    <div className="health-row__name">{check.name}</div>
                    <div className="health-row__message">{check.message}</div>
                    {check.solution && (
                      <div className="health-row__solution">{check.solution}</div>
                    )}
                  </div>
                </div>
              ))}
              {!health && <div className="empty-hint">Yükleniyor…</div>}
            </>
          )}

          {tab === "config" && (
            <table className="config-table">
              <tbody>
                {config?.map((field) => (
                  <tr key={field.name}>
                    <td>{field.name}</td>
                    <td>
                      {String(field.value)}
                      <div style={{ color: "var(--text-dim)", fontFamily: "var(--font-sans)" }}>
                        {field.description}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === "history" && (
            <>
              {history?.length === 0 && <div className="empty-hint">Henüz soru yok</div>}
              {history?.map((entry) => (
                <div className="health-row" key={entry.id}>
                  <div className="health-row__body">
                    <div className="health-row__name">{entry.question}</div>
                    <div className="health-row__message">
                      {entry.mode} · skor {entry.best_score.toFixed(4)}
                    </div>
                    <button
                      type="button"
                      className="form-btn"
                      style={{ marginTop: 6 }}
                      onClick={async () => {
                        const { question, source } = await repeatEntry(entry.id);
                        onRepeat(question, source);
                        onClose();
                      }}
                    >
                      Tekrarla
                    </button>
                  </div>
                </div>
              ))}
            </>
          )}

          {tab === "benchmark" && (
            <>
              <button
                type="button"
                className="form-btn form-btn--primary"
                onClick={runBenchmark}
                disabled={benchmarkRunning}
              >
                {benchmarkRunning ? "Çalışıyor…" : "Benchmark çalıştır"}
              </button>
              {benchmarkResults.map((result) => (
                <div key={result.model}>
                  <div className="health-row__name" style={{ marginTop: 10 }}>
                    {result.model} — {result.status}
                  </div>
                  {result.summary && (
                    <table className="benchmark-table">
                      <tbody>
                        <tr>
                          <th>Yükleme</th>
                          <td>{result.load_seconds.toFixed(2)}s</td>
                        </tr>
                        <tr>
                          <th>İlk cevap</th>
                          <td>{result.summary.cold_generation_seconds.toFixed(2)}s</td>
                        </tr>
                        <tr>
                          <th>Warm ortalama</th>
                          <td>{result.summary.average_selected_generation_seconds.toFixed(2)}s</td>
                        </tr>
                        <tr>
                          <th>Terim kapsamı</th>
                          <td>{(result.summary.average_term_coverage * 100).toFixed(0)}%</td>
                        </tr>
                      </tbody>
                    </table>
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </>
  );
}
