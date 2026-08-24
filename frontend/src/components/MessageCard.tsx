import type { ChatMessage } from "../types";

const MODE_LABELS: Record<string, string> = {
  extractive: "Doğrudan Alıntı",
  generative: "Sentezlenmiş",
  fallback_extractive: "Yedek Alıntı",
  no_evidence: "Kanıt Yok",
  ungrounded: "Dayanaksız",
};

const STAGE_LABELS: Record<string, string> = {
  retrieval: "arama",
  model: "model hazırlanıyor",
  generation: "yanıt üretiliyor",
};

export function MessageCard({ message }: { message: ChatMessage }) {
  const { result } = message;
  const mode = result?.mode;

  return (
    <div className="message">
      <div className="message__question">{message.question}</div>
      {message.addedTerms && message.addedTerms.length > 0 && (
        <div className="message__rewrite-note">
          takip sorusu · eklenen bağlam: {message.addedTerms.join(", ")}
        </div>
      )}

      <div className="answer-card">
        <div className="answer-card__head">
          {mode && <span className={`mode-badge mode-badge--${mode}`}>{MODE_LABELS[mode]}</span>}
          {message.status === "streaming" && (
            <span className="answer-card__stage">
              <span className="pulse-dot" />
              {message.stage ? STAGE_LABELS[message.stage] : "işleniyor"}
            </span>
          )}
          {result && <span className="answer-card__score mono">skor {result.best_score.toFixed(4)}</span>}
        </div>

        <div className="answer-card__body">
          {message.status === "error"
            ? message.errorMessage
            : result
              ? result.answer
              : message.liveText || "…"}
        </div>

        {result?.warning && (
          <div className="answer-card__warning">{result.warning}</div>
        )}

        {result && result.sources.length > 0 && (
          <div className="evidence">
            <div className="evidence__label">Kanıt · {result.sources.length}</div>
            {result.sources.map((source, index) => (
              <div
                key={source.id}
                className={`evidence-chip ${
                  source.context_role === "neighbor" ? "evidence-chip--neighbor" : ""
                }`}
              >
                <span className="evidence-chip__tag">{index + 1}</span>
                <div>
                  <div className="evidence-chip__meta">
                    <b>{source.source_name}</b>
                    {source.page_number != null && ` · sayfa ${source.page_number}`}
                    {source.chunk_index != null && ` · parça ${source.chunk_index}`}
                    {` · skor ${source.score.toFixed(4)}`}
                    {source.context_role === "neighbor" && " · komşu bağlam"}
                  </div>
                  <div className="evidence-chip__text">{source.chunk_text}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {result && (
          <div className="performance-strip">
            <span>arama {result.timings.retrieval_seconds.toFixed(3)}s</span>
            <span>yanıt {result.timings.generation_seconds.toFixed(3)}s</span>
            <span>toplam {result.timings.total_seconds.toFixed(3)}s</span>
          </div>
        )}
      </div>
    </div>
  );
}
